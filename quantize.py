"""
Converts the local Gemma 4 E2B model → GGUF Q4_K_M.

Pipeline:
  1. Clone llama.cpp (if not already present)
  2. Install its Python conversion requirements
  3. Convert gemma-4-E2B-it/ (safetensors) → F16 GGUF  (lossless)
  4. Quantize F16 GGUF → Q4_K_M GGUF  (~75 % size reduction)

The quantization step (4) requires the compiled llama-quantize binary.
This script tries to find it automatically, and prints instructions if it
cannot.

Usage:
    python quantize.py                          # all defaults
    python quantize.py --model gemma-4-E2B-it  # explicit model folder
    python quantize.py --quant-binary path\\to\\llama-quantize.exe
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# Force UTF-8 output so box-drawing characters display on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ─── Defaults ─────────────────────────────────────────────────────────────────

MODEL_DIR      = "gemma-4-E2B-it"
LLAMA_CPP_DIR  = "llama.cpp"
F16_GGUF       = "gemma4-e2b-f16.gguf"
Q4KM_GGUF      = "gemma4-e2b-q4km.gguf"
QUANT_TYPE     = "Q4_K_M"

# llama.cpp GitHub releases page — grab the latest Windows build automatically
LLAMA_RELEASES_API = (
    "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list, **kwargs) -> None:
    """Print and execute a command, raising on failure."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step(n: int | str, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {n}: {title}")
    print("─" * 60)


# ─── Step 1: get llama.cpp ────────────────────────────────────────────────────

def clone_llama_cpp(llama_dir: str) -> None:
    step(1, "Get llama.cpp")
    if Path(llama_dir).exists():
        print(f"  Already exists: {llama_dir}  (skipping download)")
        return

    # Try git first; fall back to downloading a ZIP from GitHub
    if shutil.which("git"):
        run(["git", "clone", "--depth=1",
             "https://github.com/ggerganov/llama.cpp", llama_dir])
        return

    print("  git not found — downloading ZIP from GitHub …")
    zip_url  = "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip"
    zip_path = Path("llama.cpp-master.zip")

    print(f"  Downloading {zip_url} …")
    urllib.request.urlretrieve(zip_url, zip_path)

    print("  Extracting …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")          # extracts to llama.cpp-master/

    zip_path.unlink()
    Path("llama.cpp-master").rename(llama_dir)
    print(f"  Extracted to: {llama_dir}/")


# ─── Step 2: install Python deps ─────────────────────────────────────────────

# Only the packages actually imported by convert_hf_to_gguf.py.
# We install with --prefer-binary to avoid source builds (no C compiler needed).
_CONVERSION_DEPS = [
    "numpy",
    "sentencepiece",
    "transformers",
    "gguf",
    "protobuf",
    "torch",
    "torchvision",
]

def install_requirements(llama_dir: str) -> None:
    step(2, "Install conversion dependencies  (--prefer-binary, no compiler needed)")
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "--prefer-binary",          # always use pre-built wheels — no source compilation
        *_CONVERSION_DEPS,
    ])


# ─── Step 3: HF → GGUF F16 ───────────────────────────────────────────────────

def convert_to_f16(llama_dir: str, model_dir: str, out_file: str) -> None:
    step(3, f"Convert {model_dir}/ → {out_file}  (F16 GGUF)")
    if Path(out_file).exists():
        print(f"  Already exists: {out_file}  (skipping conversion)")
        return

    # Try the modern script name first, fall back to the old one
    for script_name in ("convert_hf_to_gguf.py", "convert.py"):
        script = Path(llama_dir) / script_name
        if script.exists():
            run([
                sys.executable, str(script),
                model_dir,
                "--outtype", "f16",
                "--outfile", out_file,
            ])
            return

    raise FileNotFoundError(
        "Could not find convert_hf_to_gguf.py or convert.py inside llama.cpp/.\n"
        "Try:  git -C llama.cpp pull  and re-run this script."
    )


# ─── Step 4: quantize F16 → Q4_K_M ──────────────────────────────────────────

def _find_quantize_binary(llama_dir: str) -> Path | None:
    """Search common locations for the llama-quantize binary."""
    candidates = [
        # already on PATH
        shutil.which("llama-quantize"),
        shutil.which("llama-quantize.exe"),
        # built inside the cloned repo (cmake Release build)
        Path(llama_dir) / "build" / "bin" / "Release" / "llama-quantize.exe",
        Path(llama_dir) / "build" / "bin" / "llama-quantize",
        # downloaded alongside this script
        Path("llama-quantize.exe"),
        Path("llama-quantize"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _download_windows_binary(llama_dir: str) -> Path | None:
    """Download the latest pre-built llama.cpp Windows release and extract it."""
    if platform.system() != "Windows":
        return None

    # Map Python's platform.machine() → the substring used in llama.cpp asset names
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch_tag = "x64"
    elif machine in ("arm64", "aarch64"):
        arch_tag = "arm64"
    else:
        print(f"  Unknown CPU architecture: {machine}")
        return None

    print(f"\n  Detected architecture: {machine} → looking for '{arch_tag}' binary …")

    try:
        import json
        with urllib.request.urlopen(LLAMA_RELEASES_API, timeout=10) as r:
            release = json.loads(r.read())

        # Prefer CPU-only build matching the right arch; fall back to any matching arch zip
        def _score(name: str) -> int:
            name = name.lower()
            if arch_tag not in name or not name.endswith(".zip"):
                return 0
            if "cuda" in name or "vulkan" in name or "hip" in name:
                return 1   # GPU build — usable but not preferred
            return 2       # CPU build — preferred

        best = max(
            release.get("assets", []),
            key=lambda a: _score(a["name"]),
            default=None,
        )
        if best is None or _score(best["name"]) == 0:
            print(f"  Could not find a Windows {arch_tag} .zip in the latest release.")
            return None

        asset_url  = best["browser_download_url"]
        asset_name = best["name"]
        print(f"  Downloading {asset_name} …")
        zip_path = Path(asset_name)
        urllib.request.urlretrieve(asset_url, zip_path)

        extract_dir = Path("llama-cpp-bin")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        zip_path.unlink()

        for exe in extract_dir.rglob("llama-quantize.exe"):
            dest = Path("llama-quantize.exe")
            shutil.copy(exe, dest)
            print(f"  Extracted to: {dest}")
            return dest

    except Exception as exc:
        print(f"  Auto-download failed: {exc}")

    return None


def _install_vcruntime() -> bool:
    """
    Download and silently install the Visual C++ 2015-2022 x64 Redistributable.
    Returns True if the install command ran (a UAC prompt may appear).
    """
    print("\n  Downloading Visual C++ 2015-2022 x64 Redistributable …")
    vc_url  = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    vc_path = Path("vc_redist.x64.exe")
    try:
        urllib.request.urlretrieve(vc_url, vc_path)
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return False

    print("  Installing … (a UAC prompt may appear)")
    result = subprocess.run(
        [str(vc_path), "/install", "/quiet", "/norestart"],
        check=False,
    )
    vc_path.unlink(missing_ok=True)
    # 0 = success, 3010 = success but reboot required
    return result.returncode in (0, 3010)


def _quantize_with_python(f16_file: str, out_file: str, quant_type: str) -> None:
    """
    Fallback quantizer using llama-cpp-python pre-built wheels.
    Uses the project's own wheel index which supports more Python versions.
    """
    step("4b", "Install llama-cpp-python (pre-built CPU wheel)")

    # Short TEMP path to avoid Windows MAX_PATH issues during extraction
    short_temp = Path("C:/tmp")
    short_temp.mkdir(parents=True, exist_ok=True)
    install_env = os.environ.copy()
    install_env["TEMP"] = str(short_temp)
    install_env["TMP"]  = str(short_temp)

    # Try the official pre-built wheel index first (covers more Python versions)
    install_cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        "--prefer-binary",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
        "llama-cpp-python",
    ]
    result = subprocess.run(install_cmd, env=install_env)
    if result.returncode != 0:
        # Fall back to default PyPI index
        install_cmd_fallback = [
            sys.executable, "-m", "pip", "install", "-q",
            "--prefer-binary", "llama-cpp-python",
        ]
        subprocess.run(install_cmd_fallback, env=install_env, check=True)

    import importlib, ctypes
    llama_cpp = importlib.import_module("llama_cpp")

    FTYPE = {
        "Q4_0": 2,   "Q4_1": 3,
        "Q5_0": 8,   "Q5_1": 9,
        "Q8_0": 7,
        "Q2_K": 10,
        "Q3_K_S": 11, "Q3_K_M": 12, "Q3_K_L": 13,
        "Q4_K_S": 14, "Q4_K_M": 15,
        "Q5_K_S": 16, "Q5_K_M": 17,
        "Q6_K":   18,
    }
    ftype_val = FTYPE.get(quant_type, 15)

    print(f"\n  Quantizing {f16_file} → {out_file}  ({quant_type} / ftype={ftype_val}) …")
    try:
        params = llama_cpp.llama_model_quantize_default_params()
        params.ftype = ftype_val
        params.nthread = 0          # use all CPU threads
        ret = llama_cpp.llama_model_quantize(
            f16_file.encode("utf-8"),
            out_file.encode("utf-8"),
            ctypes.byref(params),
        )
        if ret != 0:
            raise RuntimeError(f"llama_model_quantize returned error code {ret}")
    except AttributeError as exc:
        raise RuntimeError(
            f"llama-cpp-python API mismatch ({exc}).\n"
            "  Try: pip install --upgrade llama-cpp-python"
        ) from exc


def quantize(llama_dir: str, f16_file: str, out_file: str,
             quant_type: str, quant_binary: str | None) -> None:
    step(4, f"Quantize {f16_file} → {out_file}  ({quant_type})")

    if Path(out_file).exists():
        print(f"  Already exists: {out_file}  (skipping quantization)")
        return

    # Locate the binary
    binary = Path(quant_binary) if quant_binary else _find_quantize_binary(llama_dir)

    if binary is None:
        binary = _download_windows_binary(llama_dir)

    if binary:
        try:
            run([str(binary), f16_file, out_file, quant_type])
        except subprocess.CalledProcessError as exc:
            DLL_NOT_FOUND = 3221225781   # 0xC0000135 — missing VC++ Runtime
            if exc.returncode == DLL_NOT_FOUND:
                print(
                    "\n  Binary failed (missing Visual C++ Runtime — code 0xC0000135)."
                    "\n  Attempting to install the VC++ Redistributable …"
                )
                if _install_vcruntime():
                    print("  Retrying llama-quantize …")
                    try:
                        run([str(binary), f16_file, out_file, quant_type])
                        # success — skip Python fallback
                        if Path(out_file).exists():
                            size_mb = Path(out_file).stat().st_size / 1_048_576
                            print(f"\n  Quantized model saved → {out_file}  ({size_mb:.0f} MB)")
                        return
                    except subprocess.CalledProcessError:
                        pass
                print("  Falling back to llama-cpp-python …")
                _quantize_with_python(f16_file, out_file, quant_type)
            else:
                raise
    else:
        print("\n  No binary found — using llama-cpp-python …")
        _quantize_with_python(f16_file, out_file, quant_type)

    if Path(out_file).exists():
        size_mb = Path(out_file).stat().st_size / 1_048_576
        print(f"\n  Quantized model saved → {out_file}  ({size_mb:.0f} MB)")
        print(
            "\n  Next step: update agent_server.py so it loads the GGUF model.\n"
            f'  Set  GGUF_PATH = "{out_file}"  near the top of agent_server.py.'
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Gemma 4 E2B to GGUF Q4_K_M")
    parser.add_argument("--model",       default=MODEL_DIR,   help="HF model folder")
    parser.add_argument("--llama-cpp",   default=LLAMA_CPP_DIR)
    parser.add_argument("--f16-out",     default=F16_GGUF,    help="F16 GGUF output path")
    parser.add_argument("--q4km-out",    default=Q4KM_GGUF,   help="Q4_K_M GGUF output path")
    parser.add_argument("--quant-type",  default=QUANT_TYPE)
    parser.add_argument("--quant-binary", default=None,
                        help="Explicit path to llama-quantize binary")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"[ERROR] Model folder not found: {args.model}")

    clone_llama_cpp(args.llama_cpp)
    install_requirements(args.llama_cpp)
    convert_to_f16(args.llama_cpp, args.model, args.f16_out)
    quantize(args.llama_cpp, args.f16_out, args.q4km_out,
             args.quant_type, args.quant_binary)


if __name__ == "__main__":
    main()
