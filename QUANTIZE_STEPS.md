# Quantizing Gemma 4 E2B to GGUF Q4_K_M

## Overview

Converts the local HuggingFace model (`gemma-4-E2B-it/`) to a 4-bit quantized GGUF file
that can run on CPU with ~63% less memory than the original.

| File | Size | Notes |
|---|---|---|
| `gemma-4-E2B-it/` | ~4 GB | Original safetensors (HF format) |
| `gemma4-e2b-f16.gguf` | 9.3 GB | Intermediate lossless F16 GGUF |
| **`gemma4-e2b-q4km.gguf`** | **3.4 GB** | Final Q4_K_M quantized model |

---

## Prerequisites

- Python 3.x installed
- Internet access (to download llama.cpp and the quantize binary)
- No C compiler required — everything uses pre-built binaries or pure Python

---

## Step 1 — Get llama.cpp

`quantize.py` downloads the llama.cpp source automatically as a ZIP from GitHub
(falls back to this when `git` is not on PATH):

```
https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip
```

Extracted to: `llama.cpp/`

**Issue encountered:** `git` was not installed on the machine.  
**Fix:** Added a ZIP download fallback using `urllib.request`.

---

## Step 2 — Install Python conversion dependencies

Installs only the packages needed by `convert_hf_to_gguf.py`, using
`--prefer-binary` to avoid any source compilation (no C compiler needed):

```powershell
pip install --prefer-binary numpy sentencepiece transformers gguf protobuf torch torchvision
```

**Issue encountered:** llama.cpp's own `requirements.txt` tried to build numpy
1.26.4 from source, which requires a C compiler that was not installed.  
**Fix:** Replaced the full `requirements.txt` install with a hand-picked list
of packages installed with `--prefer-binary`.

---

## Step 3 — Convert HuggingFace model → F16 GGUF

Runs llama.cpp's conversion script to produce a lossless F16 GGUF:

```powershell
python llama.cpp/convert_hf_to_gguf.py gemma-4-E2B-it `
    --outtype f16 `
    --outfile gemma4-e2b-f16.gguf
```

Output: `gemma4-e2b-f16.gguf` (9.3 GB, 601 tensors, GGUF V3)

---

## Step 4 — Quantize F16 GGUF → Q4_K_M GGUF

### 4a. Download pre-built llama-quantize binary

`quantize.py` calls the GitHub API to find the latest llama.cpp release and
downloads the correct binary for the detected CPU architecture:

```
https://api.github.com/repos/ggerganov/llama.cpp/releases/latest
```

**Issue encountered (first attempt):** Downloaded `arm64` binary on an `AMD64` machine.  
**Fix:** Used `platform.machine()` to detect the architecture and filter assets
by `x64` / `arm64` tag in the asset name.

### 4b. Run llama-quantize

```
llama-quantize.exe gemma4-e2b-f16.gguf gemma4-e2b-q4km.gguf Q4_K_M
```

**Issue encountered:** Binary failed with `WinError 216` / exit code `0xC0000135`
(`STATUS_DLL_NOT_FOUND`) — the Visual C++ 2015-2022 Runtime was not installed.

### 4c. Install Visual C++ Redistributable (automatic)

`quantize.py` downloaded and silently installed the VC++ runtime:

```
https://aka.ms/vs/17/release/vc_redist.x64.exe /install /quiet /norestart
```

The retry of `llama-quantize.exe` still failed after this, so the script fell
through to the Python fallback.

### 4d. Fallback — quantize via llama-cpp-python

Installed `llama-cpp-python` from the official pre-built wheel index:

```powershell
pip install --prefer-binary `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu `
    llama-cpp-python
```

Then called the bundled C library directly via Python:

```python
import llama_cpp, ctypes

params = llama_cpp.llama_model_quantize_default_params()
params.ftype = 15          # Q4_K_M
params.nthread = 0         # use all CPU threads

llama_cpp.llama_model_quantize(
    b"gemma4-e2b-f16.gguf",
    b"gemma4-e2b-q4km.gguf",
    ctypes.byref(params),
)
```

**Issue encountered during install:** `llama-cpp-python` bundles the full
llama.cpp source tree, which has paths exceeding Windows' 260-character
`MAX_PATH` limit, causing pip to fail during extraction.  
**Fix:** Set `TEMP` and `TMP` environment variables to `C:\tmp` (a short path)
before running `pip install`.

---

## Step 5 — Verify the output

```powershell
python -c "
from pathlib import Path
p = Path('gemma4-e2b-q4km.gguf')
print(f'Exists: {p.exists()}, Size: {p.stat().st_size / 1e9:.2f} GB')
"
```

Expected output:
```
Exists: True, Size: 3.43 GB
```

Quantization stats reported by llama-cpp-python:
```
model size  =  8864.87 MiB (16.00 BPW)
quant size  =  3253.99 MiB  (5.87 BPW)
```

---

## Step 6 — Use the GGUF model in the agent server

`agent_server.py` auto-detects the GGUF file. Ensure the path matches:

```python
# agent_server.py (near the top)
GGUF_PATH  = "gemma4-e2b-q4km.gguf"   # ← must match the file produced above
MODEL_PATH = "gemma-4-E2B-it"          # fallback if GGUF not found
```

Start the server — it will load via `ChatLlamaCpp` instead of `ChatHuggingFace`:

```powershell
python agent_server.py
```

---

## Cleanup (optional)

Once the GGUF model is confirmed working, the intermediate F16 file can be
deleted to reclaim 9.3 GB:

```powershell
Remove-Item gemma4-e2b-f16.gguf
```

---

## Issues Encountered & Fixes Summary

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | Unicode print error | Windows CP1252 terminal can't print `─` | `sys.stdout.reconfigure(encoding="utf-8")` |
| 2 | `git` not found | Git not installed | ZIP download fallback via `urllib.request` |
| 3 | numpy build failure | `requirements.txt` tries to compile numpy from source | Hand-pick deps with `--prefer-binary` |
| 4 | Wrong arch binary | Auto-download picked `arm64` on `AMD64` | Use `platform.machine()` to select correct asset |
| 5 | `WinError 216` / `0xC0000135` | Visual C++ Runtime missing | Auto-install VC++ Redistributable; fallback to `llama-cpp-python` |
| 6 | `llama-cpp-python` install fails | Windows MAX_PATH (260 chars) exceeded | Set `TEMP=C:\tmp` before `pip install` |
