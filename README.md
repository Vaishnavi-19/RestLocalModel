# RestLocalModel

A local AI agent server that runs **Google Gemma 4 E2B** on your own machine and exposes it as an **OpenAI-compatible REST API** using the **LangChain** agent framework.

You can talk to it with the standard OpenAI Python SDK, a LangChain chain, or the included interactive chatbots — no cloud API key required.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  agent_server.py                    │
│                                                     │
│  AGENTS registry                                    │
│  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │ rest-api-expert │  │ rest-api-expert-tools    │ │
│  │ code-assistant  │  │ (ReAct + @tool functions)│ │
│  └─────────────────┘  └──────────────────────────┘ │
│                                                     │
│  LangChain layer                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  GGUF model (ChatLlamaCpp)                   │  │
│  │  OR HuggingFace model (ChatHuggingFace)      │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  FastAPI  →  /v1/models  +  /v1/chat/completions   │
└─────────────────────────────────────────────────────┘
         ↑                          ↑
  OpenAI Python SDK          LangChain ChatOpenAI
  (agent_client.py)          (chatbot.py / local_chatbot.py)
```

---

## Project Files

### `model_download.py`
Downloads the Gemma 4 E2B model from HuggingFace into the local cache.
Run this once before starting the server.

```python
# Downloads google/gemma-4-E2B-it to ~/.cache/huggingface/
python model_download.py
```

After downloading, copy or symlink the snapshot folder to `gemma-4-E2B-it/`
in the project root so the server can find it.

---

### `agent_server.py`
The main FastAPI server. Exposes an **OpenAI-compatible REST API** powered by LangChain.

**Key concepts:**
- **Agent registry (`AGENTS` dict)** — each entry defines an agent with its own system prompt, generation parameters, and optional tools. Add a new agent by adding one dict entry — no other changes needed.
- **Auto model selection** — if `gemma4-e2b-q4km.gguf` exists, loads via `ChatLlamaCpp` (fast, CPU-friendly). Otherwise falls back to `ChatHuggingFace` with the full HF model.
- **Tool-calling agents** — any agent with a `"tools"` key becomes a LangGraph ReAct agent that can call Python functions mid-conversation.
- **LCEL chains** — agents without tools use a simple `prompt | llm | StrOutputParser()` chain.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List all registered agents |
| `POST` | `/v1/chat/completions` | Send messages, get a reply |

**Start the server:**
```powershell
python agent_server.py          # runs on http://localhost:8000
```

**Configuration (top of file):**
```python
GGUF_PATH  = "gemma4-e2b-q4km.gguf"   # set to None to force HF mode
MODEL_PATH = "gemma-4-E2B-it"          # HuggingFace model folder
```

---

### `agent_client.py`
Demonstrates two ways to call the agent server from Python.

**Option 1 — OpenAI Python SDK** (zero LangChain knowledge needed on the client):
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="rest-api-expert",
    messages=[{"role": "user", "content": "What is idempotency?"}]
)
print(response.choices[0].message.content)
```

**Option 2 — LangChain `ChatOpenAI`** (compose the agent inside an LCEL pipeline):
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="rest-api-expert", base_url="http://localhost:8000/v1", api_key="local")
chain = ChatPromptTemplate.from_messages([("human", "{question}")]) | llm | StrOutputParser()
print(chain.invoke({"question": "Explain REST constraints."}))
```

Also includes a multi-turn conversation demo and the tool-enabled agent (`rest-api-expert-tools`).

```powershell
python agent_client.py
```

---

### `chatbot.py`
A fully-featured **interactive CLI chatbot** that connects to the running agent server.

**Features:**
- Switch between agents at runtime without restarting
- Maintains full multi-turn conversation history
- Shows which agents have tools (`[tools]` tag)
- Word-wraps long replies for readability

**Commands inside the chatbot:**

| Command | Effect |
|---|---|
| `agents` | List all agents on the server |
| `agent <name>` | Switch to a different agent (history cleared) |
| `reset` | Clear conversation history |
| `help` | Show command menu |
| `exit` / `quit` | Exit |

```powershell
python chatbot.py
```

---

### `local_chatbot.py`
A **lightweight chatbot** that connects to the agent server over HTTP — no model weights are loaded here, so it starts instantly.

Ideal for running on a second terminal (or a different machine on the same network) while `agent_server.py` handles all the heavy lifting.

```powershell
python local_chatbot.py                              # default agent
python local_chatbot.py --agent code-assistant       # pick an agent
python local_chatbot.py --server http://host:8000/v1 # custom server URL
```

---

### `rest_api_chatbot.py`
The **original standalone chatbot** (before the agent server was built). Loads the model directly in-process and provides a REST API expert chatbot with built-in quick commands.

Does not require the server. Useful as a simple single-agent baseline.

```powershell
python rest_api_chatbot.py
```

Built-in shortcuts: `methods`, `status`, `mock`, `auth`, `design`, `reset`, `help`

---

### `quantize.py`
Automates the full **GGUF Q4_K_M quantization pipeline** — converting the HuggingFace model to a 4-bit quantized format that uses ~63% less memory and runs on CPU.

**Pipeline:**

```
gemma-4-E2B-it/          →  gemma4-e2b-f16.gguf  →  gemma4-e2b-q4km.gguf
(HF safetensors, ~4 GB)     (F16 GGUF, 9.3 GB)      (Q4_K_M, 3.4 GB)
```

**What the script handles automatically:**
- Downloads llama.cpp (ZIP fallback if `git` is not installed)
- Installs Python conversion dependencies with `--prefer-binary` (no C compiler needed)
- Converts the model to F16 GGUF using `convert_hf_to_gguf.py`
- Downloads the correct `llama-quantize` binary for your CPU architecture (x64 / arm64)
- Falls back to `llama-cpp-python` if the binary fails (e.g., missing VC++ Runtime)

```powershell
python quantize.py
```

After quantization, `agent_server.py` picks up the GGUF automatically (set `GGUF_PATH` if needed).

---

### `RUN_STEPS.md`
Step-by-step instructions for setting up the project from scratch — installing dependencies, downloading the model, and running the server.

### `QUANTIZE_STEPS.md`
Detailed record of every step taken to quantize the model, including all issues encountered and how they were fixed.

---

## Quick Start

```powershell
# 1. Install dependencies
pip install fastapi uvicorn langchain langchain-huggingface langchain-community langchain-openai openai requests

# 2. Download the model (first time only)
python model_download.py

# 3. (Optional) Quantize to Q4_K_M for faster/lighter inference
python quantize.py

# 4. Start the agent server
python agent_server.py

# 5. In a second terminal — start the chatbot
python chatbot.py
```

---

## Adding a New Agent

Open `agent_server.py` and add an entry to the `AGENTS` dict:

```python
"my-agent": {
    "description": "Shown in GET /v1/models",
    "system_prompt": "You are an expert in ...",
    "max_new_tokens": 512,
    "temperature": 0.4,
    "top_p": 0.9,
    # "tools": [my_tool_fn],   # uncomment to enable ReAct tool-calling
},
```

No other code changes needed — the server picks it up at startup.

---

## Model

| Property | Value |
|---|---|
| Model | Google Gemma 4 E2B Instruction-Tuned |
| HuggingFace ID | `google/gemma-4-E2B-it` |
| Parameters | ~2 billion |
| Quantized size (Q4_K_M) | 3.4 GB |
| Full precision size | ~4 GB |
| Context length | 131,072 tokens |
