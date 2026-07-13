# Run Steps for Local Gemma 4 E2B

This document records the steps used to get the model running locally in this project.

## 1) Open terminal in project

Working folder:

`C:\Users\vaish\OneDrive\Desktop\Google\Gemma4-e2b`

## 2) Install required Python packages (global user install)

Install PyTorch:

```powershell
py -m pip install --upgrade torch
```

Install vision dependency required by Gemma processor:

```powershell
py -m pip install --upgrade torchvision
```

Install audio dependency required by `any-to-any` pipeline:

```powershell
py -m pip install --upgrade librosa
```

## 3) Download/load the model once

Run:

```powershell
python model_download.py
```

Script used:

- `model_download.py`

## 4) Find downloaded model files in cache

Hugging Face cache location used:

`C:\Users\vaish\.cache\huggingface\hub\models--google--gemma-4-E2B-it\snapshots\<snapshot_id>\`

Example main weight file:

`model.safetensors`

## 5) Copy model snapshot into project (optional, done here)

Copied local model folder:

- `gemma-4-E2B-it`

Main weights inside project:

- `gemma-4-E2B-it\model.safetensors`

## 6) Run the local chatbot script

Script created:

- `local_chatbot.py`

Run:

```powershell
python local_chatbot.py
```

Optional args:

```powershell
python local_chatbot.py --model gemma-4-E2B-it --max-new-tokens 256 --temperature 0.7 --top-p 0.9
```

---

## 7) Run the REST API Expert Chatbot

Script created:

- `rest_api_chatbot.py`

Run with defaults:

```powershell
python rest_api_chatbot.py
```

Optional args:

```powershell
python rest_api_chatbot.py --model gemma-4-E2B-it --max-new-tokens 512 --temperature 0.4 --top-p 0.9
```

### What this chatbot knows

| Topic | Details |
|---|---|
| HTTP Methods | GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS, TRACE, CONNECT — safety & idempotency |
| Status Codes | All 1xx–5xx groups with common codes explained |
| REST Constraints | Roy Fielding's 6 constraints (stateless, uniform interface, HATEOAS, etc.) |
| API Design | Naming, versioning, pagination, error payloads, HTTPS |
| Authentication | API Key, Basic Auth, JWT/Bearer, OAuth 2.0, OpenID Connect |
| Test Doubles | Difference between **Mocking**, **Stubbing**, and **Service Virtualization** |
| OpenAPI / Swagger | Spec structure, tooling (Swagger UI, Redoc, Postman) |
| Idempotency Keys | Safe retries for POST (e.g., payments) |

### Built-in shortcut commands (type at the prompt)

| Command | What it asks |
|---|---|
| `methods` | Table of all HTTP methods with safety/idempotency |
| `status`  | Full HTTP status code reference |
| `mock`    | Mocking vs Stubbing vs Service Virtualization |
| `auth`    | Authentication methods comparison |
| `design`  | REST API design best practices |
| `reset`   | Clear conversation history |
| `help`    | Show the command menu |
| `exit`    | Quit the chatbot |

Chat commands:

- `reset` -> clear chat history
- `exit` or `quit` -> stop chat

## 7) Quick validation

Syntax check used:

```powershell
python -m py_compile local_chatbot.py
```

## Notes

- The warning about unauthenticated Hugging Face requests is not a crash. You can set `HF_TOKEN` for faster downloads and higher rate limits.
- On Windows, if you see symlink cache warnings, caching still works; it may just use more disk space.
