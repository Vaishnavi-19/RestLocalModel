"""
LangChain-powered Local Agent Server
Powered by the local Gemma 4 E2B model.

Uses LangChain as the agent framework:
  • HuggingFacePipeline + ChatHuggingFace  — wraps the HF model  (default)
  • ChatLlamaCpp                           — wraps a GGUF Q4_K_M model
  • LCEL chains (prompt | llm | parser)    — for agents without tools
  • create_react_agent (LangGraph)         — for agents that have tools
  • FastAPI                                — OpenAI-compatible REST surface

─── Install ────────────────────────────────────────────────────────────────────
    pip install fastapi uvicorn langchain langchain-huggingface langchain-community
    pip install llama-cpp-python           # only needed for GGUF mode

─── Quantize first (one-time) ──────────────────────────────────────────────────
    python quantize.py                     # produces gemma4-e2b-q4km.gguf

─── Start the server ───────────────────────────────────────────────────────────
    python agent_server.py

─── Use from the OpenAI Python SDK ─────────────────────────────────────────────
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
    response = client.chat.completions.create(
        model="rest-api-expert",
        messages=[{"role": "user", "content": "What is idempotency?"}]
    )
    print(response.choices[0].message.content)

─── Add a new agent ─────────────────────────────────────────────────────────────
    Add an entry to AGENTS below. Optionally set "tools" to a list of @tool
    functions — the agent will become a ReAct tool-calling agent automatically.
"""

import time
import textwrap
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# HuggingFace pipeline (used to load the raw model once)
from transformers import pipeline as hf_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Example tools  (add your own with the @tool decorator)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_http_status_meaning(code: int) -> str:
    """Return the standard meaning of an HTTP status code."""
    meanings = {
        200: "OK", 201: "Created", 204: "No Content",
        301: "Moved Permanently", 302: "Found", 304: "Not Modified",
        400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
        404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
        422: "Unprocessable Entity", 429: "Too Many Requests",
        500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable",
    }
    return meanings.get(code, "Unknown status code")


@tool
def is_http_method_safe(method: str) -> str:
    """Return whether an HTTP method is safe and/or idempotent."""
    info = {
        "GET":     {"safe": True,  "idempotent": True},
        "HEAD":    {"safe": True,  "idempotent": True},
        "OPTIONS": {"safe": True,  "idempotent": True},
        "PUT":     {"safe": False, "idempotent": True},
        "DELETE":  {"safe": False, "idempotent": True},
        "POST":    {"safe": False, "idempotent": False},
        "PATCH":   {"safe": False, "idempotent": False},
    }
    m = method.upper()
    if m not in info:
        return f"Unknown method: {method}"
    d = info[m]
    return f"{m} — safe: {d['safe']}, idempotent: {d['idempotent']}"


# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry
# key   = the "model" name the client sends in the request
# value = dict with system_prompt, optional generation overrides, optional tools
# ─────────────────────────────────────────────────────────────────────────────

AGENTS: dict = {
    "rest-api-expert": {
        "description": "Expert in REST APIs, HTTP methods, status codes, and API design",
        "system_prompt": textwrap.dedent("""
            You are an expert REST API assistant with deep knowledge of:

            1. HTTP METHODS
               - GET      : Retrieve a resource. Safe, idempotent, cacheable. Never has a body.
               - POST     : Create a new resource or trigger an action. Not safe, not idempotent.
               - PUT      : Replace a resource entirely. Not safe but idempotent.
               - PATCH    : Partially update a resource. Not safe; idempotent only if designed that way.
               - DELETE   : Remove a resource. Not safe but idempotent.
               - HEAD     : Same as GET but returns only headers. Used for existence checks / caching.
               - OPTIONS  : Describe communication options; used in CORS pre-flight requests.
               - TRACE    : Echo the request for diagnostics. Often disabled for security.
               - CONNECT  : Establish a tunnel (e.g., HTTPS through a proxy).

            2. SAFETY vs IDEMPOTENCY
               - Safe      : Does NOT change server state (GET, HEAD, OPTIONS, TRACE).
               - Idempotent: Calling N times has the same effect as once (GET, HEAD, PUT, DELETE, OPTIONS).
               - POST and most PATCH operations are neither safe nor idempotent.

            3. HTTP STATUS CODES
               1xx Informational : 100 Continue, 101 Switching Protocols
               2xx Success       : 200 OK, 201 Created, 202 Accepted, 204 No Content
               3xx Redirection   : 301 Moved Permanently, 302 Found, 304 Not Modified
               4xx Client Error  : 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found,
                                   405 Method Not Allowed, 409 Conflict, 422 Unprocessable Entity, 429 Too Many Requests
               5xx Server Error  : 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable

            4. REST ARCHITECTURAL CONSTRAINTS (Roy Fielding)
               - Client–Server, Stateless, Cacheable, Uniform Interface, Layered System, Code on Demand.

            5. API DESIGN BEST PRACTICES
               - Nouns for resources (/users, /orders/{{id}}). Plural names. Always HTTPS.
               - Versioning: URI (/v1/users) or Accept header.
               - Pagination: ?page=2&limit=20 or cursor-based.
               - RFC 7807 Problem Details for error payloads.
               - HATEOAS: embed hypermedia links so clients discover actions dynamically.

            6. AUTHENTICATION & AUTHORIZATION
               - API Key, Basic Auth, Bearer/JWT, OAuth 2.0, OpenID Connect.

            7. MOCKING vs STUBBING vs SERVICE VIRTUALIZATION
               - Stub: hard-coded fixed response; minimal logic; unit tests.
               - Mock: records calls and verifies expectations; unit/integration tests.
               - Service Virtualization: full stateful simulation, latency, multi-protocol; integration/perf tests.

            8. OPENAPI / SWAGGER
               - OAS 3.x YAML/JSON contract. Tools: Swagger UI, Redoc, Postman, Stoplight.

            Answer clearly with examples and tables where helpful.
        """).strip(),
        "max_new_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.9,
    },

    "code-assistant": {
        "description": "General-purpose coding assistant for any language or framework",
        "system_prompt": textwrap.dedent("""
            You are an expert software engineer with broad knowledge of programming languages,
            frameworks, algorithms, data structures, design patterns, and software best practices.

            When answering:
            - Provide working, idiomatic code examples.
            - Explain the reasoning behind your solution.
            - Point out common pitfalls and how to avoid them.
            - Prefer clarity over cleverness.
            - When multiple approaches exist, briefly compare them.
        """).strip(),
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
    },

    # rest-api-expert with tools — same expert but can look up status codes / methods
    "rest-api-expert-tools": {
        "description": "REST API expert with live HTTP lookup tools",
        "system_prompt": textwrap.dedent("""
            You are an expert REST API assistant. Use the available tools to look up
            accurate HTTP status codes and method properties when asked.
        """).strip(),
        "tools": [get_http_status_meaning, is_http_method_safe],
        "max_new_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.9,
    },

    # ── Template: copy this block to add a new agent ──────────────────────────
    # "my-new-agent": {
    #     "description": "Short description shown in /v1/models",
    #     "system_prompt": "You are ...",
    #     "tools": [my_tool_fn],          # omit if no tools needed
    #     "max_new_tokens": 512,
    #     "temperature": 0.4,
    #     "top_p": 0.9,
    # },
}

# Path to the quantized GGUF file produced by quantize.py.
# Set to None (or leave unset) to use the original HuggingFace model instead.
GGUF_PATH  = "gemma4-e2b-q4km.gguf"   # ← set to None to force HF mode
MODEL_PATH = "gemma-4-E2B-it"          # HF model folder (fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Load the underlying model once at startup, then build LangChain agent chains
# ─────────────────────────────────────────────────────────────────────────────

def _load_chat_model():
    """
    Return a LangChain chat model.

    Priority:
      1. GGUF file (ChatLlamaCpp)     — fast, low memory, CPU-friendly
      2. HuggingFace folder           — full-precision via transformers
    """
    if GGUF_PATH and Path(GGUF_PATH).exists():
        print(f"\nLoading GGUF model: {GGUF_PATH} …")
        try:
            from langchain_community.chat_models import ChatLlamaCpp
            return ChatLlamaCpp(
                model_path=GGUF_PATH,
                n_ctx=4096,
                n_gpu_layers=-1,   # -1 = offload all layers to GPU if available
                n_batch=512,
                verbose=False,
            )
        except ImportError:
            raise SystemExit(
                "[ERROR] llama-cpp-python is not installed.\n"
                "  Run:  pip install llama-cpp-python"
            )

    # ── Fallback: HuggingFace pipeline ───────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model folder not found: {Path(MODEL_PATH).resolve()}")
    print(f"\nLoading HuggingFace model: {MODEL_PATH} …")
    _hf_pipe = hf_pipeline(
        task="text-generation",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=_hf_pipe))


_chat_model = _load_chat_model()


def _build_agent_chain(cfg: dict):
    """
    Build a LangChain runnable for one agent entry.

    • No tools  → simple LCEL chain: prompt | chat_model | StrOutputParser()
    • Has tools → LangGraph ReAct agent (create_react_agent from langgraph.prebuilt)
    """
    tools = cfg.get("tools", [])

    if tools:
        # state_modifier injects the system prompt into every graph run
        return create_react_agent(
            _chat_model,
            tools,
            prompt=cfg["system_prompt"],
        )

    # Plain LCEL chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg["system_prompt"]),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | _chat_model | StrOutputParser()


# 3. Build every agent's chain once at startup
_CHAINS: dict = {}
for _agent_id, _cfg in AGENTS.items():
    print(f"  Building chain for agent: {_agent_id}")
    _CHAINS[_agent_id] = _build_agent_chain(_cfg)

print("Model ready — server starting.\n")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Local Agent Server",
    description="OpenAI-compatible REST API backed by a local Gemma model",
    version="1.0.0",
)


# ─── OpenAI-compatible Pydantic schemas ───────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str                          # maps to an agent name in AGENTS
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False      # accepted for SDK compat; not streamed


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# ─── GET /v1/models — list all registered agents ──────────────────────────────

@app.get("/v1/models")
def list_models():
    """Return all registered agents in the OpenAI models-list format."""
    data = [
        {
            "id": agent_id,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "description": cfg.get("description", ""),
            "has_tools": bool(cfg.get("tools")),
        }
        for agent_id, cfg in AGENTS.items()
    ]
    return {"object": "list", "data": data}


# ─── POST /v1/chat/completions — main inference endpoint ──────────────────────

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    The `model` field selects which LangChain agent to invoke.
    """
    if req.model not in _CHAINS:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Unknown agent '{req.model}'.",
                "available_agents": list(AGENTS.keys()),
            },
        )

    cfg = AGENTS[req.model]
    chain = _CHAINS[req.model]
    has_tools = bool(cfg.get("tools"))

    # Convert OpenAI messages → LangChain messages (skip client system msgs)
    lc_messages = [
        HumanMessage(content=m.content) if m.role == "user"
        else AIMessage(content=m.content)
        for m in req.messages
        if m.role in {"user", "assistant"}
    ]

    if has_tools:
        # LangGraph compiled graph: invoke with {"messages": [...]}
        # and read the last message from the returned state
        result = chain.invoke({"messages": lc_messages})
        last_msg = result["messages"][-1]
        assistant_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    else:
        # LCEL chain expects {"messages": [LangChain message objects]}
        assistant_text = chain.invoke({"messages": lc_messages})

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=str(assistant_text).strip()),
            )
        ],
        usage=Usage(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
