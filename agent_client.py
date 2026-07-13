"""
Client examples for the LangChain-powered local agent server.

Two ways to talk to the server:

  1. OpenAI Python SDK  — standard SDK, zero LangChain knowledge required on the client.
  2. LangChain ChatOpenAI — treats the server as a standard OpenAI endpoint inside a
                            LangChain chain, so you can compose it with other runnables.

Install:
    pip install openai langchain langchain-openai

Run the server first:
    python agent_server.py

Then run this file:
    python agent_client.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Option 1 — OpenAI Python SDK  (server is OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

from openai import OpenAI

_openai_client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")


def list_agents() -> None:
    """Print all agents registered on the server."""
    models = _openai_client.models.list()
    print("Available agents:")
    for m in models.data:
        has_tools = getattr(m, "has_tools", False)
        print(f"  {m.id}{'  [tools]' if has_tools else ''}")
    print()


def ask(agent: str, question: str, history: list | None = None) -> str:
    """
    Send a question to an agent and return the reply.
    Pass `history` (list of {"role":…,"content":…} dicts) for multi-turn chat.
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
    response = _openai_client.chat.completions.create(model=agent, messages=messages)
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Option 2 — LangChain ChatOpenAI pointing at the local server
# Lets you compose the agent inside a LangChain LCEL chain.
# ─────────────────────────────────────────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_lc_chain(agent_name: str):
    """
    Build a LangChain LCEL chain that calls a remote agent on the local server.
    Chain: prompt | ChatOpenAI(local server) | StrOutputParser
    """
    llm = ChatOpenAI(
        model=agent_name,
        base_url="http://localhost:8000/v1",
        api_key="local",
        temperature=0,           # overridden by the server-side agent config
    )
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    list_agents()

    # ── OpenAI SDK: single-turn ───────────────────────────────────────────────
    print("=== [OpenAI SDK] rest-api-expert ===")
    print(ask("rest-api-expert", "What is the difference between PUT and PATCH?"))
    print()

    print("=== [OpenAI SDK] code-assistant ===")
    print(ask("code-assistant", "Write a Python function that validates an email address."))
    print()

    # ── OpenAI SDK: tool-enabled agent ───────────────────────────────────────
    print("=== [OpenAI SDK] rest-api-expert-tools (ReAct + tools) ===")
    print(ask("rest-api-expert-tools", "Is the DELETE method safe? And what does status 409 mean?"))
    print()

    # ── OpenAI SDK: multi-turn conversation ───────────────────────────────────
    print("=== [OpenAI SDK] Multi-turn ===")
    history: list = []
    q1 = "What HTTP status code should I return after a successful POST that creates a resource?"
    a1 = ask("rest-api-expert", q1, history)
    history += [{"role": "user", "content": q1}, {"role": "assistant", "content": a1}]
    print(f"User : {q1}\nAgent: {a1}\n")

    q2 = "And what Location header should I include?"
    a2 = ask("rest-api-expert", q2, history)
    print(f"User : {q2}\nAgent: {a2}\n")

    # ── LangChain chain: compose the remote agent in an LCEL pipeline ─────────
    print("=== [LangChain chain] rest-api-expert ===")
    chain = build_lc_chain("rest-api-expert")
    result = chain.invoke({"question": "What are the REST architectural constraints?"})
    print(result)
    print()
