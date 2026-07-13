"""
Local Chatbot — backed by the LangChain REST agent server.

Connects to agent_server.py over HTTP so no model weights are loaded here.
The system prompt and generation settings are managed by the server-side agent.

Usage:
    # 1. Start the server (loads the model once):
    python agent_server.py

    # 2. Start this chatbot (lightweight, no GPU needed here):
    python local_chatbot.py
    python local_chatbot.py --agent code-assistant
    python local_chatbot.py --server http://localhost:8000/v1 --agent rest-api-expert
"""

import argparse

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Local chatbot that connects to the LangChain agent server"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000/v1",
        help="Base URL of the agent server (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--agent",
        default="rest-api-expert",
        help="Agent name to use — must be registered on the server (default: rest-api-expert)",
    )
    return parser.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_server(server_url: str, agent: str) -> None:
    """Raise a friendly error if the server is unreachable or the agent doesn't exist."""
    try:
        resp = requests.get(f"{server_url}/models", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"\n[ERROR] Cannot reach the agent server at {server_url}.\n"
            "  Start it first with:  python agent_server.py\n"
        )

    available = [m["id"] for m in resp.json().get("data", [])]
    if agent not in available:
        raise SystemExit(
            f"\n[ERROR] Agent '{agent}' not found on the server.\n"
            f"  Available agents: {available}\n"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print(f"\nConnecting to {args.server}  (agent: {args.agent}) …")
    _check_server(args.server, args.agent)

    # LangChain ChatOpenAI pointed at the local agent server
    llm = ChatOpenAI(
        model=args.agent,
        base_url=args.server,
        api_key="local",          # any non-empty string; server ignores it
        temperature=0,
    )

    history: list = []            # list of HumanMessage / AIMessage

    print(f"\nChat ready  [agent: {args.agent}]")
    print("Commands: 'reset' (clear history), 'exit' or 'quit' (stop).\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_text:
            continue

        cmd = user_text.lower()

        if cmd in {"exit", "quit"}:
            print("Bye!")
            break

        if cmd == "reset":
            history.clear()
            print("  History cleared.\n")
            continue

        # Send to the agent server via LangChain
        history.append(HumanMessage(content=user_text))

        try:
            response = llm.invoke(history)
            assistant_text = response.content
        except Exception as exc:
            history.pop()         # remove the failed turn from history
            print(f"  [ERROR] {exc}\n")
            continue

        history.append(AIMessage(content=assistant_text))
        print(f"Assistant: {assistant_text}\n")


if __name__ == "__main__":
    main()
