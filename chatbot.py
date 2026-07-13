"""
Interactive Chatbot — powered by the local LangChain agent server.

Connects to agent_server.py via LangChain's ChatOpenAI (OpenAI-compatible),
maintains full conversation history, and lets you switch agents at runtime.

─── Start the server first ──────────────────────────────────────────────────
    python agent_server.py

─── Then start the chatbot ──────────────────────────────────────────────────
    python chatbot.py

─── Chatbot commands ────────────────────────────────────────────────────────
    agents          List all available agents on the server
    agent <name>    Switch to a different agent (clears history)
    reset           Clear conversation history (keep current agent)
    help            Show this menu
    exit / quit     Exit
"""

import textwrap

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# ─── Configuration ────────────────────────────────────────────────────────────

SERVER_URL    = "http://localhost:8000/v1"
DEFAULT_AGENT = "rest-api-expert"
WRAP_WIDTH    = 90          # word-wrap the model reply at this column

# ─── Helpers ──────────────────────────────────────────────────────────────────

DIVIDER = "─" * 60

HELP_TEXT = f"""
{DIVIDER}
  Chatbot Commands
{DIVIDER}
  agents          List all agents available on the server
  agent <name>    Switch agent (conversation history is cleared)
  reset           Clear conversation history
  help            Show this menu
  exit / quit     Exit the chatbot
{DIVIDER}
"""


def _fetch_agents() -> list[dict]:
    """GET /v1/models and return the list of agent dicts."""
    resp = requests.get(f"{SERVER_URL}/models", timeout=5)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _build_llm(agent_name: str) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at the local server for the given agent."""
    return ChatOpenAI(
        model=agent_name,
        base_url=SERVER_URL,
        api_key="local",          # any non-empty string works
        temperature=0,
    )


def _wrap(text: str) -> str:
    """Word-wrap a multi-paragraph response for terminal display."""
    lines = []
    for paragraph in text.split("\n"):
        if len(paragraph) <= WRAP_WIDTH:
            lines.append(paragraph)
        else:
            lines.extend(textwrap.wrap(paragraph, width=WRAP_WIDTH))
    return "\n".join(lines)


# ─── Main chatbot loop ────────────────────────────────────────────────────────

def main() -> None:
    # ── Connect to the server ────────────────────────────────────────────────
    print(f"\nConnecting to agent server at {SERVER_URL} …")
    try:
        agents = _fetch_agents()
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] Cannot reach the server.\n"
            "  Start it first with:  python agent_server.py\n"
        )
        return
    except Exception as exc:
        print(f"\n[ERROR] {exc}\n")
        return

    agent_ids = [a["id"] for a in agents]

    if not agent_ids:
        print("[ERROR] Server returned no agents.")
        return

    # ── Choose starting agent ────────────────────────────────────────────────
    current_agent = DEFAULT_AGENT if DEFAULT_AGENT in agent_ids else agent_ids[0]
    llm           = _build_llm(current_agent)
    history: list = []          # list of HumanMessage / AIMessage

    print(f"Connected!  Active agent → {current_agent}")
    print(HELP_TEXT)

    # ── REPL ─────────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Built-in commands ─────────────────────────────────────────────────
        if cmd in {"exit", "quit"}:
            print("Bye!")
            break

        if cmd == "help":
            print(HELP_TEXT)
            continue

        if cmd == "agents":
            print(f"\n{DIVIDER}")
            for a in agents:
                active  = " ◄ active" if a["id"] == current_agent else ""
                tools   = " [tools]"  if a.get("has_tools")       else ""
                print(f"  {a['id']}{tools}{active}")
            print(f"{DIVIDER}\n")
            continue

        if cmd == "reset":
            history.clear()
            print("  Conversation history cleared.\n")
            continue

        if cmd.startswith("agent "):
            requested = user_input[6:].strip()
            if requested not in agent_ids:
                print(
                    f"\n  Unknown agent '{requested}'.\n"
                    f"  Type 'agents' to see what's available.\n"
                )
                continue
            current_agent = requested
            llm           = _build_llm(current_agent)
            history.clear()
            print(f"\n  Switched to: {current_agent}  (history cleared)\n")
            continue

        # ── Send message to the agent ─────────────────────────────────────────
        history.append(HumanMessage(content=user_input))

        print(f"\n  [{current_agent}] thinking …\n")
        try:
            response       = llm.invoke(history)
            assistant_text = response.content
        except Exception as exc:
            history.pop()   # remove the failed turn
            print(f"  [ERROR] {exc}\n")
            continue

        history.append(AIMessage(content=assistant_text))

        # Pretty-print the reply
        print(DIVIDER)
        print(_wrap(assistant_text))
        print(f"{DIVIDER}\n")


if __name__ == "__main__":
    main()
