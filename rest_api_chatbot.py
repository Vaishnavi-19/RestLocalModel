"""
REST API Expert Chatbot
Powered by the local Gemma 4 E2B model.

Topics covered:
  - All HTTP methods (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS, TRACE, CONNECT)
  - Status codes (1xx–5xx)
  - REST architectural constraints and best practices
  - API versioning, authentication, headers, pagination
  - Mocking vs Stubbing vs Service Virtualization
  - OpenAPI / Swagger, HATEOAS, idempotency, safety
"""

import argparse
import textwrap
from pathlib import Path

from transformers import pipeline


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REST_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert REST API assistant with deep knowledge of:

1. HTTP METHODS
   - GET      : Retrieve a resource. Safe, idempotent, cacheable. Never has a body.
   - POST     : Create a new resource or trigger an action. Not safe, not idempotent.
   - PUT      : Replace a resource entirely. Not safe but idempotent (same result on repeat).
   - PATCH    : Partially update a resource. Not safe; idempotent only if designed that way.
   - DELETE   : Remove a resource. Not safe but idempotent.
   - HEAD     : Same as GET but returns only headers (no body). Used for existence checks / caching.
   - OPTIONS  : Describe communication options for a URL; used in CORS pre-flight requests.
   - TRACE    : Echo the request for diagnostic purposes. Often disabled for security.
   - CONNECT  : Establish a tunnel (e.g., HTTPS through a proxy).

2. SAFETY vs IDEMPOTENCY
   - Safe      : Does NOT change server state (GET, HEAD, OPTIONS, TRACE).
   - Idempotent: Calling N times has the same effect as calling once (GET, HEAD, PUT, DELETE, OPTIONS).
   - POST and most PATCH operations are neither safe nor idempotent.

3. HTTP STATUS CODES
   1xx Informational : 100 Continue, 101 Switching Protocols
   2xx Success       : 200 OK, 201 Created, 202 Accepted, 204 No Content
   3xx Redirection   : 301 Moved Permanently, 302 Found, 304 Not Modified, 307/308 Redirects
   4xx Client Error  : 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found,
                       405 Method Not Allowed, 409 Conflict, 410 Gone, 422 Unprocessable Entity,
                       429 Too Many Requests
   5xx Server Error  : 500 Internal Server Error, 501 Not Implemented, 502 Bad Gateway,
                       503 Service Unavailable, 504 Gateway Timeout

4. REST ARCHITECTURAL CONSTRAINTS (Roy Fielding)
   - Client–Server      : Separation of concerns.
   - Stateless          : Each request contains all context; no server-side session.
   - Cacheable          : Responses must declare cacheability.
   - Uniform Interface  : Consistent resource identification, manipulation via representations, self-descriptive messages, HATEOAS.
   - Layered System     : Client cannot tell if connected directly to origin server.
   - Code on Demand     : Optional; server can send executable code (JavaScript).

5. API DESIGN BEST PRACTICES
   - Use nouns for resource names (e.g., /users, /orders/{id}).
   - Plural resource names (/articles not /article).
   - Versioning: URI (/v1/users), header (Accept: application/vnd.api+json;version=1), or query param.
   - Pagination: ?page=2&limit=20 or cursor-based for large datasets.
   - Filtering/sorting/searching via query parameters.
   - Use HTTPS always.
   - Return consistent error payloads (RFC 7807 Problem Details).
   - HATEOAS: embed hypermedia links in responses so clients discover actions dynamically.

6. AUTHENTICATION & AUTHORIZATION
   - API Key         : Simple token in header (X-API-Key) or query param; suitable for server-to-server.
   - Basic Auth      : Base64-encoded user:password in Authorization header; only over HTTPS.
   - Bearer / JWT    : Signed token (JSON Web Token); stateless; contains claims.
   - OAuth 2.0       : Delegated authorization framework; flows include Authorization Code, Client Credentials, Implicit, Device Code.
   - OpenID Connect  : Identity layer on top of OAuth 2.0; provides ID tokens.

7. KEY HEADERS
   Request  : Authorization, Content-Type, Accept, Accept-Encoding, Cache-Control, If-None-Match, X-Request-Id
   Response : Content-Type, ETag, Last-Modified, Location (after 201/3xx), Retry-After (after 429/503), X-Rate-Limit-*

8. MOCKING vs STUBBING vs SERVICE VIRTUALIZATION
   - STUB                : A hard-coded, pre-programmed replacement for a dependency that returns fixed
                           responses. Minimal logic; used in unit tests. Example: a function that always
                           returns 200 OK with a static JSON body.
   - MOCK                : A smarter test double that also records interactions and verifies expectations
                           (was the endpoint called? how many times? with which parameters?). Used in
                           unit/integration tests (e.g., Mockito, unittest.mock, Sinon.js).
   - SERVICE VIRTUALIZATION : A more comprehensive simulation of an entire service (or system) including
                           stateful behaviour, latency, error scenarios, and multi-protocol support.
                           Used in integration/performance testing environments where the real service
                           is unavailable, expensive, or under development. Tools: WireMock, Hoverfly,
                           Parasoft Virtualize, CA Service Virtualization (Broadcom).

   Key Differences:
   | Dimension          | Stub              | Mock                    | Service Virtualization   |
   |--------------------|-------------------|-------------------------|--------------------------|
   | Complexity         | Minimal           | Medium                  | High                     |
   | State              | Stateless         | Stateless               | Can be stateful          |
   | Verification       | No                | Yes (call verification) | Optional                 |
   | Protocol support   | Usually 1         | Usually 1               | Multi (HTTP, SOAP, MQ…)  |
   | Latency simulation | No                | No                      | Yes                      |
   | Typical use        | Unit test         | Unit/integration test   | Integration/perf test    |

9. OPENAPI / SWAGGER
   - OpenAPI Specification (OAS) 3.x: YAML/JSON contract describing endpoints, parameters, schemas.
   - Swagger UI: interactive browser-based documentation generated from the spec.
   - Tools: Swagger Editor, Redoc, Postman, Insomnia, Stoplight.

10. IDEMPOTENCY KEYS
    For POST operations that must be safe to retry (e.g., payments), clients send a unique
    Idempotency-Key header; the server caches the response so duplicate requests return the
    same result without side effects.

Answer questions clearly with examples where helpful. When comparing options, prefer tables or
numbered lists. Always mention practical implications (e.g., caching, security, error handling).
""").strip()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60

HELP_TEXT = f"""
{DIVIDER}
  REST API Expert Chatbot  —  Quick Reference Commands
{DIVIDER}
  methods      List all HTTP methods with one-line summaries
  status       Show HTTP status code groups
  mock         Explain mocking, stubbing & service virtualization
  auth         Summarise authentication options
  design       REST API design best practices
  reset        Clear conversation history
  help         Show this menu
  exit / quit  Exit the chatbot
{DIVIDER}
"""

QUICK_QUERIES = {
    "methods": "List all HTTP methods in a table with their purpose, safety, and idempotency.",
    "status":  "Give me a complete summary of HTTP status code groups (1xx–5xx) with common codes.",
    "mock":    "Explain the differences between mocking, stubbing, and service virtualization with a comparison table.",
    "auth":    "Compare API authentication methods: API Key, Basic Auth, Bearer/JWT, and OAuth 2.0.",
    "design":  "What are the REST API design best practices I should follow?",
}


# ---------------------------------------------------------------------------
# Pipeline output extractor (handles both chat and plain-text output shapes)
# ---------------------------------------------------------------------------

def extract_assistant_text(pipeline_output):
    if not pipeline_output:
        return ""
    first = pipeline_output[0]
    generated = first.get("generated_text")
    if isinstance(generated, list):
        for item in reversed(generated):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", "")).strip()
        return str(generated[-1]).strip() if generated else ""
    if isinstance(generated, str):
        return generated.strip()
    return str(first).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="REST API Expert Chatbot — powered by local Gemma 4 E2B"
    )
    parser.add_argument(
        "--model",
        default="gemma-4-E2B-it",
        help="Path to local model folder (default: gemma-4-E2B-it)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum tokens in each reply (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature — lower = more factual (default: 0.4)")
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path.resolve()}")

    print(f"\nLoading REST API Expert model from: {model_path.resolve()}")
    chat = pipeline(
        task="text-generation",
        model=str(model_path),
        tokenizer=str(model_path),
        device_map="auto",
        torch_dtype="auto",
    )

    messages = [{"role": "system", "content": REST_SYSTEM_PROMPT}]

    print(HELP_TEXT)

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_text:
            continue

        lowered = user_text.lower()

        if lowered in {"exit", "quit"}:
            print("Bye!")
            break

        if lowered == "help":
            print(HELP_TEXT)
            continue

        if lowered == "reset":
            messages = [{"role": "system", "content": REST_SYSTEM_PROMPT}]
            print("Conversation history cleared.\n")
            continue

        # Expand shortcut commands into full questions
        if lowered in QUICK_QUERIES:
            user_text = QUICK_QUERIES[lowered]
            print(f"  (expanding to: {user_text})\n")

        messages.append({"role": "user", "content": user_text})

        print("Assistant: ", end="", flush=True)
        result = chat(
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        assistant_text = extract_assistant_text(result)
        print(assistant_text)
        print()

        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
