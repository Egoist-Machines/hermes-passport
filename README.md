# AI Passport memory for Hermes

Makes the owner's [AI Passport](https://ego.ist) the durable memory behind a
Hermes agent. Owner-approved context is injected each turn, and two tools let the
model read more or propose a new fact for review. Nothing is readable that the
owner has not approved a pass for, and nothing is written without their review.

- Ambient recall reads the agent-backend plane (`POST /agent/prefetch`), which is
  side-effect-free: no approval requests, no pass claims, no receipts. A per-turn
  read never nags the owner.
- `passport_recall` and `passport_remember` map onto the Passport MCP tools, where
  recall can request the owner's approval (including live data from their
  connected apps) and remember writes a proposal into their private inbox.
- Native `MEMORY.md` / `USER.md` stay ON. This provider is additive, and their
  durable writes are mirrored into the owner's Passport inbox for review.

The provider advertises `domovoy_contract_version = 1` for the reviewed
authenticated Domovoy profile. In that mode Hermes replaces the broad upstream
tool schemas with exactly one memory category or one connector/data-category
pair, and requires one category for a proposed memory. Passport recall returns
a server-authored structured control sibling: approval URLs and request IDs are
preserved as a typed event rather than scraped from prose. Hermes generates a
stable `save_id` only after phone confirmation; the provider forwards it
unchanged to `propose_memory` and echoes it in the typed submission result.
Domovoy configuration disables ambient context, automatic mirroring, the token
environment sink, and the plugin policy hook.

The reviewed `passport_recall` schema does not expose `time_min`, `time_max`, or
`time_zone`. Calls containing those fields are rejected before any request is
sent, so a requested date limit cannot silently become an unbounded read. Use a
separate Passport MCP `recall` tool for time filters only when the host offers
that tool and its current schema exposes the fields. Otherwise, tell the owner
that a time-limited read is unavailable through this installed wrapper. Connector
reads remain unavailable in unattended runs. Adding these fields to the reviewed
phone profile requires a coordinated host contract update.

## Install

```bash
hermes plugins install Egoist-Machines/hermes-passport
hermes memory setup            # pick "ai_passport"
```

Then redeem a connect code from the owner's Passport, which writes
`<hermes_home>/ai-passport-refresh.json` (mode 600). Full walkthrough:
<https://ego.ist/hermes>.

Nothing works until that file exists: `hermes memory status` and
`hermes ai_passport status` both say so plainly rather than failing quietly.

## Commands

Registered once `memory.provider: ai_passport` is active in `config.yaml`.

| Command | What it does |
| --- | --- |
| `hermes ai_passport status` | Connection, config, and which categories are awaiting the owner's approval |
| `hermes ai_passport probe [--query TEXT]` | One live read of the plane, reporting counts only |
| `hermes ai_passport refresh` | Force a token refresh and update the MCP bearer line |
| `hermes ai_passport setup` | Re-run memory setup for this provider |

Neither command prints memory text or a token, so a status paste is safe to
share in an issue.

## Config

`<hermes_home>/ai-passport.json`, all fields optional. Bad values fall back to
the default rather than failing: a memory provider that refuses to load takes the
agent's whole memory surface with it.

```json
{
  "categories": ["preference", "fact", "project", "instruction"],
  "cache_ttl_ms": 60000,
  "context": {
    "enabled": true,
    "limit": 6,
    "max_chars": 2000,
    "timeout_ms": 800,
    "send_prompt_as_query": true,
    "include_recent": true
  },
  "tools": { "enabled": true, "timeout_ms": 10000 },
  "mirror": { "enabled": true, "include_background_review": false },
  "env_sink": { "enabled": true, "var": "AI_PASSPORT_TOKEN" },
  "policy": { "enabled": true, "timeout_ms": 2000, "fail_closed": false }
}
```

- `categories` are the governed Passport categories to ask for ambiently. Rows
  only ever come back for categories the owner approved, so widening this asks
  for more rather than taking more.
- `context.send_prompt_as_query: false` keeps the user's turn on the machine: the
  provider then reads recent rows only. The prompt is otherwise sent as a
  256-character search query, and the backend never logs it.
- `context.timeout_ms` is the whole ambient budget, shared by both reads. A slow
  or unreachable Passport serves the last good answer instead of stalling the
  turn.
- `mirror.include_background_review: true` also mirrors Hermes' automated review
  writes. Off by default because each mirrored write costs the owner a review
  decision.
- `env_sink` keeps `$AI_PASSPORT_TOKEN` in `<hermes_home>/.env` in step with the
  token this provider refreshes, so the MCP server entry's static bearer does not
  expire. It is REWRITE-ONLY: if that line is absent the provider leaves the file
  alone, because a missing line is the owner's opt-out.

- `policy` is the tool-call policy surface (issue #425 phases 4+5).
  `hermes memory setup` wires `policy_hook.py` into `config.yaml` as a
  `hooks.pre_tool_call` shell hook; Hermes itself asks for consent the first
  time the hook runs (unattended installs need `hooks_auto_accept: true`). The
  hook posts each tool call's NAME plus a sha256 digest of its input (never
  the input); in audit mode it always exits 0 silently, and in enforce mode it
  vetoes denied tools and blocks approval-gated calls with the owner's inbox
  link (the agent retries after the owner approves; a shell hook cannot wait
  minutes, so veto-plus-link is the ceiling here). During an outage the last
  snapshot's rules are evaluated locally: fail closed exactly for the tools
  the owner constrained, fail open otherwise. It NEVER refreshes the token:
  rotation belongs to the provider, and a second refresher in a short-lived
  process is the two-refresher race all over again. An owner running enforce
  mode also sets `policy.fail_closed: true` and re-runs setup so a crashed
  hook process blocks rather than silently allows. Set
  `policy.enabled: false` and re-run `hermes memory setup` to unwire it.
  Verify with `python3 <plugin dir>/policy_hook.py --home ~/.hermes --selftest`.

`AI_PASSPORT_BASE_URL` overrides the Passport origin for local development. By
default the origin is derived from the credentials file's `token_url`, so one
guide serves production and a local stack.

## How the two halves stay out of each other's way

The install has two refreshers sharing one single-use rotating refresh token:
this provider, and the agent's own documented repair of the MCP bearer. The file
on disk is the source of truth, re-read whenever its mtime moves, so the agent's
rotation is picked up rather than overwritten. The race itself is settled
server-side: for five minutes after a rotation the backend keeps a sealed copy
of the response it issued and returns it again to a replay from the same client
address, so whichever half rotates second lands on the same successor chain.

The provider therefore does exactly one token POST per rotation and never
retries an `invalid_grant`. Past that recovery window a replay is classified as
theft and revokes the whole token family, so a retry would spend tokens from a
chain that is already dead. In the one narrow case a retry could have healed
(an in-window replay from a different egress address than the winner's, which
answers `invalid_grant` without killing the family), the terminal latch
re-probes within minutes and the file re-read picks up the winner's rotation,
so the install recovers on its own.

The self-heal has one precondition: the rotated token must actually reach the
file. When the credentials write fails (disk full, read-only home), the file
keeps the SPENT token and the live successor exists only in this process's
memory; an agent that then runs the guide's manual file-based refresh replays
the spent token, and past the recovery window that revokes the family,
successor included. The provider warns loudly in that state, retries the write
on every read, and `hermes ai_passport status` shows it as a stale-file
warning; the guide's refresh step must not run while it shows.

A refresh POST is not idempotent, so nothing cancels one in flight. The ambient
path stops waiting when its deadline passes and serves stale context; the
rotation finishes on a non-daemon worker and persists.

## Tests

```bash
python3 clients/hermes-passport/tests/run.py -v
```

Standard library only. With a Hermes install present (`HERMES_AGENT_DIR`, or
`~/.hermes/hermes-agent`) the suite runs against the real `MemoryProvider`
contract and the real loader semantics; without one it stubs the host and still
passes. `tests/test_config.py` fails if the governed category vocabulary drifts
from the backend's `shared/contracts.js`.

## Layout

| File | Role |
| --- | --- |
| `__init__.py` | The `MemoryProvider` implementation and `register(ctx)` |
| `client.py` | `/agent/prefetch` client: cache, backoff, request budget, degradation |
| `credentials.py` | Connect-code file, OAuth refresh, single-use token rotation |
| `mcp.py` | Direct JSON-RPC calls to `/mcp` for the two tools |
| `formatting.py` | The per-turn context block and its framing |
| `config.py` | Config resolution and the governed vocabulary |
| `cache.py` | TTL cache that keeps expired entries as fallback material |
| `env_sink.py` | Rewrite-only `.env` sink for the MCP bearer |
| `status.py` | Live probes shared by setup, status and the CLI |
| `cli.py` | `hermes ai_passport …` |
| `policy_hook.py` | Standalone `pre_tool_call` shell hook: the tool-call audit |

`cli.py` and `status.py` are split out on purpose: Hermes loads a plugin's
`cli.py` under a package shell whose `__init__.py` never executes, so a CLI
command can import real submodules and nothing else.
