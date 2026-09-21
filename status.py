"""Live probes of the agent-backend plane, shared by setup, status and the CLI.

This lives in its own submodule rather than in ``__init__.py`` on purpose.
Hermes loads a plugin's ``cli.py`` under a synthetic parent package whose
``__init__.py`` is NEVER executed (plugins/memory/__init__.py
``discover_plugin_cli_commands``), so a CLI command can import real submodules
but cannot reach a name defined in the provider module. Anything both the
provider and the CLI need has to sit here.

Everything reported is content-free: counts, closed-vocabulary category names
and skip reasons. Never row text, never the token.
"""

from __future__ import annotations

from typing import Optional

from .client import PassportClient
from .config import ProviderConfig
from .credentials import Credentials
from .env_sink import EnvTokenSink
from .formatting import describe_skipped

# A human is watching a probe, so it may take longer than a turn is allowed to.
PROBE_TIMEOUT_S = 10.0


def build_client(hermes_home: str):
    """Assemble the same runtime the provider uses, outside a session."""
    config = ProviderConfig.load(hermes_home)
    sink = EnvTokenSink(hermes_home=hermes_home, var=config.env_sink_var, enabled=config.env_sink_enabled)
    credentials = Credentials(config.credentials_path, on_access_token=sink)
    client = PassportClient(config=config, credentials=credentials)
    return config, credentials, client


def probe(hermes_home: str, *, query: Optional[str] = None) -> dict:
    """One live read of ``/agent/prefetch``."""
    config, _credentials, client = build_client(hermes_home)
    status: dict = {
        "ok": False,
        "error": "",
        "credentials": str(config.credentials_path),
        "categories": list(config.categories),
        "rows": 0,
        "skipped": [],
        "approval_url": None,
        "stale": False,
    }
    if not config.credentials_path.exists():
        status["error"] = f"no connect-code file at {config.credentials_path}"
        return status
    # The client never raises: a failure comes back as None plus a reason in its
    # state, which is what this reports.
    result = client.prefetch(
        categories=config.categories,
        query=query,
        limit=config.context_limit,
        session_key=None,
        timeout_s=PROBE_TIMEOUT_S,
    )
    if result is None:
        state = client.state()
        status["error"] = state.get("terminal_reason") or state.get("backoff_reason") or "no answer"
        return status
    status.update(
        {
            "ok": True,
            "rows": len(result.rows),
            "skipped": result.skipped,
            "approval_url": result.approval_url,
            "stale": result.stale,
        }
    )
    return status


def probe_summary(hermes_home: str) -> str:
    status = probe(hermes_home)
    if not status["ok"]:
        return f"✗ {status['error']}"
    rows = status["rows"]
    noun = "row" if rows == 1 else "rows"
    return (
        f"✓ Connected · {rows} approved {noun} in {', '.join(status['categories'])} · "
        f"awaiting approval: {describe_skipped(status['skipped'])}"
    )
