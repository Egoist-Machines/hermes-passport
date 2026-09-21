"""``hermes ai_passport …`` commands.

Registered by Hermes only while this provider is the ACTIVE one
(``memory.provider: ai_passport`` in config.yaml): plugins/memory
``discover_plugin_cli_commands`` reads that key and loads the matching plugin's
``cli.py`` alone. Before activation, the entry point is ``hermes memory setup``.

That loader imports this file under a synthetic parent package whose
``__init__.py`` is never executed, so everything here imports real submodules
(``from .status import …``) and never a name defined in the provider module.
Reaching into ``__init__.py`` from here is what breaks a plugin CLI.

Nothing printed by these commands includes memory text or a token: an owner
pasting their status into an issue must not be pasting their Passport.
"""

from __future__ import annotations

import json

from .config import ProviderConfig, config_path
from .credentials import Credentials, PassportAuthError
from .env_sink import EnvTokenSink
from .formatting import describe_skipped
from .status import probe

PROVIDER_NAME = "ai_passport"
INSTALL_GUIDE_URL = "https://ego.ist/hermes"


def _hermes_home() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def register_cli(parser) -> None:
    """Add this plugin's subcommands. Called with the ``hermes ai_passport`` parser."""
    sub = parser.add_subparsers(dest="ai_passport_command", metavar="<command>")

    status = sub.add_parser("status", help="Show the connection, config and what is awaiting approval")
    status.add_argument("--json", action="store_true", help="Machine-readable output")

    check = sub.add_parser("probe", help="Read the agent-backend plane once and report what came back")
    check.add_argument("--query", default=None, help="Optional keyword to send as the read's query")
    check.add_argument("--json", action="store_true", help="Machine-readable output")

    sub.add_parser("refresh", help="Force an access-token refresh and update the MCP bearer line")
    sub.add_parser("setup", help="Re-run memory setup for AI Passport")


def _print_status(hermes_home: str, as_json: bool) -> int:
    config = ProviderConfig.load(hermes_home)
    credentials = Credentials(config.credentials_path)
    payload = {
        "provider": PROVIDER_NAME,
        "config_file": str(config_path(hermes_home)),
        "config": config.describe(),
        "credentials": credentials.summary(),
        "probe": probe(hermes_home),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["probe"]["ok"] else 1

    creds = payload["credentials"]
    print("\n  AI Passport memory provider\n")
    print(f"  config file        {payload['config_file']}")
    print(f"  credentials        {config.credentials_path}")
    if creds.get("ok"):
        expires_in = creds["access_token_expires_in_s"]
        token_state = f"{expires_in}s left" if creds["has_access_token"] else "none stored"
        print(f"  client_id          {creds['client_id']}")
        print(f"  token endpoint     {creds['token_url']}")
        print(f"  access token       {token_state}")
        if creds.get("pending_disk_write"):
            # Not a cosmetic lag: the refresh token ON DISK is already spent,
            # and replaying a spent token more than five minutes after its
            # rotation revokes the whole token family. The agent's manual
            # file-based refresh must not run while this line shows.
            print("  ⚠ disk             the credentials file is STALE (a rotated token could not be written);")
            print("                     do NOT refresh manually from the file until this clears, that replay")
            print("                     can revoke the install")
    else:
        print(f"  credentials        {creds.get('error')}")
    print(f"  ambient categories {', '.join(config.categories)}")
    print(f"  context            {'on' if config.context_enabled else 'off'}, "
          f"{config.context_limit} rows, {config.context_timeout_ms}ms budget")
    print(f"  tools              {'on' if config.tools_enabled else 'off'}")
    print(f"  mirror writes      {'on' if config.mirror_enabled else 'off'}"
          f"{' (including background review)' if config.mirror_background_review else ''}")
    print(f"  MCP bearer line    {config.env_sink_var if config.env_sink_enabled else '(off)'}")

    result = payload["probe"]
    print()
    if result["ok"]:
        rows = result["rows"]
        print(f"  ✓ plane answered: {rows} approved row{'' if rows == 1 else 's'}"
              f"{' (served from cache)' if result['stale'] else ''}")
        print(f"  awaiting approval: {describe_skipped(result['skipped'])}")
        if result["approval_url"]:
            print(f"  the owner approves passes at {result['approval_url']}")
    else:
        print(f"  ✗ {result['error']}")
        print(f"  install guide: {INSTALL_GUIDE_URL}")
    print()
    return 0 if result["ok"] else 1


def _print_probe(hermes_home: str, query, as_json: bool) -> int:
    result = probe(hermes_home, query=query)
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if not result["ok"]:
        print(f"\n  ✗ {result['error']}\n")
        return 1
    print(f"\n  ✓ {result['rows']} approved row(s) in {', '.join(result['categories'])}"
          f"{' (served from cache)' if result['stale'] else ''}")
    print(f"  awaiting approval: {describe_skipped(result['skipped'])}")
    print()
    return 0


def _refresh(hermes_home: str) -> int:
    config = ProviderConfig.load(hermes_home)
    if not config.credentials_path.exists():
        print(f"\n  ✗ no connect-code file at {config.credentials_path}")
        print(f"  install guide: {INSTALL_GUIDE_URL}\n")
        return 1
    sink = EnvTokenSink(hermes_home=hermes_home, var=config.env_sink_var, enabled=config.env_sink_enabled)
    wrote = {"value": False}

    def announce(token: str) -> None:
        wrote["value"] = sink.sync(token)

    credentials = Credentials(config.credentials_path, on_access_token=announce)
    try:
        credentials.access_token(force=True)
    except PassportAuthError as err:
        print(f"\n  ✗ {err.args[0]}\n")
        return 1
    print("\n  ✓ refreshed the access token and rotated the refresh token")
    if config.env_sink_enabled:
        if wrote["value"]:
            print(f"  ✓ updated {config.env_sink_var} in {hermes_home}/.env")
        else:
            # The rewrite-only rule: a missing line is the owner's opt-out, not a
            # failure, so say which of the two happened.
            print(f"  · left {hermes_home}/.env alone ({config.env_sink_var} line absent or already current)")
    print()
    return 0


def ai_passport_command(args) -> int:
    """Entry point Hermes wires to ``hermes ai_passport``."""
    command = getattr(args, "ai_passport_command", None) or "status"
    hermes_home = _hermes_home()

    if command == "setup":
        from hermes_cli.memory_setup import cmd_setup_provider

        cmd_setup_provider(PROVIDER_NAME)
        return 0
    if command == "probe":
        return _print_probe(hermes_home, getattr(args, "query", None), bool(getattr(args, "json", False)))
    if command == "refresh":
        return _refresh(hermes_home)
    return _print_status(hermes_home, bool(getattr(args, "json", False)))
