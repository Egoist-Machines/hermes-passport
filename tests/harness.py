"""Load the plugin the way Hermes loads it, with the host stubbed.

Two load paths matter and both are exercised here, because they behave
differently and the difference has bitten this plugin's design:

- ``load_provider()`` mirrors ``plugins/memory/__init__.py``
  ``_load_provider_from_dir``: a synthetic parent package with a real
  ``__path__``, submodules pre-registered, then ``__init__.py`` executed.
- ``load_cli()`` mirrors ``discover_plugin_cli_commands``: the parent package is
  a SHELL whose ``__init__.py`` is never executed, and only ``cli.py`` is
  imported. A CLI that reaches for a name defined in ``__init__.py`` breaks
  there and nowhere else, so tests/test_loader.py pins it.

The host modules the plugin imports are resolved from a real Hermes install when
one is present (so the tests run against the REAL MemoryProvider contract), and
stubbed otherwise so this suite works on a machine without Hermes.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PLUGIN_DIR.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
NAMESPACE = "_hermes_user_memory"
PLUGIN_NAME = "ai_passport"
MODULE_NAME = f"{NAMESPACE}.{PLUGIN_NAME}"
CLI_NAMESPACE = f"{NAMESPACE}_cli_only"
CLI_MODULE_NAME = f"{CLI_NAMESPACE}.{PLUGIN_NAME}.cli"


def fixture_text(name: str) -> str:
    """A vendored /agent/* contract body (tests/fixtures, synced
    byte-identical from the backend repo's shared/agent-contract/ by
    scripts/sync_shared.mjs). One loader for the behavioral and the contract
    suites, so the path and any future validation live in exactly one
    place."""
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> dict:
    return json.loads(fixture_text(name))


def hermes_agent_dir() -> Path | None:
    env = os.environ.get("HERMES_AGENT_DIR")
    if env:
        # The env var is an explicit ask to test against THIS host (the
        # monthly canary and release-evidence runs depend on it). Silently
        # falling back to ~/.hermes/hermes-agent here once meant a typo'd or
        # reorganized checkout still printed the "real Hermes" banner while
        # testing a different install, so an unusable value is an error, not
        # a shrug.
        candidate = Path(env)
        if (candidate / "agent" / "memory_provider.py").exists():
            return candidate
        raise RuntimeError(
            f"HERMES_AGENT_DIR is set to {env}, but {candidate / 'agent' / 'memory_provider.py'} does not exist. "
            "Either the path is wrong or hermes-agent moved its provider ABC; fix the path or update tests/harness.py."
        )
    candidate = Path.home() / ".hermes" / "hermes-agent"
    if (candidate / "agent" / "memory_provider.py").exists():
        return candidate
    return None


def _register_package(name: str, search_locations: list[str]) -> None:
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_host_stubs(hermes_home: str | None = None) -> None:
    """Make ``agent.*``, ``tools.*`` and ``hermes_constants`` importable."""
    agent_dir = hermes_agent_dir()

    if "agent" not in sys.modules:
        _register_package("agent", [str(agent_dir / "agent")] if agent_dir else [])
    if "agent.memory_provider" not in sys.modules:
        if agent_dir:
            # The real ABC, loaded without executing agent/__init__.py (which
            # pulls in the host's preload machinery).
            _load_file_as("agent.memory_provider", agent_dir / "agent" / "memory_provider.py")
        else:
            module = types.ModuleType("agent.memory_provider")

            class MemoryProvider:  # minimal stand-in for a Hermes-free machine
                pass

            def is_trivial_prompt(text):
                if not text:
                    return True
                stripped = text.strip()
                return not stripped or stripped.startswith("/") or stripped.lower() in {
                    "yes", "no", "ok", "okay", "thanks", "hi", "hey", "continue", "go ahead",
                }

            module.MemoryProvider = MemoryProvider
            module.is_trivial_prompt = is_trivial_prompt
            sys.modules["agent.memory_provider"] = module
        sys.modules["agent"].memory_provider = sys.modules["agent.memory_provider"]

    if "agent.secret_scope" not in sys.modules:
        module = types.ModuleType("agent.secret_scope")
        module.is_multiplex_active = lambda: False
        module.get_secret = lambda name, default=None: os.environ.get(name, default)
        sys.modules["agent.secret_scope"] = module

    if "tools" not in sys.modules:
        _register_package("tools", [])
    if "tools.registry" not in sys.modules:
        module = types.ModuleType("tools.registry")

        def tool_error(message, **extra):
            payload = {"error": str(message)}
            payload.update(extra)
            return json.dumps(payload, ensure_ascii=False)

        module.tool_error = tool_error
        sys.modules["tools.registry"] = module

    if "hermes_constants" not in sys.modules:
        module = types.ModuleType("hermes_constants")
        module.get_hermes_home = lambda: Path(hermes_home or os.environ.get("HERMES_HOME", "."))
        sys.modules["hermes_constants"] = module
    elif hermes_home:
        sys.modules["hermes_constants"].get_hermes_home = lambda: Path(hermes_home)


def set_hermes_home(hermes_home: str) -> None:
    """Point the stubbed ``get_hermes_home`` at a scratch dir."""
    install_host_stubs(hermes_home)
    sys.modules["hermes_constants"].get_hermes_home = lambda: Path(hermes_home)


def load_provider():
    """Import the plugin package the way the memory-provider loader does."""
    install_host_stubs()
    if MODULE_NAME in sys.modules and getattr(sys.modules[MODULE_NAME], "__file__", None):
        return sys.modules[MODULE_NAME]

    _register_package(NAMESPACE, [])
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        str(PLUGIN_DIR / "__init__.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    # The host pre-registers every top-level submodule before executing the
    # package, so mirror that: it is what makes a plugin's relative imports
    # resolve even though the package lives nowhere on sys.path.
    for sub_file in sorted(PLUGIN_DIR.glob("*.py")):
        if sub_file.name == "__init__.py":
            continue
        sub_name = f"{MODULE_NAME}.{sub_file.stem}"
        if sub_name not in sys.modules:
            _load_file_as(sub_name, sub_file)
    spec.loader.exec_module(module)
    return module


def submodule(name: str):
    """Fetch one of the plugin's submodules (loads the package first)."""
    load_provider()
    return sys.modules[f"{MODULE_NAME}.{name}"]


def load_cli():
    """Import only ``cli.py``, under a shell parent, exactly like Hermes does."""
    install_host_stubs()
    if CLI_MODULE_NAME in sys.modules:
        return sys.modules[CLI_MODULE_NAME]
    _register_package(CLI_NAMESPACE, [])
    # Note what is NOT done here: the plugin package's __init__.py is never
    # executed. The shell only carries a __path__ so real submodule imports work.
    _register_package(f"{CLI_NAMESPACE}.{PLUGIN_NAME}", [str(PLUGIN_DIR)])
    return _load_file_as(CLI_MODULE_NAME, PLUGIN_DIR / "cli.py")


class FakeResponse:
    """Minimal urlopen() context manager."""

    def __init__(self, status: int, body: str = "", headers: dict | None = None):
        self.status = status
        self._body = body.encode("utf-8")
        self.headers = headers or {"content-type": "application/json"}

    def read(self, limit: int | None = None) -> bytes:
        return self._body if limit is None else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHttp:
    """Scripted opener: a queue of responses (or exceptions) per URL suffix.

    Plain queues dispense in ARRIVAL order, which is only deterministic for
    requests the code under test makes sequentially. The ambient path submits
    its two reads to a thread pool concurrently BY DESIGN, so a test whose two
    fixtures differ must key them on the request SHAPE instead: pass
    ``when=<predicate over the parsed JSON body>`` and the entry answers only
    a matching request (issue #644: the query/recent fixtures swapped under an
    unlucky thread schedule and failed a merge-order assertion against a
    provider that had merged correctly).
    """

    def __init__(self):
        self.responses: dict[str, list] = {}
        self.calls: list[dict] = []

    def queue(self, suffix: str, *responses, when=None) -> None:
        self.responses.setdefault(suffix, []).extend({"when": when, "response": r} for r in responses)

    def __call__(self, request, timeout=None):
        url = request.full_url
        body = request.data.decode("utf-8") if request.data else ""
        self.calls.append(
            {
                "url": url,
                "body": body,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "timeout": timeout,
            }
        )
        try:
            parsed = json.loads(body) if body else {}
        except ValueError:
            parsed = {}
        # Predicates always see a dict so `"key" in body` is safe even for a
        # bodyless GET; a predicate that raises is a broken test and should
        # fail loudly rather than be skipped.
        parsed = parsed if isinstance(parsed, dict) else {}
        for suffix, entries in self.responses.items():
            if not url.endswith(suffix):
                continue
            for index, entry in enumerate(entries):
                if entry["when"] is not None and not entry["when"](parsed):
                    continue
                answer = entries.pop(index)["response"]
                if isinstance(answer, Exception):
                    raise answer
                if callable(answer):
                    return answer(request)
                return answer
        raise AssertionError(f"unscripted request to {url}")

    def bodies(self, suffix: str) -> list:
        return [json.loads(call["body"]) for call in self.calls if call["url"].endswith(suffix) and call["body"]]

    def count(self, suffix: str) -> int:
        return len([call for call in self.calls if call["url"].endswith(suffix)])


def write_credentials(path: Path, **overrides) -> dict:
    payload = {
        "token_url": "https://passport.test/oauth/token",
        "client_id": "client-abc",
        "refresh_token": "refresh-1",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)
    return payload


def prefetch_body(rows=(), skipped=(), approval_url="https://my.ego.ist/passes") -> str:
    return json.dumps(
        {
            "rows": list(rows),
            "skipped_categories": list(skipped),
            "approval_url": approval_url,
        }
    )


def row(memory_id: str, content: str, category: str = "preference", created_at: str = "2026-08-14T00:00:00Z") -> dict:
    return {"memory_id": memory_id, "content": content, "category": category, "created_at": created_at}
