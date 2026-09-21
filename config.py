"""Config resolution for the AI Passport memory provider.

Every field is optional and every bad value falls back to its default rather
than raising. A memory provider that refuses to load takes the agent's whole
memory surface down with it, and this one is an additive read path whose worst
honest failure mode is "no Passport rows".

Owner-facing config lives at ``<hermes_home>/ai-passport.json``, next to the
credentials the install guide writes. Nothing here reads ``~/.hermes``
directly: Hermes hands the provider its ``hermes_home`` so profile-scoped
installs and HERMES_HOME overrides keep working.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "ai-passport.json"
CREDENTIALS_FILE_NAME = "ai-passport-refresh.json"

# The governed vocabulary, kept in lockstep with the backend's memory
# categories (shared/contracts.js MEMORY_CATEGORIES). tests/test_config.py
# fails when the two drift.
GOVERNED_CATEGORIES = (
    "preference",
    "fact",
    "project",
    "relationship",
    "instruction",
    "event",
    "purchase",
    "claim",
    "other",
)

# `claim` is Passport-computed and server-asserted (ADR-0027), so no calling
# model may mint one. Mirrors shared/contracts.js WRITABLE_MEMORY_CATEGORIES.
DERIVED_CATEGORIES = ("claim",)
WRITABLE_CATEGORIES = tuple(c for c in GOVERNED_CATEGORIES if c not in DERIVED_CATEGORIES)

# Rows only ever come back for categories the owner approved a pass for, so the
# default set is about what is worth ASKING for on an ambient read: the four
# that describe the person and their work. Owners widen it in config.
DEFAULT_CATEGORIES = ("preference", "fact", "project", "instruction")

DEFAULTS: dict[str, Any] = {
    "base_url": "",
    "categories": list(DEFAULT_CATEGORIES),
    "cache_ttl_ms": 60_000,
    "context": {
        "enabled": True,
        "limit": 6,
        "max_chars": 2000,
        # prefetch() runs before EVERY API call and Hermes documents no timeout
        # contract for it, so the provider brings its own and it has to be the
        # budget an owner would not notice. A slow or unreachable Passport
        # serves the last good answer instead of stalling the turn.
        "timeout_ms": 800,
        "send_prompt_as_query": True,
        "include_recent": True,
    },
    "tools": {
        "enabled": True,
        # A model-initiated recall is allowed to be slow: it can mint an
        # approval request and the user is waiting on an answer, not on a
        # prompt build.
        "timeout_ms": 10_000,
    },
    "mirror": {
        # Mirror the built-in memory tool's durable writes into the owner's
        # Passport review inbox.
        "enabled": True,
        # background_review writes are Hermes' own automated pass over the
        # transcript. They are legitimate memory, but they arrive in volume and
        # every one of them costs the owner a review decision, so mirroring
        # them is opt-in.
        "include_background_review": False,
        "timeout_ms": 10_000,
    },
    "env_sink": {
        # Keep the MCP server's static bearer in step with the token this
        # provider refreshes. See env_sink.py for why this is rewrite-only.
        "enabled": True,
        "var": "AI_PASSPORT_TOKEN",
    },
    "policy": {
        # Report each tool call's name (never its arguments) to the owner's
        # tool-policy audit trail, via the shell pre_tool_call hook that
        # post_setup wires (policy_hook.py). In audit mode the hook never
        # blocks; in enforce mode it vetoes denied tools and blocks pending
        # approvals with the owner's approval link.
        "enabled": True,
        # The hook's own HTTP budget per check POST. The process spawn around
        # it is Hermes' cost, bounded by the YAML entry's `timeout`.
        "timeout_ms": 2000,
        # Whether Hermes should BLOCK a tool call when the hook process itself
        # fails to run (crash, kill, spawn error). Off by default: an audit
        # reporter must never veto by dying. An owner running enforce mode
        # sets this true and re-runs `hermes memory setup` so a crashed hook
        # cannot silently allow what their rules would have blocked.
        "fail_closed": False,
    },
}


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    # `hermes memory setup` prompts free text and hands back strings whatever
    # the schema says its type is, so coercion happens here rather than there.
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return fallback


def _bounded(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except (TypeError, ValueError):
            return fallback
    if not isinstance(value, (int, float)):
        return fallback
    try:
        rounded = int(round(value))
    except (OverflowError, ValueError):
        return fallback
    return min(maximum, max(minimum, rounded))


def _trimmed(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_base_url(value: Any) -> Optional[str]:
    """A base URL is usable only if it parses AND is http(s).

    Anything else would send a bearer token somewhere unintended.
    """
    raw = _trimmed(value)
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{path}"


def config_path(hermes_home: str) -> Path:
    return Path(hermes_home) / CONFIG_FILE_NAME


def read_config_file(hermes_home: str) -> dict:
    path = config_path(hermes_home)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("ai-passport: %s is not readable JSON", path, exc_info=True)
        return {}
    return raw if isinstance(raw, dict) else {}


def write_config_file(hermes_home: str, values: dict) -> None:
    """Merge ``values`` into the provider's own config file.

    Merge rather than replace: ``hermes memory setup`` only ever collects the
    fields it prompted for, and an owner's hand-edited context budget must not
    be erased by a re-run of setup.
    """
    path = config_path(hermes_home)
    existing = read_config_file(hermes_home)
    existing.update({k: v for k, v in (values or {}).items() if v is not None})
    try:
        from utils import atomic_json_write  # type: ignore

        atomic_json_write(path, existing, mode=0o600, sort_keys=True)
        return
    except Exception:
        # Standing on our own feet when the host helper is unavailable (a bare
        # unit-test import, an older Hermes). Same shape: write beside the
        # target, then rename, so a crash cannot leave a half-written config.
        pass
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    os.replace(temp, path)


class ProviderConfig:
    """The normalized, immutable-enough view the runtime reads."""

    def __init__(self, raw: Any, hermes_home: str, env: Optional[dict] = None):
        env = os.environ if env is None else env
        source = raw if isinstance(raw, dict) else {}
        context = source.get("context") if isinstance(source.get("context"), dict) else {}
        tools = source.get("tools") if isinstance(source.get("tools"), dict) else {}
        mirror = source.get("mirror") if isinstance(source.get("mirror"), dict) else {}
        env_sink = source.get("env_sink") if isinstance(source.get("env_sink"), dict) else {}
        policy = source.get("policy") if isinstance(source.get("policy"), dict) else {}

        self.hermes_home = hermes_home
        self.credentials_path = Path(
            _trimmed(source.get("credentials_path")) or str(Path(hermes_home) / CREDENTIALS_FILE_NAME)
        )
        # None means "derive it from the credentials file's token_url", which is
        # what makes one guide work against passport.ego.ist and a local stack
        # without the owner editing two places.
        self.base_url = normalize_base_url(source.get("base_url")) or normalize_base_url(
            env.get("AI_PASSPORT_BASE_URL")
        )

        declared = source.get("categories")
        declared = declared if isinstance(declared, list) else []
        kept = [c for c in declared if c in GOVERNED_CATEGORIES]
        dropped = [c for c in declared if c not in GOVERNED_CATEGORIES]
        if dropped:
            # Falling back is the right survival posture, but doing it silently
            # would have the provider ambiently reading categories the owner
            # never asked for while never fetching the ones they misspelled.
            logger.warning(
                "ai-passport: ignoring unknown categories in config: %s%s",
                ", ".join(str(c) for c in dropped),
                "" if kept else "; using the default set",
            )
        seen: list[str] = []
        for category in kept or DEFAULT_CATEGORIES:
            if category not in seen:
                seen.append(category)
        self.categories = tuple(seen)

        self.cache_ttl_ms = _bounded(source.get("cache_ttl_ms"), 1_000, 600_000, DEFAULTS["cache_ttl_ms"])

        self.context_enabled = _as_bool(context.get("enabled"), DEFAULTS["context"]["enabled"])
        self.context_limit = _bounded(context.get("limit"), 1, 50, DEFAULTS["context"]["limit"])
        self.context_max_chars = _bounded(context.get("max_chars"), 200, 20_000, DEFAULTS["context"]["max_chars"])
        self.context_timeout_ms = _bounded(context.get("timeout_ms"), 200, 10_000, DEFAULTS["context"]["timeout_ms"])
        self.send_prompt_as_query = _as_bool(
            context.get("send_prompt_as_query"), DEFAULTS["context"]["send_prompt_as_query"]
        )
        self.include_recent = _as_bool(context.get("include_recent"), DEFAULTS["context"]["include_recent"])

        self.tools_enabled = _as_bool(tools.get("enabled"), DEFAULTS["tools"]["enabled"])
        self.tools_timeout_ms = _bounded(tools.get("timeout_ms"), 1_000, 60_000, DEFAULTS["tools"]["timeout_ms"])

        self.mirror_enabled = _as_bool(mirror.get("enabled"), DEFAULTS["mirror"]["enabled"])
        self.mirror_background_review = _as_bool(
            mirror.get("include_background_review"), DEFAULTS["mirror"]["include_background_review"]
        )
        self.mirror_timeout_ms = _bounded(mirror.get("timeout_ms"), 1_000, 60_000, DEFAULTS["mirror"]["timeout_ms"])

        self.env_sink_enabled = _as_bool(env_sink.get("enabled"), DEFAULTS["env_sink"]["enabled"])
        self.env_sink_var = _trimmed(env_sink.get("var")) or DEFAULTS["env_sink"]["var"]

        self.policy_enabled = _as_bool(policy.get("enabled"), DEFAULTS["policy"]["enabled"])
        self.policy_timeout_ms = _bounded(policy.get("timeout_ms"), 200, 10_000, DEFAULTS["policy"]["timeout_ms"])
        self.policy_fail_closed = _as_bool(policy.get("fail_closed"), DEFAULTS["policy"]["fail_closed"])

    @classmethod
    def load(cls, hermes_home: str, env: Optional[dict] = None) -> "ProviderConfig":
        return cls(read_config_file(hermes_home), hermes_home, env=env)

    def describe(self) -> dict:
        """Content-free summary for `hermes ai_passport status`."""
        return {
            "credentials_path": str(self.credentials_path),
            "base_url": self.base_url or "(from the credentials file)",
            "categories": list(self.categories),
            "context_enabled": self.context_enabled,
            "context_timeout_ms": self.context_timeout_ms,
            "tools_enabled": self.tools_enabled,
            "mirror_enabled": self.mirror_enabled,
            "mirror_background_review": self.mirror_background_review,
            "env_sink": self.env_sink_var if self.env_sink_enabled else "(off)",
            "policy_enabled": self.policy_enabled,
            "policy_fail_closed": self.policy_fail_closed,
        }
