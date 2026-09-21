#!/usr/bin/env python3
"""AI Passport tool-policy hook for Hermes (issue #425 phase 4, audit-only).

Hermes' approval gate is closed to external deciders, so the honest seam is a
shell ``pre_tool_call`` hook: Hermes pipes ``{tool_name, tool_input,
session_id, ...}`` on stdin, and a hook may answer ``{"action": "block", ...}``
on stdout. This script posts the tool's NAME and a DIGEST of its input to
``POST /agent/policy/check`` so the owner's activity feed sees the calls MCP
alone never could, then stays silent. ``post_setup`` wires it into
``config.yaml`` with ``fail_closed: false``; Hermes itself asks the user for
consent the first time the hook runs.

Standalone by design. Hermes spawns a fresh process per tool call, so this
file imports only the standard library and shares nothing in-process with the
memory provider. What the two DO share is on disk: the credentials file the
provider maintains, the provider's own config file, and a small state file
this script keeps for its backoff, dedupe and budget.

Rules the fresh-process shape forces:

- NEVER refresh the token. Rotation of a single-use refresh token belongs to
  the provider's single-flighted refresher; a second refresher in a short-lived
  process would race it from another process, which is exactly the two-refresher
  hazard the #619 alignment removed. An expired access token skips the report;
  the provider's next prefetch refreshes it.
- Fail open, silently, always. In audit mode there is nothing to protect and a
  tool call must never be slowed or failed by its own audit. Every error path
  exits 0 with no stdout. (``fail_closed: false`` in the YAML says the same
  thing from Hermes' side.)
- Persist the backoff. Without the state file every tool call would retry an
  unreachable Passport at full price, because no memory survives the process.

The content boundary is the same as everywhere else on this plane: the
arguments themselves never leave the machine, only ``h1_`` + sha256 over a
key-sorted JSON form (the backend shape-checks exactly that, so argument text
cannot be smuggled into the owner's feed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CREDENTIALS_FILE_NAME = "ai-passport-refresh.json"
CONFIG_FILE_NAME = "ai-passport.json"
STATE_FILE_NAME = "ai-passport-policy-state.json"

DEFAULT_TIMEOUT_MS = 2000
DEFAULT_CACHE_TTL_S = 60
MAX_CACHE_TTL_S = 600
MAX_CACHED_ANSWERS = 256
# Client-side share of the backend's 600/min check budget, spent process-wide
# through the state file. Deliberately below the OpenClaw twin's 300: a state
# file updated by racing processes undercounts, so the headroom is bigger.
BUDGET_WINDOW_S = 60
BUDGET_MAX = 120

BACKOFF_S = {
    "rate_limited": 60,
    "forbidden": 300,
    "invalid_request": 300,
    "unavailable": 30,
    "auth": 60,
}

# Mirrors the backend's TOOL_SHAPE (lib/agentPolicy.js), checked here so one
# host-owned name outside the grammar costs its own report, not a five-minute
# version-skew backoff against every other tool.
TOOL_SHAPE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
SESSION_KEY_MAX = 128

# The matcher twin for LOCAL evaluation during an outage (enforce mode only,
# issue #425 phase 5). Same tiny grammar and total order as the backend's
# lib/agentPolicy.js: exact beats prefix, longer prefix beats shorter, bare *
# is the floor, default allow. SNAPSHOT_VERSION pins the contract; a snapshot
# from a newer protocol is refused whole rather than half-understood.
SNAPSHOT_VERSION = 1
SNAPSHOT_TTL_S = 60
PATTERN_SHAPE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.:-]*\*?|\*)$")
POLICY_ACTIONS = ("allow", "deny", "require_approval")


def resolve_local_policy(tool, rules) -> dict:
    best = {"action": "allow", "matched_pattern": None}
    if not isinstance(tool, str) or not TOOL_SHAPE.match(tool) or not isinstance(rules, list):
        return best
    best_score = -1
    for rule in rules:
        pattern = rule.get("tool_pattern") if isinstance(rule, dict) else None
        action = rule.get("action") if isinstance(rule, dict) else None
        if not isinstance(pattern, str) or not PATTERN_SHAPE.match(pattern) or action not in POLICY_ACTIONS:
            continue
        if pattern == "*":
            matches, score = True, 0
        elif pattern.endswith("*"):
            matches, score = tool.startswith(pattern[:-1]), len(pattern)
        else:
            matches, score = tool == pattern, 1000 + len(pattern)
        if matches and score > best_score:
            best_score = score
            best = {"action": action, "matched_pattern": pattern}
    return best


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _write_state(path: str, state: dict) -> None:
    temp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, separators=(",", ":"))
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    except Exception:
        try:
            os.unlink(temp)
        except Exception:
            pass


def args_digest(tool_input) -> str | None:
    """``h1_`` + sha256 over a key-sorted JSON form, or None when the input
    cannot be canonicalized. None is a valid check-route value meaning "no
    digest", so failure degrades to a coarser audit row, never a dropped one.

    ``allow_nan=False`` keeps this in step with the OpenClaw twin
    (src/policy.js canonicalJson): Python would otherwise emit bare ``NaN``,
    which is not JSON at all.
    """
    if not isinstance(tool_input, dict) or not tool_input:
        return None
    try:
        canonical = json.dumps(tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except Exception:
        return None
    return "h1_" + hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()


class PolicyState:
    """The cross-process memory: backoff, dedupe cache, request budget.

    Last-writer-wins on a race between concurrent hook processes; the worst
    case is an undercounted budget or a lost cache entry, both of which cost
    one extra HTTP request, never a wrong answer.
    """

    def __init__(self, hermes_home: str, now=time.time):
        self._path = os.path.join(hermes_home, STATE_FILE_NAME)
        self._now = now
        raw = _read_json(self._path)
        self.backoff_until = raw.get("backoff_until") if isinstance(raw.get("backoff_until"), (int, float)) else 0
        self.backoff_reason = raw.get("backoff_reason") if isinstance(raw.get("backoff_reason"), str) else None
        cache = raw.get("cache") if isinstance(raw.get("cache"), dict) else {}
        # Entries are {"expires": epoch, "answer": {...}}; anything else
        # (including the expiry-only shape an older hook wrote) is dropped
        # rather than trusted as an answer it never carried.
        self.cache = {
            k: v
            for k, v in cache.items()
            if isinstance(v, dict) and isinstance(v.get("expires"), (int, float)) and v["expires"] > now()
        }
        budget = raw.get("budget") if isinstance(raw.get("budget"), list) else []
        cutoff = now() - BUDGET_WINDOW_S
        self.budget = [t for t in budget if isinstance(t, (int, float)) and t > cutoff]
        # The last snapshot the plane served ({mode, rules, fetched_at}): the
        # enforce-mode outage posture. Kept past its refresh interval on
        # purpose; stale rules beat no rules when the plane stops answering.
        snapshot = raw.get("snapshot")
        self.snapshot = snapshot if isinstance(snapshot, dict) else None

    def backed_off(self) -> bool:
        return self.backoff_until > self._now()

    def back_off(self, reason: str) -> None:
        self.backoff_until = self._now() + BACKOFF_S.get(reason, BACKOFF_S["unavailable"])
        self.backoff_reason = reason

    def clear_backoff(self) -> None:
        self.backoff_until = 0
        self.backoff_reason = None

    def cached(self, key: str):
        """The stored ANSWER for a fresh entry, else None.

        The answer itself is stored, not just an expiry: a phase-5 enforce
        deny with a cache_ttl must keep blocking identical repeats for its
        whole TTL, and an expiry-only cache would silently allow them (the
        exact forward-compat hole hook_response exists to prevent).
        """
        entry = self.cache.get(key)
        if not isinstance(entry, dict):
            return None
        expires = entry.get("expires")
        if not isinstance(expires, (int, float)) or expires <= self._now():
            return None
        answer = entry.get("answer")
        return answer if isinstance(answer, dict) else None

    def remember(self, key: str, ttl_s: float, answer: dict) -> None:
        ttl = min(MAX_CACHE_TTL_S, ttl_s) if isinstance(ttl_s, (int, float)) and ttl_s > 0 else DEFAULT_CACHE_TTL_S
        self.cache[key] = {"expires": self._now() + ttl, "answer": answer}
        while len(self.cache) > MAX_CACHED_ANSWERS:
            self.cache.pop(min(self.cache, key=lambda k: self.cache[k].get("expires", 0)))

    def budget_exhausted(self) -> bool:
        return len(self.budget) >= BUDGET_MAX

    def spend(self) -> None:
        self.budget.append(self._now())

    def save(self) -> None:
        _write_state(
            self._path,
            {
                "backoff_until": self.backoff_until,
                "backoff_reason": self.backoff_reason,
                "cache": self.cache,
                "budget": self.budget,
                "snapshot": self.snapshot,
            },
        )


def load_settings(hermes_home: str, env=None) -> dict:
    """The subset of the provider's config this script needs, same fallbacks."""
    env = os.environ if env is None else env
    raw = _read_json(os.path.join(hermes_home, CONFIG_FILE_NAME))
    policy = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
    enabled = policy.get("enabled")
    timeout_ms = policy.get("timeout_ms")
    if not isinstance(timeout_ms, (int, float)) or isinstance(timeout_ms, bool):
        timeout_ms = DEFAULT_TIMEOUT_MS
    timeout_ms = min(10_000, max(200, int(timeout_ms)))
    base_url = raw.get("base_url") if isinstance(raw.get("base_url"), str) else ""
    base_url = base_url.strip() or (env.get("AI_PASSPORT_BASE_URL") or "").strip()
    parts = urllib.parse.urlsplit(base_url) if base_url else None
    if not parts or parts.scheme not in {"http", "https"} or not parts.netloc:
        base_url = ""
    # The same credentials_path override the provider honors (config.py). The
    # hook reading a different file than the provider writes is an audit
    # surface that silently dies on exactly the installs that customized it.
    credentials_path = raw.get("credentials_path") if isinstance(raw.get("credentials_path"), str) else ""
    credentials_path = credentials_path.strip() or os.path.join(hermes_home, CREDENTIALS_FILE_NAME)
    return {
        "enabled": enabled if isinstance(enabled, bool) else True,
        "timeout_s": timeout_ms / 1000.0,
        "base_url": base_url.rstrip("/") if base_url else "",
        "credentials_path": credentials_path,
    }


def load_token(credentials_path: str, now=time.time) -> tuple[str, str] | None:
    """(base_url, access_token) from the provider-maintained credentials file,
    or None when there is no token fresh enough to use WITHOUT refreshing."""
    raw = _read_json(credentials_path)
    token = raw.get("access_token")
    expires_at = raw.get("access_token_expires_at")
    token_url = raw.get("token_url")
    if not isinstance(token, str) or not token:
        return None
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        return None
    # 30s slack: a token about to expire mid-request is not worth the 401.
    if expires_at <= now() + 30:
        return None
    parts = urllib.parse.urlsplit(token_url if isinstance(token_url, str) else "")
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}", token


def _fetch(request, timeout_s: float, opener=None) -> tuple[int, dict]:
    """One HTTP exchange. Returns (status, payload); transport failures raise."""
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout_s) as response:
            status = response.status
            text = response.read(1_048_576).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        status = err.code
        try:
            text = err.read(1_048_576).decode("utf-8", errors="replace")
        except Exception:
            text = ""
    try:
        payload = json.loads(text) if text else {}
    except Exception:
        payload = None  # a 200 that is not JSON is a middlebox, not an answer
    return status, payload if isinstance(payload, dict) else ({} if status != 200 else None)


def post_check(base_url: str, token: str, body: dict, timeout_s: float, opener=None) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}/agent/policy/check",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    return _fetch(request, timeout_s, opener=opener)


def get_snapshot(base_url: str, token: str, timeout_s: float, opener=None) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}/agent/policy/snapshot",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    return _fetch(request, timeout_s, opener=opener)


def _maybe_refresh_snapshot(state: "PolicyState", base_url: str, token: str, timeout_s: float, *, opener=None, now=time.time) -> None:
    """Refresh the cached rule snapshot when enforce mode needs it, best effort.

    Gated on the snapshot's OWN mode saying enforce: audit installs never pay
    a snapshot request (the mode arrives free on every check answer, which is
    also how an enforce flip creates the stub that turns this on). A failed
    refresh keeps the old snapshot: stale rules beat no rules when the plane
    goes dark right after, which is exactly when they get used.
    """
    snapshot = state.snapshot
    if not isinstance(snapshot, dict) or snapshot.get("mode") != "enforce":
        return
    if now() - (snapshot.get("fetched_at") or 0) < SNAPSHOT_TTL_S:
        return
    if state.budget_exhausted():
        return
    state.spend()
    try:
        status, payload = get_snapshot(base_url, token, timeout_s, opener=opener)
    except Exception:
        return
    version = payload.get("version") if isinstance(payload, dict) else None
    if status != 200 or not isinstance(payload, dict) or not isinstance(version, (int, float)) or version > SNAPSHOT_VERSION:
        # Unknown protocol or a refusal: half-understood rules are worse than
        # the documented stale/unknown posture.
        return
    rules = payload.get("rules")
    state.snapshot = {
        "mode": "enforce" if payload.get("mode") == "enforce" else "audit",
        "rules": rules if isinstance(rules, list) else [],
        "fetched_at": now(),
    }


def _local_verdict(state: "PolicyState", tool_name: str, *, degraded: bool = False) -> dict | None:
    """The enforce-mode outage posture: the last-known rules decide locally.

    Fail closed exactly for the tools the owner's rules constrain, fail open
    for everything else; without a snapshot (or in audit mode) there is no
    verdict and the caller stays silent. A mode stub learned from a check
    answer (fetched_at 0: enforce known, rules never fetched) also stays
    silent: evaluating an empty list would fail open ANYWAY for the tools the
    owner constrained, and pretending it was a rules decision would be worse
    than the honest unknown posture.

    ``degraded`` names the failure honestly: a no-opinion 200 means the
    Passport ANSWERED and only its policy store is down, and telling the
    owner it is "unreachable" would send them debugging connectivity and
    credentials while the service is up.
    """
    snapshot = state.snapshot
    if not isinstance(snapshot, dict) or snapshot.get("mode") != "enforce":
        return None
    if not snapshot.get("fetched_at"):
        return None
    local = resolve_local_policy(tool_name, snapshot.get("rules") or [])
    if local["action"] == "deny":
        why = (
            "their Passport's policy service is temporarily degraded"
            if degraded
            else "their Passport is currently unreachable"
        )
        return {
            "effective": "deny",
            "message": f"Blocked by the owner's AI Passport tool policy (rule {local['matched_pattern']}); {why}.",
        }
    if local["action"] == "require_approval":
        why = (
            "their AI Passport's policy service is temporarily degraded"
            if degraded
            else "their AI Passport is unreachable"
        )
        return {
            "effective": "deny",
            "message": (
                f"This tool needs the owner's approval (rule {local['matched_pattern']}) and {why}, "
                "so no approval can be requested right now. Retry later."
            ),
        }
    return None


# EVERY key _is_no_opinion reads. _replayable includes this tuple wholesale,
# so a cached body always classifies exactly as its first arrival did; a new
# classifier input belongs HERE, never read directly off the answer.
_CLASSIFIER_KEYS = ("mode", "degraded", "degraded_reason")


def _replayable(answer: dict) -> dict:
    """Only what a replay must be able to act on; never event_id, so a
    replayed answer cannot claim a recording that did not happen, and never
    "message", which only locally synthesized verdicts carry and those are
    never cached (the cache holds the WIRE answer, never a frozen local
    verdict). The classifier keys stay so _deliver classifies a replay
    exactly as it classified the first arrival."""
    keys = ("effective", "decision", "approval_url") + _CLASSIFIER_KEYS
    return {key: answer.get(key) for key in keys if answer.get(key) is not None}


def _is_no_opinion(answer: dict) -> bool:
    """The rules-unreachable degraded flavor: the plane had NO OPINION
    (nothing was readable), as opposed to a real verdict whose event
    RECORDING failed. Named on the wire by ``degraded_reason`` (phase 6).
    Only the KNOWN verdict reason is trusted as a verdict; everything else,
    including reason strings this plugin has never heard of, falls back to
    the mode heuristic (the no-opinion body hardcodes mode audit). Plugins
    live on owner machines for years while the backend deploys continuously,
    so a future no-opinion flavor must not be mistaken for a verdict that
    teaches its hardcoded audit mode. Reads only _CLASSIFIER_KEYS, all of
    which _replayable preserves."""
    if answer.get("degraded") is not True:
        return False
    reason = answer.get("degraded_reason")
    if reason == "rules_unavailable":
        return True
    if reason == "event_write_failed":
        return False
    return answer.get("mode") != "enforce"


def _deliver(state: "PolicyState", tool_name: str, answer: dict) -> dict:
    """Every answer leaves run_check through here, fresh or replayed, so a
    body is classified and resolved exactly once and identically at both
    exits. A no-opinion body defers to the last-known rules (its wire allow
    covers the tools they do not constrain), evaluated on EVERY call so a
    rule change or a snapshot refresh landing between identical repeats
    reaches the next one instead of being frozen out by a cached local
    verdict for the TTL; anything else IS the verdict."""
    if _is_no_opinion(answer):
        local = _local_verdict(state, tool_name, degraded=True)
        return local if local is not None else answer
    return answer


def run_check(hermes_home: str, tool_name, tool_input, session_id, *, now=time.time, opener=None) -> dict | None:
    """Decide one tool call. Returns the check payload on a usable 200, a
    locally synthesized verdict (carrying ``message``) when the plane does not
    answer and the last snapshot says enforce, else None (silence). Every path
    updates and persists the state file."""
    settings = load_settings(hermes_home)
    if not settings["enabled"]:
        return None
    if not isinstance(tool_name, str) or not TOOL_SHAPE.match(tool_name):
        return None

    state = PolicyState(hermes_home, now=now)
    digest = args_digest(tool_input)
    cache_key = f"{tool_name} {digest or ''}"
    try:
        # A cached answer IS the answer: an identical repeat inside the TTL
        # must get the same verdict (an enforce deny keeps denying), it just
        # costs no network. _deliver is the one chokepoint both this exit and
        # the fresh-answer exit share: a no-opinion body is never a verdict,
        # so its replay re-runs the local rules, same as its first arrival.
        cached = state.cached(cache_key)
        if cached is not None:
            return _deliver(state, tool_name, cached)

        # The backoff gate comes BEFORE any file I/O: a long-dead install must
        # cost one stat per backoff window, not one per tool call.
        if not state.backed_off():
            loaded = load_token(settings["credentials_path"], now=now)
            if loaded is None:
                # No fresh token and refreshing is not this process's right.
                # The provider's next prefetch will refresh; the backoff is
                # what keeps this from re-stating the file every call.
                state.back_off("auth")
            elif not state.budget_exhausted():
                derived_base, token = loaded
                base_url = settings["base_url"] or derived_base
                _maybe_refresh_snapshot(state, base_url, token, settings["timeout_s"], opener=opener, now=now)
                # The snapshot refresh may have spent the window's last slot;
                # the check POST re-checks rather than going one over.
                if not state.budget_exhausted():
                    body = {"tool": tool_name}
                    if digest:
                        body["args_digest"] = digest
                    if isinstance(session_id, str) and session_id.strip():
                        body["session_key"] = session_id[:SESSION_KEY_MAX]

                    state.spend()
                    try:
                        status, payload = post_check(base_url, token, body, settings["timeout_s"], opener=opener)
                    except Exception:
                        state.back_off("unavailable")
                    else:
                        if status == 200 and payload is not None:
                            state.clear_backoff()
                            ttl = payload.get("cache_ttl")
                            single_use = isinstance(ttl, (int, float)) and ttl <= 0
                            ttl_s = ttl if isinstance(ttl, (int, float)) else DEFAULT_CACHE_TTL_S
                            no_opinion = _is_no_opinion(payload)
                            # Every answer that NAMES its mode teaches it
                            # (keep it fresh so a flip reaches the next call,
                            # and learn a stub when no snapshot exists yet;
                            # fetched_at 0 makes the next enforce call fetch
                            # the real rules). Recording-degraded verdicts
                            # (event_write_failed) teach too: while the event
                            # table is down they are the only flip detector,
                            # in BOTH directions. Only the no-opinion flavor
                            # is mute; its mode is hardcoded audit and must
                            # never overwrite the learned one.
                            if not no_opinion and payload.get("mode") in ("audit", "enforce"):
                                if isinstance(state.snapshot, dict):
                                    state.snapshot["mode"] = payload.get("mode")
                                else:
                                    state.snapshot = {"mode": payload.get("mode"), "rules": [], "fetched_at": 0}
                            # An approval answer is never cached (ONE wait on
                            # ONE event), nor anything the server marks
                            # single-use with cache_ttl <= 0 (a grant-consumed
                            # allow: allow_once means once), nor the degraded
                            # approval fall-open (decision require_approval
                            # with the wait unrecordable): that allow admits
                            # exactly the ONE call whose wait could not be
                            # recorded, and the mode-gated decision clause is
                            # the belt for a backend that predates its
                            # cache_ttl 0 (mode-gated because an AUDIT answer
                            # for an approval-ruled tool during the same
                            # outage enforced nothing and must keep its ttl,
                            # or a degraded plane loses its dedupe shield).
                            # Degraded bodies cache the WIRE answer, never a
                            # frozen local verdict: _deliver re-runs the local
                            # rules on every exit, replays included, so a
                            # synthesized "retry later" deny cannot outlive
                            # the outage it named. The cache is still what
                            # keeps per-call checks from hammering an
                            # already-degraded plane.
                            fall_open = (
                                payload.get("degraded") is True
                                and payload.get("mode") == "enforce"
                                and payload.get("decision") == "require_approval"
                            )
                            if not single_use and not fall_open and payload.get("effective") != "require_approval":
                                state.remember(cache_key, ttl_s, _replayable(payload))
                            return _deliver(state, tool_name, payload)
                        if status == 200:
                            state.back_off("unavailable")  # malformed 200: middlebox
                        elif status == 429:
                            state.back_off("rate_limited")
                        elif status in (403, 404):
                            state.back_off("forbidden")
                        elif status == 401:
                            # The provider owns refresh; one 401 here means
                            # the file's token aged out between our freshness
                            # check and the server's.
                            state.back_off("auth")
                        elif status == 400:
                            state.back_off("invalid_request")
                        else:
                            state.back_off("unavailable")

        # No answer from the plane (no token, backed off, out of budget, or
        # the request failed): in enforce mode the last-known rules decide.
        return _local_verdict(state, tool_name)
    finally:
        state.save()


def hook_response(payload: dict | None) -> dict | None:
    """Translate a check answer into the hook wire shape, or None for silence.

    Audit mode always answers ``effective: allow`` so this returns None on
    every phase-4 path; the block branch exists because the wire contract
    already carries ``effective`` and a phase-5 enforce answer must not be
    silently ignored by an already-installed hook.
    """
    if not isinstance(payload, dict):
        return None
    effective = payload.get("effective")
    if effective == "deny":
        # `message` only ever arrives on locally synthesized verdicts (the
        # wire payload has no such field), where it names the rule and the
        # outage so the agent can tell the owner something actionable.
        return {"action": "block", "message": payload.get("message") or "Blocked by the owner's AI Passport tool policy."}
    if effective == "require_approval":
        # Veto-plus-link is the ceiling on Hermes: a shell hook process cannot
        # wait minutes for the owner, so it blocks NOW with the link and the
        # agent retries after the owner approves (the retry's check lands on
        # the resolved state or mints a fresh wait).
        approval = payload.get("approval_url")
        suffix = f" The owner can approve it at {approval}." if isinstance(approval, str) and approval else ""
        return {
            "action": "block",
            "message": f"This tool call is awaiting the owner's approval in their AI Passport.{suffix} Retry after they approve.",
        }
    return None


def selftest(hermes_home: str) -> int:
    """A human-readable probe for the install guide and the release evidence."""
    settings = load_settings(hermes_home)
    if not settings["enabled"]:
        print("tool-call audit: off (policy.enabled=false in ai-passport.json)")
        return 0
    loaded = load_token(settings["credentials_path"])
    if loaded is None:
        print(f"tool-call audit: no fresh access token at {settings['credentials_path']}; run one Hermes turn first")
        return 1
    # Unique probe args per run: a re-run within the dedupe TTL must be a real
    # probe (the first draft's constant args hit the cache and flagged a
    # perfectly healthy install as broken on the second run).
    payload = run_check(hermes_home, "ai_passport.policy_selftest", {"probe": time.time()}, "selftest")
    state = PolicyState(hermes_home)
    if payload is None:
        if state.backed_off():
            print(f"tool-call audit: check did not land (backed off: {state.backoff_reason})")
        elif state.budget_exhausted():
            print("tool-call audit: check skipped (this minute's request budget is spent); try again shortly")
        else:
            print("tool-call audit: check did not land")
        return 1
    print(
        "tool-call audit: reporting"
        + f" (mode {payload.get('mode', '?')}, decision {payload.get('decision', '?')},"
        + f" event {'recorded' if payload.get('event_id') else 'NOT recorded'})"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI Passport pre_tool_call policy hook")
    parser.add_argument("--home", required=True, help="Hermes home directory (wired by post_setup)")
    parser.add_argument("--selftest", action="store_true", help="Probe the policy plane and print one line")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest(args.home)

    # The real hook path: everything is best-effort and silence is success.
    try:
        event = json.loads(sys.stdin.read() or "{}")
        if not isinstance(event, dict):
            return 0
        payload = run_check(args.home, event.get("tool_name"), event.get("tool_input"), event.get("session_id"))
        answer = hook_response(payload)
        if answer is not None:
            print(json.dumps(answer))
    except Exception:
        # Fail open: in audit mode no failure here is worth a tool call, and
        # exit code 2 is the one thing this script must never emit by accident.
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
