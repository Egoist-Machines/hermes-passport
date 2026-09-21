"""The ``/agent/prefetch`` client (issue #425 phase 1 plane).

Contract this side of the wire commits to:

- it NEVER raises. Its callers sit on paths where an exception is worse than an
  empty answer: a raising prefetch costs the turn, and a raising tool call
  costs the model's whole answer. Every failure resolves to the last good
  answer, or to None.
- it never nags. The route is side-effect-free by construction (no access
  requests, no pass claims, no receipts), so polling it is cheap for the owner;
  the provider keeps it cheap for the backend with a TTL cache and by honoring
  429 with a real backoff instead of a retry loop.
- it logs a given failure class once per backoff window, not per turn.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from .cache import TtlCache
from .credentials import Credentials, PassportAuthError
from .transport import open_request

logger = logging.getLogger(__name__)

# The backend's own request bounds (lib/agentBackendRoutes.js). Clamping here
# rather than in each caller means a long user prompt or an unusual session key
# can never turn into a 400 that backs the whole provider off. The backend
# checks JavaScript string .length, which counts UTF-16 CODE UNITS, so the
# clamp below counts the same way: a Python code-point slice would pass an
# astral-heavy prompt (a run of emoji) at up to twice the backend's bound and
# the resulting 400 latches the five-minute invalid_request backoff.
QUERY_MAX_LENGTH = 256
SESSION_KEY_MAX_LENGTH = 128
LIMIT_MAX = 50


def _clamp_utf16(text: str, max_units: int) -> str:
    units = 0
    out = []
    for ch in text:
        units += 2 if ord(ch) > 0xFFFF else 1
        if units > max_units:
            break
        out.append(ch)
    return "".join(out)

# A prefetch answer is a bounded page of memory rows. A body past this is a
# middlebox or a version skew, not the plane, and reading it unbounded would
# hand whatever answered the provider's memory.
MAX_BODY_BYTES = 2 * 1024 * 1024

BACKOFF_S = {
    # 429 is the backend telling us our own cadence is wrong; pause for at least
    # one cache window so the next call is served from cache anyway.
    "rate_limited": 60.0,
    # 403/404 mean the plane is closed to this install: the agent backend is
    # disabled on that deployment (it ships behind a flag and Express answers
    # its default 404 while it is dark), or this client is not an agent-connect
    # registration. Nothing the provider does will change that soon, so a
    # REPEAT latches the long window. A single 403 can also be one transient
    # store blip during the backend's token verification (it deliberately
    # leaves a failed client lookup uncached so a blip costs one read, not a
    # lockout), so the first one only pauses for forbidden_probe and the next
    # read re-probes instead of amplifying the blip into a five-minute blackout.
    "forbidden": 5 * 60.0,
    "forbidden_probe": 60.0,
    # 400 means the provider sent a shape the backend refuses: a version skew,
    # not a transient fault. Back off hard and say so once.
    "invalid_request": 5 * 60.0,
    "unavailable": 30.0,
    "auth": 60.0,
}

# A terminal verdict (not installed, refresh token dead) stops network traffic,
# but the documented recovery is the owner rewriting the credentials file, and a
# long-running gateway must pick that up without a restart. Probe again after
# this window; an unchanged file re-latches at the cost of at most one stat
# (not_installed) or one token POST (invalid_grant) per window.
TERMINAL_RECHECK_S = 5 * 60.0

# Client-side share of the backend's per-client throttle (60/min). All
# concurrent sessions of one install share a client_id, and novel prompt text
# makes most context reads cache misses, so an uncapped process can trip the
# backend throttle and black out EVERY surface for a minute. Spending at most
# half the budget leaves headroom for the install's other processes (gateway,
# cron, CLI) before the backend has to say 429.
REQUEST_BUDGET_WINDOW_S = 60.0
REQUEST_BUDGET_MAX = 30


def clamp_query(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    trimmed = " ".join(value.split()).strip()
    if not trimmed:
        return None
    return _clamp_utf16(trimmed, QUERY_MAX_LENGTH)


def clamp_session_key(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return _clamp_utf16(trimmed, SESSION_KEY_MAX_LENGTH)


def clamp_limit(value: Any, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = fallback
    return min(LIMIT_MAX, max(1, int(round(value))))


class PrefetchResult:
    __slots__ = ("rows", "skipped", "approval_url", "stale")

    def __init__(self, rows: list, skipped: list, approval_url: Optional[str], stale: bool):
        self.rows = rows
        self.skipped = skipped
        self.approval_url = approval_url
        self.stale = stale


class _Pending:
    """One in-flight read, shared by every caller of the same cache key."""

    def __init__(self):
        self.done = threading.Event()
        self.value: Optional[PrefetchResult] = None


def _normalize(payload: Any) -> tuple[list, list, Optional[str]]:
    """The backend answers a closed vocabulary.

    Anything else is a version skew and is dropped rather than forwarded into a
    prompt.
    """
    rows = []
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        for row in payload["rows"]:
            if not isinstance(row, dict):
                continue
            memory_id = row.get("memory_id")
            content = row.get("content")
            if not isinstance(memory_id, str) or not isinstance(content, str):
                continue
            rows.append(
                {
                    "memory_id": memory_id,
                    "content": content,
                    "category": row["category"] if isinstance(row.get("category"), str) else "other",
                    "created_at": row["created_at"] if isinstance(row.get("created_at"), str) else None,
                    "source": row["source"] if isinstance(row.get("source"), str) else None,
                }
            )
    skipped = []
    if isinstance(payload, dict) and isinstance(payload.get("skipped_categories"), list):
        for entry in payload["skipped_categories"]:
            if not isinstance(entry, dict):
                continue
            category = entry.get("category")
            reason = entry.get("reason")
            if isinstance(category, str) and isinstance(reason, str):
                skipped.append({"category": category, "reason": reason})
    approval_url = None
    if isinstance(payload, dict) and isinstance(payload.get("approval_url"), str):
        approval_url = payload["approval_url"]
    return rows, skipped, approval_url


class PassportClient:
    def __init__(
        self,
        *,
        config,
        credentials: Credentials,
        opener: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._credentials = credentials
        self._opener = opener or open_request
        self._clock = clock
        self._cache = TtlCache(config.cache_ttl_ms, clock=clock)
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._backoff_until = 0.0
        self._backoff_reason: Optional[str] = None
        # Consecutive plane-closed answers, so one transient 403 (a store blip
        # on the backend's verification path) is probed past quickly while a
        # genuinely closed plane still converges to the long backoff on its
        # second answer.
        self._forbidden_streak = 0
        self._terminal_reason: Optional[str] = None
        self._terminal_recheck_at = 0.0
        self._request_log: list[float] = []

    # -- bookkeeping ---------------------------------------------------------

    def _budget_exhausted(self) -> bool:
        cutoff = self._clock() - REQUEST_BUDGET_WINDOW_S
        while self._request_log and self._request_log[0] <= cutoff:
            self._request_log.pop(0)
        return len(self._request_log) >= REQUEST_BUDGET_MAX

    def _back_off(self, reason: str, detail: str = "", seconds: Optional[float] = None) -> None:
        window = BACKOFF_S.get(reason, BACKOFF_S["unavailable"]) if seconds is None else seconds
        already = self._backoff_until > self._clock() and self._backoff_reason == reason
        self._backoff_until = self._clock() + window
        self._backoff_reason = reason
        if not already:
            logger.warning("ai-passport: prefetch %s%s", reason, f" ({detail})" if detail else "")

    def _latch_terminal(self, err: PassportAuthError) -> None:
        self._terminal_reason = err.code
        self._terminal_recheck_at = self._clock() + TERMINAL_RECHECK_S
        logger.warning("ai-passport: %s", err.args[0])

    # -- the wire ------------------------------------------------------------

    def _request_once(self, base_url: str, access_token: str, body: dict, timeout_s: float):
        request = urllib.request.Request(
            f"{base_url}/agent/prefetch",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "authorization": f"Bearer {access_token}",
                "content-type": "application/json",
                "accept": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=timeout_s) as response:
                status = getattr(response, "status", 200) or 200
                text = response.read(MAX_BODY_BYTES).decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            status = err.code
            try:
                text = err.read(MAX_BODY_BYTES).decode("utf-8", "replace")
            except Exception:
                text = ""
        malformed = False
        payload: Any = None
        try:
            payload = json.loads(text) if text else None
        except Exception:
            malformed = True
        return status, payload, malformed

    # -- public --------------------------------------------------------------

    def prefetch(
        self,
        *,
        categories,
        query: Optional[str] = None,
        limit: int,
        session_key: Optional[str] = None,
        timeout_s: float,
    ) -> Optional[PrefetchResult]:
        """Ask for the owner-approved rows in ``categories``.

        Returns a PrefetchResult, or None when nothing is knowable. Never raises.
        """
        bounded_query = clamp_query(query)
        bounded_session_key = clamp_session_key(session_key)
        bounded_limit = clamp_limit(limit, 20)
        # session_key is deliberately NOT part of the cache key: the backend
        # validates it and then ignores it (reserved for the policy plane's
        # audit events), so the answer is identical across sessions and keying
        # on it would turn one read into a cache miss per concurrent session.
        cache_key = json.dumps([list(categories), bounded_limit, bounded_query or ""], sort_keys=True)

        cached = self._cache.get(cache_key)
        if cached.hit and cached.fresh:
            rows, skipped, approval_url = cached.value
            return PrefetchResult(rows, skipped, approval_url, False)

        # In-flight de-duplication by cache key: the TTL cache stores only
        # COMPLETED answers, so at cold start or on TTL expiry N concurrent
        # turns of one gateway would otherwise each spend the request budget,
        # and the backend's 60/min ceiling, on N copies of one identical read.
        with self._lock:
            pending = self._pending.get(cache_key)
            leader = pending is None
            if leader:
                pending = _Pending()
                self._pending[cache_key] = pending

        if not leader:
            # Wait no longer than this caller's own budget. A follower that
            # times out serves stale rather than holding the turn open for the
            # leader's slower read.
            if pending.done.wait(timeout=timeout_s):
                return pending.value
            return self._serve_stale(cache_key)

        result = None
        try:
            result = self._prefetch_once(
                cache_key=cache_key,
                categories=categories,
                query=bounded_query,
                session_key=bounded_session_key,
                limit=bounded_limit,
                timeout_s=timeout_s,
            )
        except Exception:
            # Belt and braces: the body below is written not to raise, and a
            # bug there must still not cost the turn.
            logger.debug("ai-passport: prefetch raised", exc_info=True)
            result = self._serve_stale(cache_key)
        finally:
            # Publish to followers BEFORE releasing the key: popping first
            # would let a new caller become a second leader and duplicate the
            # read while this one's followers are still waiting on the event.
            pending.value = result
            pending.done.set()
            with self._lock:
                self._pending.pop(cache_key, None)
        return result

    def _serve_stale(self, cache_key: str) -> Optional[PrefetchResult]:
        cached = self._cache.get(cache_key)
        if not cached.hit:
            return None
        rows, skipped, approval_url = cached.value
        return PrefetchResult(rows, skipped, approval_url, True)

    def _prefetch_once(self, *, cache_key, categories, query, session_key, limit, timeout_s):
        now = self._clock()
        if self._terminal_reason:
            if now < self._terminal_recheck_at:
                return self._serve_stale(cache_key)
            # Clear the verdict and fall through: the credentials layer re-reads
            # the file on mtime change, so a reinstall is picked up here.
            self._terminal_reason = None
        if self._backoff_until > now:
            return self._serve_stale(cache_key)
        if self._budget_exhausted():
            return self._serve_stale(cache_key)

        try:
            base_url = self._config.base_url or self._credentials.base_url()
            access_token = self._credentials.access_token()
        except PassportAuthError as err:
            if err.terminal:
                # Not installed, or the refresh token is genuinely dead. Say it
                # once and stop touching the network until the recheck window
                # probes for the owner's reinstall.
                self._latch_terminal(err)
                return self._serve_stale(cache_key)
            self._back_off("auth", err.code)
            return self._serve_stale(cache_key)
        except Exception as err:
            self._back_off("auth", type(err).__name__)
            return self._serve_stale(cache_key)

        # session_key has no pass-lifecycle effect today (an agent session pass
        # is a plain 24h cap and concurrent sessions of one install share it);
        # it is sent so the provider's request contract is already right for the
        # policy plane's per-session audit events.
        body: dict[str, Any] = {"categories": list(categories), "limit": limit}
        if query:
            body["query"] = query
        if session_key:
            body["session_key"] = session_key

        try:
            self._request_log.append(self._clock())
            status, payload, malformed = self._request_once(base_url, access_token, body, timeout_s)
            if status == 401:
                # The stored access token outlived its hour, or the server
                # rotated out from under us. One forced refresh, one retry, then
                # give up for this window; the refresh path owns the terminal
                # cases.
                refreshed = self._credentials.access_token(force=True)
                self._request_log.append(self._clock())
                status, payload, malformed = self._request_once(base_url, refreshed, body, timeout_s)
        except PassportAuthError as err:
            if err.terminal:
                self._latch_terminal(err)
                return self._serve_stale(cache_key)
            self._back_off("auth", err.code)
            return self._serve_stale(cache_key)
        except Exception as err:
            # A socket timeout, a DNS failure and a reset connection all mean
            # the same thing to the caller.
            self._back_off("unavailable", type(err).__name__)
            return self._serve_stale(cache_key)

        if status == 200:
            if malformed or not isinstance(payload, dict):
                # A 200 whose body is not the plane's JSON object is a middlebox
                # answering in the backend's place (captive portal, proxy error
                # page), not an empty Passport. Caching it as a fresh empty
                # answer would have the agent tell the owner their Passport
                # holds nothing; degrade like any other transport fault instead.
                self._back_off("unavailable", "malformed_200")
                return self._serve_stale(cache_key)
            rows, skipped, approval_url = _normalize(payload)
            self._cache.set(cache_key, (rows, skipped, approval_url))
            self._backoff_until = 0.0
            self._backoff_reason = None
            self._forbidden_streak = 0
            return PrefetchResult(rows, skipped, approval_url, False)
        if status == 429:
            self._back_off("rate_limited")
            return self._serve_stale(cache_key)
        if status in (403, 404):
            # 403 is the scope guard refusing this install; 404 is a deployment
            # that never mounted /agent/prefetch (the plane ships behind a flag,
            # and while it is dark Express answers its default 404). Both mean
            # the same thing to the owner, and neither is an outage.
            # The streak only advances when no forbidden backoff is already
            # standing: one turn's two ambient reads run concurrently, and both
            # answering 403 on one transient blip must not count as the repeat
            # that latches the five-minute window.
            if not (self._backoff_reason == "forbidden" and self._backoff_until > self._clock()):
                self._forbidden_streak += 1
            self._back_off(
                "forbidden",
                f"{status}: ask the owner to enable the AI Passport agent backend",
                BACKOFF_S["forbidden"] if self._forbidden_streak > 1 else BACKOFF_S["forbidden_probe"],
            )
            return self._serve_stale(cache_key)
        if status == 401:
            self._back_off("auth", "401 after refresh")
            return self._serve_stale(cache_key)
        if status == 400:
            error = payload.get("error") if isinstance(payload, dict) else None
            self._back_off("invalid_request", str(error or "bad_request"))
            return self._serve_stale(cache_key)
        self._back_off("unavailable", str(status))
        return self._serve_stale(cache_key)

    def state(self) -> dict:
        """Content-free introspection for the status CLI and tests."""
        return {
            "backoff_reason": self._backoff_reason if self._backoff_until > self._clock() else None,
            "terminal_reason": self._terminal_reason,
            "cache_size": self._cache.size,
            "requests_in_window": len(self._request_log),
        }
