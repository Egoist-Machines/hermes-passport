"""Connect-code credentials and OAuth refresh, owned by the provider.

The install guide writes ``{token_url, client_id, refresh_token}`` to
``<hermes_home>/ai-passport-refresh.json`` at mode 600 and teaches the AGENT to
refresh the ai-passport MCP server's static bearer from it, so two refreshers
share one single-use rotating refresh token. Refresh tokens rotate and replay
detection revokes the affected token family, so only one process should own
refresh for an install; the rotation gate below keeps concurrent provider calls
safe. A cross-process race (the agent's own curl rotating first) is covered
server-side: for five minutes the backend keeps a sealed copy of the response
it issued and returns it again to a replay from the same client address, so
both refreshers converge on one successor chain rather than fighting. The file
on disk stays the source of truth, re-read whenever its mtime moves, so the
agent's rotation is picked up rather than overwritten.

The provider also hands each new access token to the env sink (env_sink.py),
which keeps the agent from needing to refresh at all.

A refresh POST is NOT idempotent: the server rotates the single-use refresh
token as it answers, so a caller that gives up waiting must never cancel one in
flight. Nothing here takes a deadline. Callers that are on a latency budget
(the ambient prefetch) hand the call to a worker thread, stop WAITING on it, and
let it finish and persist on its own.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional

from .transport import open_request

logger = logging.getLogger(__name__)

REFRESH_MARGIN_S = 5 * 60
TOKEN_REQUEST_TIMEOUT_S = 10.0
# A token response is a small JSON object. Anything larger is a middlebox or a
# misrouted request, and reading it unbounded would hand a hostile endpoint the
# provider's memory.
MAX_TOKEN_BODY_BYTES = 64 * 1024

RECONNECT_HINT = (
    "AI Passport refresh token is spent or revoked. Ask the owner for a new connect code "
    "and reinstall (ego.ist/hermes)."
)


class PassportAuthError(Exception):
    def __init__(self, message: str, *, terminal: bool = False, code: str = "auth_failed"):
        super().__init__(message)
        self.terminal = terminal
        self.code = code


class _Parsed:
    __slots__ = ("raw", "token_url", "client_id", "refresh_token", "access_token", "access_token_expires_at")

    def __init__(self, raw: dict, token_url: str, client_id: str, refresh_token: str,
                 access_token: Optional[str], access_token_expires_at: float):
        self.raw = raw
        self.token_url = token_url
        self.client_id = client_id
        self.refresh_token = refresh_token
        self.access_token = access_token
        self.access_token_expires_at = access_token_expires_at


def _parse_credentials(text: str) -> _Parsed:
    try:
        parsed = json.loads(text)
    except Exception:
        raise PassportAuthError("AI Passport credentials file is not valid JSON.", code="invalid_file")
    if not isinstance(parsed, dict):
        raise PassportAuthError("AI Passport credentials file is not a JSON object.", code="invalid_file")
    token_url = str(parsed.get("token_url") or "").strip()
    client_id = str(parsed.get("client_id") or "").strip()
    refresh_token = str(parsed.get("refresh_token") or "").strip()
    if not token_url or not client_id or not refresh_token:
        raise PassportAuthError(
            "AI Passport credentials file is missing token_url, client_id, or refresh_token.",
            code="invalid_file",
        )
    access_token = str(parsed.get("access_token") or "").strip() or None
    expires_at = parsed.get("access_token_expires_at")
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        expires_at = 0.0
    return _Parsed(parsed, token_url, client_id, refresh_token, access_token, float(expires_at))


def _write_atomically(path: Path, payload: dict) -> None:
    """Same-directory temp file plus rename.

    A crash mid-write must never leave a half-written credentials file, and a
    cross-device temp dir would make the rename fail. Mode 600 is set on the
    temp file BEFORE it holds a token.
    """
    directory = path.parent
    fd, temp_name = tempfile.mkstemp(prefix=".ai-passport-", dir=str(directory))
    temp = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        # The temp file may hold a live token, so it is never left behind.
        try:
            temp.unlink()
        except OSError:
            pass
        raise


class Credentials:
    def __init__(
        self,
        credentials_path: Path,
        *,
        opener: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.time,
        on_access_token: Optional[Callable[[str], None]] = None,
    ):
        self._path = Path(credentials_path)
        self._opener = opener or open_request
        self._clock = clock
        self._on_access_token = on_access_token
        self._cached: Optional[_Parsed] = None
        self._cached_mtime = -1.0
        # True while a rotated token lives only in memory because the file write
        # failed. Retried on every read until the disk catches up.
        self._needs_persist = False
        # The state lock protects the cached file view and is held only for
        # in-memory work and file IO, NEVER across a network call: the ambient
        # path reads tokens on every turn and must not queue behind a slow POST.
        self._lock = threading.RLock()
        # One rotation at a time per process: the refresh token is single-use, so
        # two concurrent refreshes would spend it twice and revoke the install.
        # This gate is what serializes them; the POST runs holding ONLY this.
        self._rotation_gate = threading.Lock()
        # The proactive margin refresh runs here so readers with a still-valid
        # token never wait on it. Non-daemon on purpose: a refresh POST is not
        # idempotent, so interpreter exit must let it finish and persist.
        self._background_lock = threading.Lock()
        self._background_thread: Optional[threading.Thread] = None

    # -- reading -------------------------------------------------------------

    def _load(self) -> _Parsed:
        with self._lock:
            # While a rotated token lives only in memory, memory outranks disk:
            # the file's token is spent by definition, so re-reading it here
            # would clobber the only live copy and brick the install on the next
            # refresh. _retry_persist() is what heals the file, never a re-read.
            if self._needs_persist and self._cached is not None:
                return self._cached
            try:
                file_stat = self._path.stat()
            except FileNotFoundError:
                raise PassportAuthError(
                    f"AI Passport is not installed for this agent: {self._path} is missing. "
                    "Redeem a connect code (ego.ist/hermes).",
                    terminal=True,
                    code="not_installed",
                )
            except OSError as err:
                raise PassportAuthError(
                    f"AI Passport credentials file is unreadable: {err.errno or 'error'}.",
                    code="unreadable",
                )
            if self._cached is not None and file_stat.st_mtime == self._cached_mtime:
                return self._cached
            try:
                text = self._path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValueError) as err:
                # stat() succeeding does not make the content readable: a chmod
                # 000, a directory at this path, or a non-UTF-8 rewrite all land
                # here, and the CLI promises a plain error line, not a traceback.
                raise PassportAuthError(
                    f"AI Passport credentials file is unreadable: {type(err).__name__}.",
                    code="unreadable",
                )
            parsed = _parse_credentials(text)
            self._cached = parsed
            self._cached_mtime = file_stat.st_mtime
            return parsed

    def base_url(self) -> str:
        """The Passport origin the install was paired against.

        Deriving it from token_url means one guide serves prod and a local stack.
        """
        with self._lock:
            current = self._load()
        parts = urllib.parse.urlsplit(current.token_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise PassportAuthError("AI Passport credentials file has an unusable token_url.", code="invalid_file")
        return f"{parts.scheme}://{parts.netloc}"

    # -- writing -------------------------------------------------------------

    def _persist(self, current: _Parsed, access_token: str, refresh_token: str, expires_at: float) -> None:
        with self._lock:
            self._persist_locked(current, access_token, refresh_token, expires_at)

    def _persist_locked(self, current: _Parsed, access_token: str, refresh_token: str, expires_at: float) -> None:
        payload = dict(current.raw)
        payload.update(
            {
                "token_url": current.token_url,
                "client_id": current.client_id,
                "refresh_token": refresh_token,
                "access_token": access_token,
                "access_token_expires_at": expires_at,
            }
        )
        # Commit to memory BEFORE touching disk. The server already rotated, so
        # this process's copy of the new single-use refresh token is the only one
        # anywhere; a failed file write must degrade to "disk is behind, retry
        # later", never to losing the token (which bricks the install).
        self._cached = _Parsed(payload, current.token_url, current.client_id, refresh_token, access_token, expires_at)
        try:
            _write_atomically(self._path, payload)
            self._needs_persist = False
        except Exception as err:
            self._needs_persist = True
            # Name the danger, not just the failure: the file now holds a
            # SPENT refresh token, and any refresher that reads it (the
            # agent's documented 401 repair) replays it; more than five
            # minutes after the rotation that replay revokes the whole token
            # family, in-memory successor included. The status CLI surfaces
            # the same state as pending_disk_write.
            logger.warning(
                "ai-passport: could not write rotated credentials (%s); keeping them in memory and retrying. "
                "The on-disk refresh token is now SPENT: a manual refresh from the file can revoke this install "
                "until the write succeeds.",
                type(err).__name__,
            )
        # Re-stat rather than trusting our own write: the next load must not
        # decide the file changed under it and re-read on every single call.
        # After a FAILED write the _needs_persist guard in _load() is what
        # protects the in-memory rotation; a failed stat costs one extra read.
        try:
            self._cached_mtime = self._path.stat().st_mtime
        except OSError:
            self._cached_mtime = -1.0

    def _retry_persist(self) -> None:
        with self._lock:
            self._retry_persist_locked()

    def _retry_persist_locked(self) -> None:
        if not self._needs_persist or self._cached is None:
            return
        try:
            _write_atomically(self._path, self._cached.raw)
        except Exception:
            # Still failing; keep serving from memory and try again next read.
            return
        self._needs_persist = False
        try:
            self._cached_mtime = self._path.stat().st_mtime
        except OSError:
            self._cached_mtime = -1.0
        logger.debug("ai-passport: rotated credentials written after an earlier failure")

    # -- rotation ------------------------------------------------------------

    def _request_token(self, current: _Parsed) -> tuple[str, str, float]:
        # `resource` is deliberately NOT sent, unlike the OpenClaw twin.
        # Verified against lib/oauth.js exchangeRefreshToken (2026-08-15): the
        # audience a rotation binds is `preview.resource ?? ...`, i.e. the one
        # already stored on the chain, and connect-code redeem always stores it
        # (lib/agentConnect.js). So the parameter cannot change the outcome of
        # OUR refreshes. It can only fail them: when the value sent is not
        # byte-equal to the server's canonical issuer + /mcp, the exchange
        # answers invalid_grant, which this provider latches as TERMINAL and
        # which sends the owner off to re-pair a perfectly healthy install. We
        # would have to derive that value from token_url, and the two diverge
        # for any deployment whose public URL carries a path or whose issuer
        # and public URL are configured separately. Omitting it keeps the
        # binding (the server derives it) and drops the failure mode.
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": current.client_id,
                "refresh_token": current.refresh_token,
            }
        ).encode("ascii")
        request = urllib.request.Request(
            current.token_url,
            data=body,
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "accept": "application/json",
            },
            method="POST",
        )
        status = 0
        text = ""
        try:
            with self._opener(request, timeout=TOKEN_REQUEST_TIMEOUT_S) as response:
                status = getattr(response, "status", 200) or 200
                text = response.read(MAX_TOKEN_BODY_BYTES).decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            status = err.code
            try:
                text = err.read(MAX_TOKEN_BODY_BYTES).decode("utf-8", "replace")
            except Exception:
                text = ""
        except Exception as err:
            raise PassportAuthError(
                f"AI Passport token endpoint is unreachable: {type(err).__name__}.",
                code="unreachable",
            )

        payload: Any = None
        try:
            payload = json.loads(text) if text else None
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            payload = {}

        if 300 <= status < 400:
            # The transport refuses redirects (see transport.py); a 3xx here is
            # a middlebox answering in the token endpoint's place, never the
            # endpoint itself.
            raise PassportAuthError(
                f"AI Passport token endpoint answered a {status} redirect; refusing to follow it with a credential.",
                code="unreachable",
            )
        if status == 400 and payload.get("error") == "invalid_grant":
            raise PassportAuthError(RECONNECT_HINT, terminal=True, code="invalid_grant")
        if payload.get("error") == "invalid_client":
            # The registration this install was paired under no longer exists:
            # the owner severed it, or the deployment's client store was reset.
            # Verified against the backend (2026-08-14): its store THROWS on a
            # failed read, which the OAuth layer answers as a 500, so
            # invalid_client is never a transient blip and retrying it hourly
            # only hides a dead install behind "answered 400". The terminal latch
            # still re-probes after its recheck window, so a wrong verdict costs
            # stale context for five minutes rather than the install. Same rule
            # in the OpenClaw twin (clients/openclaw-passport/src/credentials.js).
            raise PassportAuthError(RECONNECT_HINT, terminal=True, code="invalid_client")
        if status >= 400:
            raise PassportAuthError(f"AI Passport token endpoint answered {status}.", code="token_error")

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise PassportAuthError("AI Passport token response carried no access_token.", code="token_error")
        expires_in = payload.get("expires_in")
        if not isinstance(expires_in, (int, float)) or isinstance(expires_in, bool):
            expires_in = 3600
        # A rotation is expected on every use; a server that echoes no new
        # refresh token keeps the old one working rather than losing the install.
        rotated = payload.get("refresh_token")
        refresh_token = rotated if isinstance(rotated, str) and rotated else current.refresh_token
        return access_token, refresh_token, self._clock() + max(0.0, float(expires_in))

    def _refresh(self) -> str:
        # No retry on invalid_grant. The backend's five-minute recovery window
        # covers the two-refresher race now: a replay from this same host is
        # answered with the sealed copy of the response the winner already
        # got, so the common case never reaches this code as an error at all.
        # Past that window a replay is classified as theft and revokes the
        # whole token family, so a re-read-and-retry would be spending tokens
        # from a chain that is already dead. The one case a retry could still
        # heal instantly is an IN-window replay whose transport identity does
        # not match the winner's (the backend answers invalid_grant WITHOUT a
        # family kill there, so the file's rotated token stays alive); that
        # needs the two refreshers of one install to egress from different
        # addresses mid-race, and it self-heals anyway: the terminal latch
        # re-probes within minutes, _load() picks the winner's rotation off
        # disk by mtime, and the next refresh succeeds. Not worth a retry
        # branch; same verdict as the OpenClaw twin
        # (clients/openclaw-passport/src/credentials.js).
        current = self._load()
        access_token, refresh_token, expires_at = self._request_token(current)
        self._persist(current, access_token, refresh_token, expires_at)
        self._announce(access_token)
        return access_token

    def _announce(self, access_token: str) -> None:
        if not self._on_access_token:
            return
        try:
            self._on_access_token(access_token)
        except Exception:
            # The sink is a convenience for the MCP entry, never a reason to
            # fail a refresh that already rotated the server's token.
            logger.debug("ai-passport: access-token sink raised", exc_info=True)

    def _rotate(self, observed_access_token: Optional[str]) -> str:
        """One serialized rotation. ``observed_access_token`` is the token the
        caller last saw; when the stored token has already moved past it (and is
        fresh), another rotation finished while this caller waited on the gate,
        so its answer is reused instead of spending a second refresh token.
        That collapses N concurrent 401 recoveries into one rotation.
        """
        with self._rotation_gate:
            with self._lock:
                current = self._load()
                if (
                    current.access_token
                    and current.access_token != observed_access_token
                    and current.access_token_expires_at - REFRESH_MARGIN_S > self._clock()
                ):
                    self._announce(current.access_token)
                    return current.access_token
            token = self._refresh()
            logger.debug("ai-passport: refreshed access token")
            return token

    def _start_background_rotation(self, observed_access_token: Optional[str]) -> None:
        with self._background_lock:
            if self._background_thread is not None and self._background_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._background_rotation,
                args=(observed_access_token,),
                daemon=False,
                name="ai-passport-refresh",
            )
            self._background_thread = thread
            thread.start()

    def _background_rotation(self, observed_access_token: Optional[str]) -> None:
        try:
            self._rotate(observed_access_token)
        except PassportAuthError as err:
            # The token being rotated is still valid, so a failed proactive
            # refresh costs nothing yet; the expiry path will retry and report.
            logger.debug("ai-passport: background refresh failed (%s)", err.code)
        except Exception:
            logger.debug("ai-passport: background refresh failed", exc_info=True)

    def access_token(self, *, force: bool = False) -> str:
        observed = None
        with self._lock:
            if self._needs_persist:
                self._retry_persist_locked()
            current = self._load()
            observed = current.access_token
            if not force and current.access_token:
                remaining = current.access_token_expires_at - self._clock()
                if remaining > REFRESH_MARGIN_S:
                    # Reconcile on every read, not only on rotation. Whoever
                    # holds the paired MCP bearer has to end up with the token
                    # that is actually valid, and rotation is not the only way
                    # the two drift: another process can rotate, a crash can
                    # land between the file write and the env write, or a human
                    # can edit either side. Edge-triggered syncing leaves those
                    # broken until the next rotation, which is up to an hour of
                    # 401s on every MCP tool call. The sink no-ops when the line
                    # already matches, so this costs one small file read.
                    self._announce(current.access_token)
                    return current.access_token
                if remaining > 0:
                    # Inside the refresh margin but still valid: serve it and
                    # rotate in the background. Blocking here would queue every
                    # concurrent reader behind a network POST for a token they
                    # already hold, and the ambient path reads on every turn.
                    self._start_background_rotation(observed)
                    self._announce(current.access_token)
                    return current.access_token
        # Expired or forced: the caller genuinely needs a rotation's answer.
        # The state lock is NOT held across this; the gate serializes it.
        return self._rotate(observed)

    # -- introspection -------------------------------------------------------

    def installed(self) -> bool:
        """Cheap, no-network answer for ``is_available()``."""
        return self._path.exists()

    def summary(self) -> dict:
        """Content-free description for the status CLI. Never the token itself."""
        with self._lock:
            try:
                current = self._load()
            except PassportAuthError as err:
                return {"ok": False, "error": err.args[0], "code": err.code}
            remaining = int(current.access_token_expires_at - self._clock()) if current.access_token else 0
            return {
                "ok": True,
                "path": str(self._path),
                "client_id": current.client_id,
                "token_url": current.token_url,
                "has_access_token": bool(current.access_token),
                "access_token_expires_in_s": max(0, remaining),
                "pending_disk_write": self._needs_persist,
            }

    def _reset_for_tests(self) -> None:
        with self._lock:
            self._cached = None
            self._cached_mtime = -1.0
            self._needs_persist = False

    def _join_background_rotation(self, timeout: float = 5.0) -> None:
        """Test seam: wait for an in-flight proactive refresh to land."""
        with self._background_lock:
            thread = self._background_thread
        if thread is not None:
            thread.join(timeout)
