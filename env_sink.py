"""Keep the ai-passport MCP bearer in step with the token this provider mints.

The Hermes install guide configures the ai-passport MCP server with a static
bearer read from ``$AI_PASSPORT_TOKEN`` in ``<hermes_home>/.env``. That token
expires after an hour. Before this provider existed, the agent itself noticed the
401 and rewrote the line. Now the provider refreshes on its own schedule and
rotates the single-use refresh token, so it owes the MCP entry the token it just
minted; otherwise the two halves of the install fight over one credential.

REWRITE-ONLY, and that is the whole design. The provider updates the
``AI_PASSPORT_TOKEN`` line only when the line already exists, matched with the
exact case dotenv keys have. A missing line is the owner's opt-out: they run MCP
with OAuth, or with a hand-managed header, or not at all, and a plugin that
helpfully appends a bearer token to their .env would be adding a credential to a
file they deliberately kept clean. Nothing here creates the file either.

LEVEL-TRIGGERED, deliberately. sync() runs on every access-token read
(credentials.py announces on each), and it re-reads the file each time rather
than memoizing the last token it wrote: rotation is not the only way the file
drifts (the owner edits it, a backup restore rewrites it), and an edge-triggered
sink leaves that drift broken until the next rotation, which is up to an hour of
401s on every MCP tool call. The cost is one small file read per token read.

A failed sync is not fatal: the agent's documented 401 path still repairs the
entry, and it normally finds the rotated refresh token the provider already
persisted. The exception is a credentials-file write failure (pending_disk_write
in the status CLI): there the file still holds the SPENT token, and the agent's
file-based refresh would replay it, which past the server's five-minute recovery
window revokes the whole token family. credentials.py warns loudly in that state
and retries the write on every read.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _env_line_safe(value: str) -> str:
    """Neutralize characters that would break .env line structure.

    ``.env`` is strictly line-oriented and values are interpolated straight into
    the line, so an embedded CR/LF would spill onto a new line and be re-parsed
    as a SEPARATE ``KEY=VALUE`` entry. Mirrors the host's own writer
    (hermes_cli/memory_setup._env_line_safe).
    """
    text = value if isinstance(value, str) else str(value)
    return "".join(text.replace("\x00", "").splitlines())


class EnvTokenSink:
    def __init__(self, *, hermes_home: str, var: str, enabled: bool = True):
        self._path = Path(hermes_home) / ".env"
        self._var = var
        self._enabled = enabled
        # dotenv keys are case-sensitive, so the match is too: a lowercase
        # ai_passport_token is someone else's variable, and rewriting it would
        # both break its consumer and add a credential to a file whose
        # canonical-cased line is absent, which is the owner's opt-out.
        self._pattern = re.compile(rf"^(\s*(?:export\s+)?){re.escape(var)}\s*=")
        # One warning per failure streak, not per read: sync runs on every
        # token read, and a persistently unwritable file would otherwise flood
        # the log with the same line.
        self._write_failure_warned = False

    def __call__(self, access_token: str) -> bool:
        return self.sync(access_token)

    def sync(self, access_token: str) -> bool:
        """Rewrite the token line if it exists and has drifted. Returns True on a write."""
        if not self._enabled or not access_token:
            return False
        try:
            if not self._path.exists():
                return False
            original = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as err:
            logger.debug("ai-passport: could not read %s (%s)", self._path, type(err).__name__)
            return False

        safe = _env_line_safe(access_token)
        lines = original.splitlines(keepends=True)
        found = False
        changed = False
        for index, line in enumerate(lines):
            match = self._pattern.match(line)
            if not match:
                continue
            found = True
            newline = "\n" if line.endswith("\n") else ""
            replacement = f"{match.group(1)}{self._var}={safe}{newline}"
            if replacement != line:
                # Every occurrence, not the first: the last line wins at read
                # time, so updating only the first would leave a stale token in
                # charge.
                lines[index] = replacement
                changed = True
        if not found or not changed:
            return False

        try:
            self._write("".join(lines))
        except OSError as err:
            if not self._write_failure_warned:
                self._write_failure_warned = True
                logger.warning(
                    "ai-passport: could not update %s in %s (%s); will keep retrying quietly",
                    self._var,
                    self._path,
                    type(err).__name__,
                )
            return False
        self._write_failure_warned = False
        self._export(access_token)
        logger.debug("ai-passport: refreshed the %s line for the MCP entry", self._var)
        return True

    def _write(self, text: str) -> None:
        """Write in place through a same-directory temp file.

        The .env holds every other credential on the install, so two rules: the
        temp file gets the target's mode BEFORE any content lands in it (a
        umask-mode window would expose every secret world-readable for the
        duration of the write), and a failure never leaves the temp file behind.
        Mode is taken from the existing file when readable, so a 600 .env stays
        600. Same pattern as credentials._write_atomically.
        """
        mode = 0o600
        try:
            mode = self._path.stat().st_mode & 0o777
        except OSError:
            pass
        fd, temp_name = tempfile.mkstemp(prefix=f".{self._path.name}.ai-passport-", dir=str(self._path.parent))
        temp = Path(temp_name)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self._path)
        except BaseException:
            try:
                temp.unlink()
            except OSError:
                pass
            raise

    def _export(self, access_token: str) -> None:
        """Make the new token visible to processes this one spawns.

        Single-profile convenience only: never write a token into the
        process-global environ under a multiplexed gateway, where sibling
        profiles' turns (and any subprocess spawned with env=os.environ) would
        inherit another owner's credential. Same rule the bundled providers
        follow for their API keys.
        """
        try:
            from agent.secret_scope import is_multiplex_active  # type: ignore

            if is_multiplex_active():
                return
        except Exception:
            # No host to ask (a unit test, an older Hermes). Leave the environ
            # alone rather than guess that this process is alone on the machine.
            return
        os.environ[self._var] = access_token
