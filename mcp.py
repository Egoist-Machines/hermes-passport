"""Direct JSON-RPC calls to the Passport MCP endpoint.

The ambient half of this provider reads ``/agent/prefetch``, which is
side-effect-free by construction. The two TOOLS are different: recall may mint
an approval request and remember writes a proposal into the owner's review
inbox, and both of those semantics live on MCP, which is the LLM-facing
surface. Rather than shipping a second copy of them, the provider calls the
same ``/mcp`` endpoint the install's MCP server entry points at, reusing the
bearer token it already refreshes.

Why a hand-rolled client and not an MCP SDK: the endpoint is mounted stateless
(``sessionIdGenerator: undefined`` in lib/app.js), so there is no session to
establish and no handshake to keep alive, and a single POST carrying one
``tools/call`` is the entire protocol surface this provider needs. Adding an SDK
would add a pip dependency to a plugin that otherwise runs on the standard
library, and a sealed venv is exactly where memory must not break.

The response may come back as ``application/json`` or as a one-event SSE stream,
depending on what the transport decides; both shapes are parsed here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from .credentials import Credentials, PassportAuthError
from .transport import open_request

logger = logging.getLogger(__name__)

# The MCP revision the endpoint's SDK speaks. Sent so a future server can tell
# an old plugin apart from a browser poking at the route.
MCP_PROTOCOL_VERSION = "2025-06-18"

# A tool answer is text. This bound is generous for a page of memories and still
# refuses to buffer whatever a middlebox decides to stream at us.
MAX_BODY_BYTES = 4 * 1024 * 1024


class McpError(Exception):
    """A call that could not be completed. Carries owner-readable text."""

    def __init__(self, message: str, *, code: str = "mcp_error"):
        super().__init__(message)
        self.code = code


class McpToolResult:
    """The bounded text answer plus optional MCP structured content."""

    __slots__ = ("text", "structured_content")

    def __init__(self, text: str, structured_content: Optional[dict]):
        self.text = text
        self.structured_content = structured_content


def _parse_sse(text: str) -> list:
    """Collect the JSON payloads out of an SSE body.

    Only ``data:`` lines carry JSON-RPC. Multi-line data fields are joined with
    newlines per the SSE spec before parsing.
    """
    messages = []
    data_lines: list[str] = []

    def flush():
        if not data_lines:
            return
        blob = "\n".join(data_lines)
        data_lines.clear()
        try:
            messages.append(json.loads(blob))
        except Exception:
            logger.debug("ai-passport: unparseable SSE data frame")

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    flush()
    return messages


def _messages_from_body(text: str, content_type: str) -> list:
    if "text/event-stream" in (content_type or "").lower():
        return _parse_sse(text)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        # A JSON content type that is not JSON is a middlebox answering in the
        # backend's place. Try SSE before giving up: a proxy may have relabeled
        # the stream.
        return _parse_sse(text)
    return payload if isinstance(payload, list) else [payload]


def _tool_text(result: Any) -> str:
    """Flatten an MCP tool result's content blocks into text."""
    if not isinstance(result, dict):
        return ""
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return ""
    parts = []
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part).strip()


class McpClient:
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
        self._id_lock = threading.Lock()
        self._next_id = 0

    def _request_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _post(self, base_url: str, access_token: str, payload: dict, timeout_s: float):
        request = urllib.request.Request(
            f"{base_url}/mcp",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "authorization": f"Bearer {access_token}",
                "content-type": "application/json",
                # Both shapes are acceptable; the transport picks.
                "accept": "application/json, text/event-stream",
                "mcp-protocol-version": MCP_PROTOCOL_VERSION,
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=timeout_s) as response:
                status = getattr(response, "status", 200) or 200
                headers = getattr(response, "headers", None)
                content_type = headers.get("content-type", "") if headers else ""
                text = response.read(MAX_BODY_BYTES).decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            status = err.code
            content_type = err.headers.get("content-type", "") if getattr(err, "headers", None) else ""
            try:
                text = err.read(MAX_BODY_BYTES).decode("utf-8", "replace")
            except Exception:
                text = ""
        return status, content_type, text

    def call_tool_result(self, name: str, arguments: dict, *, timeout_s: float) -> McpToolResult:
        """Call one MCP tool and preserve its optional structured content.

        Raises McpError with owner-readable text on every failure, so the tool
        handler upstream can hand the model one honest sentence instead of a
        stack trace.
        """
        try:
            base_url = self._config.base_url or self._credentials.base_url()
            access_token = self._credentials.access_token()
        except PassportAuthError as err:
            raise McpError(err.args[0], code=err.code)
        except Exception as err:
            raise McpError(f"AI Passport credentials are unreadable ({type(err).__name__}).", code="auth")

        request_id = self._request_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }

        try:
            status, content_type, text = self._post(base_url, access_token, payload, timeout_s)
            if status == 401:
                refreshed = self._credentials.access_token(force=True)
                status, content_type, text = self._post(base_url, refreshed, payload, timeout_s)
        except PassportAuthError as err:
            raise McpError(err.args[0], code=err.code)
        except Exception as err:
            raise McpError(f"AI Passport is unreachable ({type(err).__name__}).", code="unavailable")

        if 300 <= status < 400:
            # The transport refuses redirects (transport.py); a 3xx is a
            # middlebox answering in the backend's place.
            raise McpError(
                "AI Passport answered a redirect; refusing to follow it with a credential.",
                code="unavailable",
            )
        if status in (403, 404):
            raise McpError(
                "This app is not allowed to use the AI Passport tools yet. Ask the owner to check the install.",
                code="forbidden",
            )
        if status == 401:
            raise McpError("AI Passport rejected this app's credentials. The owner needs to reinstall.", code="auth")
        if status == 429:
            raise McpError("AI Passport is rate limiting this app. Try again in a minute.", code="rate_limited")
        if status >= 400:
            raise McpError(f"AI Passport answered {status}.", code="http_error")

        for message in _messages_from_body(text, content_type):
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            error = message.get("error")
            if isinstance(error, dict):
                detail = error.get("message")
                raise McpError(
                    str(detail) if detail else "AI Passport refused the call.",
                    code="tool_error",
                )
            result = message.get("result")
            answer = _tool_text(result)
            if isinstance(result, dict) and result.get("isError"):
                # A tool-level error is the trust loop talking (no pass yet, a
                # rejected declaration). Its text is the useful part: it is what
                # tells the model to ask the owner for approval.
                raise McpError(answer or "AI Passport refused the call.", code="tool_refused")
            structured = result.get("structuredContent") if isinstance(result, dict) else None
            return McpToolResult(answer, structured if isinstance(structured, dict) else None)
        raise McpError("AI Passport sent no answer to this call.", code="no_answer")

    def call_tool(self, name: str, arguments: dict, *, timeout_s: float) -> str:
        """Call one MCP tool and return its text answer."""
        return self.call_tool_result(name, arguments, timeout_s=timeout_s).text
