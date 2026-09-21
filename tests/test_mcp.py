"""The direct JSON-RPC client for the MCP tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import harness

mcp_mod = harness.submodule("mcp")
config_mod = harness.submodule("config")
credentials_mod = harness.submodule("credentials")

MCP = "/mcp"
TOKEN = "/oauth/token"


def json_rpc_result(request_id, text="ok", is_error=False, structured_content=None):
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": text}]},
    }
    if is_error:
        payload["result"]["isError"] = True
    if structured_content is not None:
        payload["result"]["structuredContent"] = structured_content
    return harness.FakeResponse(200, json.dumps(payload))


def sse(request_id, text="ok"):
    body = (
        "event: message\n"
        f"data: {json.dumps({'jsonrpc': '2.0', 'id': request_id, 'result': {'content': [{'type': 'text', 'text': text}]}})}\n"
        "\n"
    )
    return harness.FakeResponse(200, body, headers={"content-type": "text/event-stream"})


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.home / "ai-passport-refresh.json"
        harness.write_credentials(self.path, access_token="live", access_token_expires_at=10 ** 12)
        self.http = harness.FakeHttp()
        self.config = config_mod.ProviderConfig({}, str(self.home), env={})
        self.credentials = credentials_mod.Credentials(self.path, opener=self.http, clock=lambda: 0.0)
        self.client = mcp_mod.McpClient(config=self.config, credentials=self.credentials, opener=self.http)

    def call(self, name="recall", arguments=None, timeout_s=5.0):
        return self.client.call_tool(name, arguments or {}, timeout_s=timeout_s)


class Transport(unittest.TestCase):
    def test_sse_frames_are_collected(self):
        body = 'event: message\ndata: {"a": 1}\n\ndata: {"b"\ndata: : 2}\n\n: comment\n'
        self.assertEqual(mcp_mod._parse_sse(body), [{"a": 1}, {"b": 2}])

    def test_unparseable_sse_frames_are_skipped(self):
        self.assertEqual(mcp_mod._parse_sse("data: not json\n\ndata: {}\n\n"), [{}])

    def test_a_json_body_is_wrapped_in_a_list(self):
        self.assertEqual(mcp_mod._messages_from_body('{"id": 1}', "application/json"), [{"id": 1}])

    def test_a_json_array_body_is_kept(self):
        self.assertEqual(mcp_mod._messages_from_body('[{"id": 1}]', "application/json"), [{"id": 1}])

    def test_a_relabeled_sse_stream_is_still_parsed(self):
        self.assertEqual(
            mcp_mod._messages_from_body('data: {"id": 2}\n\n', "application/json"),
            [{"id": 2}],
        )

    def test_tool_text_joins_the_blocks(self):
        result = {"content": [{"type": "text", "text": "one"}, {"type": "image"}, {"type": "text", "text": "two"}]}
        self.assertEqual(mcp_mod._tool_text(result), "one\ntwo")

    def test_tool_text_of_nothing_is_empty(self):
        self.assertEqual(mcp_mod._tool_text(None), "")
        self.assertEqual(mcp_mod._tool_text({}), "")


class Calls(Base):
    def test_a_json_answer_is_returned(self):
        self.http.queue(MCP, lambda request: json_rpc_result(json.loads(request.data)["id"], "remembered"))
        self.assertEqual(self.call(), "remembered")

    def test_structured_content_is_preserved_without_changing_the_text_api(self):
        structured = {
            "ai_passport_domovoy": {"contract_version": 1, "kind": "data"}
        }
        self.http.queue(
            MCP,
            lambda request: json_rpc_result(
                json.loads(request.data)["id"], "remembered", structured_content=structured
            ),
        )
        result = self.client.call_tool_result("recall", {}, timeout_s=5.0)
        self.assertEqual(result.text, "remembered")
        self.assertEqual(result.structured_content, structured)

    def test_an_sse_answer_is_returned(self):
        self.http.queue(MCP, lambda request: sse(json.loads(request.data)["id"], "streamed"))
        self.assertEqual(self.call(), "streamed")

    def test_the_request_is_a_tools_call(self):
        self.http.queue(MCP, lambda request: json_rpc_result(json.loads(request.data)["id"]))
        self.call(name="recall", arguments={"categories": ["fact"]})
        body = self.http.bodies(MCP)[0]
        self.assertEqual(body["method"], "tools/call")
        self.assertEqual(body["params"], {"name": "recall", "arguments": {"categories": ["fact"]}})
        self.assertEqual(body["jsonrpc"], "2.0")
        headers = self.http.calls[0]["headers"]
        self.assertEqual(headers["authorization"], "Bearer live")
        self.assertIn("text/event-stream", headers["accept"])
        self.assertEqual(headers["mcp-protocol-version"], mcp_mod.MCP_PROTOCOL_VERSION)

    def test_request_ids_do_not_repeat(self):
        self.http.queue(
            MCP,
            lambda request: json_rpc_result(json.loads(request.data)["id"]),
            lambda request: json_rpc_result(json.loads(request.data)["id"]),
        )
        self.call()
        self.call()
        ids = [body["id"] for body in self.http.bodies(MCP)]
        self.assertEqual(len(set(ids)), 2)

    def test_an_answer_for_another_id_is_not_accepted(self):
        self.http.queue(MCP, json_rpc_result(9999))
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "no_answer")

    def test_a_tool_level_refusal_carries_the_trust_loops_text(self):
        # The refusal text is the useful part: it is what tells the model to hand
        # the user an approval link.
        self.http.queue(
            MCP,
            lambda request: json_rpc_result(
                json.loads(request.data)["id"], "Ask the owner to approve: https://my.ego.ist/passes", is_error=True
            ),
        )
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "tool_refused")
        self.assertIn("my.ego.ist/passes", str(caught.exception))

    def test_a_protocol_error_is_reported(self):
        self.http.queue(
            MCP,
            lambda request: harness.FakeResponse(
                200,
                json.dumps({"jsonrpc": "2.0", "id": json.loads(request.data)["id"],
                            "error": {"code": -32602, "message": "bad params"}}),
            ),
        )
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "tool_error")
        self.assertIn("bad params", str(caught.exception))

    def test_a_401_forces_one_refresh_and_one_retry(self):
        self.http.queue(
            MCP,
            harness.FakeResponse(401, "{}"),
            lambda request: json_rpc_result(json.loads(request.data)["id"], "after refresh"),
        )
        self.http.queue(TOKEN, harness.FakeResponse(200, json.dumps({"access_token": "fresh", "expires_in": 3600})))
        self.assertEqual(self.call(), "after refresh")
        self.assertEqual(self.http.calls[-1]["headers"]["authorization"], "Bearer fresh")

    def test_a_403_reads_as_an_install_problem(self):
        self.http.queue(MCP, harness.FakeResponse(403, "{}"))
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "forbidden")

    def test_a_redirect_is_refused(self):
        # Following it would hand the bearer to the redirect target.
        self.http.queue(MCP, harness.FakeResponse(302, "", headers={"location": "https://evil.example/"}))
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "unavailable")
        self.assertIn("redirect", str(caught.exception))

    def test_a_429_says_to_try_again(self):
        self.http.queue(MCP, harness.FakeResponse(429, "{}"))
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "rate_limited")

    def test_a_transport_failure_is_reported_not_raised_raw(self):
        self.http.queue(MCP, OSError("connection reset"))
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "unavailable")

    def test_a_missing_install_is_reported_as_such(self):
        self.path.unlink()
        self.credentials._reset_for_tests()
        with self.assertRaises(mcp_mod.McpError) as caught:
            self.call()
        self.assertEqual(caught.exception.code, "not_installed")
        self.assertEqual(self.http.calls, [])

    def test_the_configured_base_url_wins_over_the_credentials_origin(self):
        config = config_mod.ProviderConfig({"base_url": "http://localhost:3020"}, str(self.home), env={})
        client = mcp_mod.McpClient(config=config, credentials=self.credentials, opener=self.http)
        self.http.queue(MCP, lambda request: json_rpc_result(json.loads(request.data)["id"]))
        client.call_tool("recall", {}, timeout_s=1.0)
        self.assertEqual(self.http.calls[0]["url"], "http://localhost:3020/mcp")


if __name__ == "__main__":
    unittest.main()
