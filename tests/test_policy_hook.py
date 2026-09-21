"""The shell pre_tool_call hook: audit reports that never block and never refresh."""

from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import harness

policy_hook = harness.submodule("policy_hook")


def check_ok(**overrides) -> str:
    payload = {
        "decision": "allow",
        "effective": "allow",
        "mode": "audit",
        "matched_pattern": None,
        "event_id": "6b9c8f9e-0000-4000-8000-000000000001",
        "approval_url": None,
        "poll_url": None,
        "cache_ttl": 60,
    }
    payload.update(overrides)
    return json.dumps(payload)


def check_fixture(name: str, **overrides) -> str:
    """A vendored wire body (harness.fixture_json), optionally tweaked for a
    scenario. Building the degraded bodies FROM the fixtures keeps these
    behavioral tests from quietly drifting off the shapes the backend
    actually sends."""
    payload = harness.fixture_json(name)
    payload.update(overrides)
    return json.dumps(payload)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.http = harness.FakeHttp()
        self.clock = time.time()

    def now(self):
        return self.clock

    def write_credentials(self, **overrides):
        payload = {
            "access_token": "access-1",
            "access_token_expires_at": self.clock + 3600,
        }
        payload.update(overrides)
        return harness.write_credentials(self.home / "ai-passport-refresh.json", **payload)

    def write_config(self, values):
        (self.home / "ai-passport.json").write_text(json.dumps(values), encoding="utf-8")

    def run_check(self, tool="terminal", tool_input=None, session_id="sess-1"):
        return policy_hook.run_check(
            str(self.home), tool, tool_input if tool_input is not None else {"command": "ls"}, session_id,
            now=self.now, opener=self.http,
        )

    def queue_snapshot(self, mode="audit", rules=(), version=1):
        self.http.queue(
            "/agent/policy/snapshot",
            harness.FakeResponse(
                200,
                json.dumps({"version": version, "mode": mode, "default_action": "allow", "rules": list(rules), "cache_ttl": 60}),
            ),
        )

    def seed_snapshot(self, mode="enforce", rules=(), fetched_at=None):
        """Write a snapshot straight into the state file, as an earlier hook
        process would have left it."""
        state_path = self.home / "ai-passport-policy-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state["snapshot"] = {"mode": mode, "rules": list(rules), "fetched_at": fetched_at if fetched_at is not None else self.clock}
        state_path.write_text(json.dumps(state), encoding="utf-8")


class ArgsDigest(unittest.TestCase):
    def test_stable_across_key_order_and_shaped_for_the_wire(self):
        a = policy_hook.args_digest({"command": "ls", "cwd": "/tmp"})
        b = policy_hook.args_digest({"cwd": "/tmp", "command": "ls"})
        self.assertEqual(a, b)
        self.assertRegex(a, r"^h1_[0-9a-f]{64}$")

    def test_non_dict_and_empty_input_have_no_digest(self):
        self.assertIsNone(policy_hook.args_digest(None))
        self.assertIsNone(policy_hook.args_digest("rm -rf /"))
        self.assertIsNone(policy_hook.args_digest({}))

    def test_uncanonicalizable_input_degrades_to_none(self):
        # Python would happily emit bare NaN, which is not JSON; the digest
        # must refuse rather than diverge from the OpenClaw twin.
        self.assertIsNone(policy_hook.args_digest({"x": float("nan")}))
        circular = {}
        circular["self"] = circular
        self.assertIsNone(policy_hook.args_digest(circular))


class RunCheck(Base):
    def test_posts_name_digest_and_session_and_returns_the_answer(self):
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        payload = self.run_check(tool="terminal", tool_input={"command": "rm -rf /home"}, session_id="sess-9")
        self.assertEqual(payload["decision"], "allow")
        body = self.http.bodies("/agent/policy/check")[0]
        self.assertEqual(body["tool"], "terminal")
        self.assertEqual(body["session_key"], "sess-9")
        self.assertRegex(body["args_digest"], r"^h1_[0-9a-f]{64}$")
        self.assertNotIn("rm -rf", json.dumps(body), "arguments never leave the machine")
        # An audit install makes exactly ONE request per novel call: the mode
        # arrives free on the check answer, so no snapshot is ever fetched.
        self.assertEqual([call["url"].rsplit("/", 1)[-1] for call in self.http.calls], ["check"])
        self.assertEqual(self.http.calls[0]["headers"]["authorization"], "Bearer access-1")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"], {"mode": "audit", "rules": [], "fetched_at": 0},
                         "the learned mode persists as a stub for the next process")

    def test_identical_calls_inside_the_ttl_replay_the_answer_across_processes(self):
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.assertIsNotNone(self.run_check())
        # A second hook process: fresh run_check, same state file. The ANSWER
        # comes back (not None): a deduped repeat must still be actionable.
        replay = self.run_check()
        self.assertEqual(replay["effective"], "allow")
        self.assertNotIn("event_id", replay, "a replay must not claim a recording that did not happen")
        self.assertEqual(self.http.count("/agent/policy/check"), 1)
        # A different digest is a different audit event.
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.assertIsNotNone(self.run_check(tool_input={"command": "pwd"}))
        # The TTL bounds the dedupe.
        self.clock += 61
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.assertIsNotNone(self.run_check())

    def test_a_cached_deny_keeps_denying_for_its_whole_ttl(self):
        # Phase-5 forward compat: an enforce-mode deny with cache_ttl=60 must
        # block identical repeats, not silently allow them from the cache.
        self.write_credentials()
        self.http.queue(
            "/agent/policy/check",
            harness.FakeResponse(200, check_ok(decision="deny", effective="deny", mode="enforce", matched_pattern="terminal")),
        )
        first = self.run_check()
        self.assertEqual(policy_hook.hook_response(first)["action"], "block")
        replay = self.run_check()
        self.assertEqual(self.http.count("/agent/policy/check"), 1)
        self.assertEqual(replay["effective"], "deny")
        self.assertEqual(policy_hook.hook_response(replay)["action"], "block")

    def test_an_older_expiry_only_cache_entry_is_a_miss_not_an_answer(self):
        self.write_credentials()
        state_path = self.home / "ai-passport-policy-state.json"
        digest = policy_hook.args_digest({"command": "ls"})
        state_path.write_text(json.dumps({"cache": {f"terminal {digest}": self.clock + 60}}), encoding="utf-8")
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.assertIsNotNone(self.run_check())
        self.assertEqual(self.http.count("/agent/policy/check"), 1, "the legacy shape re-posts rather than inventing an answer")

    def test_honors_the_credentials_path_override(self):
        override = self.home / "elsewhere" / "creds.json"
        harness.write_credentials(override, access_token="access-9", access_token_expires_at=self.clock + 3600)
        self.write_config({"credentials_path": str(override)})
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.assertIsNotNone(self.run_check())
        self.assertEqual(self.http.calls[0]["headers"]["authorization"], "Bearer access-9")

    def test_never_refreshes_and_skips_without_a_fresh_token(self):
        # Expired access token: the provider owns rotation; the hook must not
        # spend the single-use refresh token from a second process.
        self.write_credentials(access_token_expires_at=self.clock - 10)
        self.assertIsNone(self.run_check())
        self.assertEqual(self.http.calls, [], "no check POST and, critically, no token POST")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["backoff_reason"], "auth")

    def test_missing_credentials_file_skips_silently(self):
        self.assertIsNone(self.run_check())
        self.assertEqual(self.http.calls, [])

    def test_429_and_403_latch_a_persisted_backoff(self):
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(429, json.dumps({"error": "rate_limited"})))
        self.assertIsNone(self.run_check())
        self.assertIsNone(self.run_check(tool_input={"command": "other"}))
        self.assertEqual(self.http.count("/agent/policy/check"), 1, "the backoff covers every tool")
        self.clock += 61
        self.http.queue("/agent/policy/check", harness.FakeResponse(403, json.dumps({"error": "forbidden"})))
        self.assertIsNone(self.run_check(tool_input={"command": "third"}))
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["backoff_reason"], "forbidden")
        self.assertGreater(state["backoff_until"], self.clock + 200, "forbidden latches minutes, not seconds")

    def test_transport_failure_backs_off_and_stays_silent(self):
        self.write_credentials()
        self.http.queue("/agent/policy/check", urllib.error.URLError("connection refused"))
        self.assertIsNone(self.run_check())
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["backoff_reason"], "unavailable")

    def test_malformed_200_is_a_middlebox_not_an_answer(self):
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, "<html>captive portal</html>"))
        self.assertIsNone(self.run_check())
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["backoff_reason"], "unavailable")
        self.assertEqual(state["cache"], {}, "a middlebox answer must not be cached as a decision")

    def test_tool_names_outside_the_grammar_are_skipped_without_backoff(self):
        self.write_credentials()
        for name in (":exec", ".hidden", "tool with spaces", "", None, 42):
            self.assertIsNone(self.run_check(tool=name))
        self.assertEqual(self.http.calls, [])
        self.assertFalse((self.home / "ai-passport-policy-state.json").exists() and
                         json.loads((self.home / "ai-passport-policy-state.json").read_text()).get("backoff_reason"))

    def test_policy_disabled_reports_nothing(self):
        self.write_credentials()
        self.write_config({"policy": {"enabled": False}})
        self.assertIsNone(self.run_check())
        self.assertEqual(self.http.calls, [])

    def test_base_url_override_wins_over_the_credentials_origin(self):
        self.write_credentials()
        self.write_config({"base_url": "http://localhost:3020"})
        self.queue_snapshot()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.run_check()
        for call in self.http.calls:
            self.assertTrue(call["url"].startswith("http://localhost:3020/agent/policy/"), call["url"])

    def test_session_key_is_clamped_to_the_wire_bound(self):
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.run_check(session_id="s" * 300)
        body = self.http.bodies("/agent/policy/check")[0]
        self.assertEqual(len(body["session_key"]), 128)

    def test_budget_is_spent_across_processes_through_the_state_file(self):
        self.write_credentials()
        # Audit installs never fetch snapshots, so the whole window belongs to
        # checks; the call past the budget is skipped, not queued.
        for i in range(policy_hook.BUDGET_MAX):
            self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
            self.run_check(tool_input={"command": f"c{i}"})
        self.run_check(tool_input={"command": "over-budget"})
        self.assertEqual(self.http.count("/agent/policy/check"), policy_hook.BUDGET_MAX)
        self.assertEqual(self.http.count("/agent/policy/snapshot"), 0)


class Enforce(Base):
    RULES = (
        {"tool_pattern": "shell*", "action": "deny"},
        {"tool_pattern": "payments*", "action": "require_approval"},
    )

    def test_local_matcher_mirrors_the_backend_total_order(self):
        rules = [
            {"tool_pattern": "*", "action": "deny"},
            {"tool_pattern": "shell*", "action": "require_approval"},
            {"tool_pattern": "shell.run", "action": "allow"},
            {"tool_pattern": "not a pattern", "action": "deny"},
        ]
        self.assertEqual(policy_hook.resolve_local_policy("shell.run", rules), {"action": "allow", "matched_pattern": "shell.run"})
        self.assertEqual(
            policy_hook.resolve_local_policy("shell.exec", rules), {"action": "require_approval", "matched_pattern": "shell*"}
        )
        self.assertEqual(policy_hook.resolve_local_policy("web_search", rules), {"action": "deny", "matched_pattern": "*"})
        self.assertEqual(policy_hook.resolve_local_policy("anything", []), {"action": "allow", "matched_pattern": None})

    def test_outage_in_enforce_mode_fails_closed_only_for_the_owners_rules(self):
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=self.RULES)
        self.http.queue("/agent/policy/check", urllib.error.URLError("connection refused"))
        denied = self.run_check(tool="shell.run")
        self.assertEqual(denied["effective"], "deny")
        self.assertIn("unreachable", denied["message"])
        self.assertEqual(policy_hook.hook_response(denied)["action"], "block")
        # Backed off now: the next calls evaluate locally without the network.
        approval = self.run_check(tool="payments.charge", tool_input={"amount": 5})
        self.assertEqual(approval["effective"], "deny")
        self.assertIn("no approval can be requested", approval["message"])
        self.assertIsNone(self.run_check(tool="web_search", tool_input={"q": "x"}), "unrestricted tools stay open")

    def test_outage_in_audit_mode_stays_silent(self):
        self.write_credentials()
        self.seed_snapshot(mode="audit", rules=self.RULES)
        self.http.queue("/agent/policy/check", urllib.error.URLError("connection refused"))
        self.assertIsNone(self.run_check(tool="shell.run"))

    def test_no_token_in_enforce_mode_still_vetoes_by_the_last_known_rules(self):
        # Enforce mode with a dead install: the rules the owner wrote keep
        # holding even though no report can land.
        self.write_credentials(access_token_expires_at=self.clock - 10)
        self.seed_snapshot(mode="enforce", rules=self.RULES)
        denied = self.run_check(tool="shell.run")
        self.assertEqual(denied["effective"], "deny")
        self.assertEqual(self.http.calls, [])

    def test_a_newer_snapshot_protocol_is_refused_whole(self):
        self.write_credentials()
        self.queue_snapshot(mode="enforce", rules=[{"tool_pattern": "*", "action": "deny"}], version=2)
        self.http.queue("/agent/policy/check", urllib.error.URLError("connection refused"))
        self.assertIsNone(self.run_check(tool="shell.run"), "half-understood rules must not be acted on")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertIsNone(state["snapshot"])

    def test_an_approval_answer_is_never_cached_and_blocks_with_the_link(self):
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=self.RULES)
        pending = check_ok(
            decision="require_approval",
            effective="require_approval",
            mode="enforce",
            matched_pattern="payments*",
            approval_url="https://my.ego.ist/inbox",
        )
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, pending), harness.FakeResponse(200, pending))
        first = self.run_check(tool="payments.charge", tool_input={"amount": 5})
        block = policy_hook.hook_response(first)
        self.assertEqual(block["action"], "block")
        self.assertIn("https://my.ego.ist/inbox", block["message"])
        self.assertIn("Retry after they approve", block["message"])
        # The identical retry posts a fresh check (landing on the resolved
        # state or minting a fresh wait) instead of replaying the old block.
        second = self.run_check(tool="payments.charge", tool_input={"amount": 5})
        self.assertEqual(self.http.count("/agent/policy/check"), 2)
        self.assertEqual(second["effective"], "require_approval")

    def test_a_check_answer_keeps_the_snapshot_mode_fresh(self):
        self.write_credentials()
        self.seed_snapshot(mode="audit", rules=[])
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok(mode="enforce")))
        self.run_check()
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"]["mode"], "enforce", "a mode flip reaches the next process before the snapshot poll")

    def test_a_degraded_answer_defers_to_the_last_known_rules(self):
        # The plane's rules-unreachable shape (the vendored degraded fixture)
        # is a shrug, not a verdict: believing its mode would drop an enforce
        # install to the audit posture and fail every deny rule open for the
        # length of a backend store outage.
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_fixture("agent-policy-check-degraded.json")))
        payload = self.run_check(tool="shell.run")
        self.assertEqual(payload["effective"], "deny")
        self.assertIn("rule shell*", payload["message"])
        # Honest copy: the Passport ANSWERED (a degraded 200), so the block
        # reason must not send the owner debugging connectivity.
        self.assertIn("policy service is temporarily degraded", payload["message"])
        self.assertNotIn("unreachable", payload["message"])
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"]["mode"], "enforce", "the degraded audit mode never overwrites enforce")
        self.assertEqual(state["snapshot"]["rules"], [{"tool_pattern": "shell*", "action": "deny"}])

    def test_an_unknown_degraded_reason_is_not_mistaken_for_a_verdict(self):
        # Closed-world only on the VERDICT side: plugins live on owner
        # machines for years while the backend deploys continuously, so a
        # future no-opinion flavor (hardcoded audit mode, some new reason
        # string) must fall back to the mode heuristic instead of teaching
        # its audit mode over an enforce install.
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        self.http.queue(
            "/agent/policy/check",
            harness.FakeResponse(200, check_fixture("agent-policy-check-degraded.json", degraded_reason="snapshot_unavailable")),
        )
        payload = self.run_check(tool="shell.run")
        self.assertEqual(payload["effective"], "deny", "the last-known rules decide, not the unknown shrug")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"]["mode"], "enforce", "the unknown flavor's audit mode taught nothing")

    def test_a_degraded_answer_leaves_unconstrained_tools_alone(self):
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_fixture("agent-policy-check-degraded.json")))
        payload = self.run_check(tool="web_search", tool_input={"q": "x"})
        self.assertEqual(payload["effective"], "allow")
        self.assertIsNone(policy_hook.hook_response(payload))

    def test_a_replayed_no_opinion_body_reevaluates_the_local_rules(self):
        # The cache holds the WIRE answer, never a frozen local verdict: a
        # rule change landing between identical repeats must reach the replay.
        # (Symmetrically, a locally synthesized "retry later" deny must not
        # outlive the recovery of the plane that synthesized it.)
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_fixture("agent-policy-check-degraded.json")))
        first = self.run_check(tool="shell.run")
        self.assertEqual(first["effective"], "allow", "no local rules yet, so the wire allow stands")
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        replay = self.run_check(tool="shell.run")
        self.assertEqual(replay["effective"], "deny", "the replay ran the fresh rules, not the frozen first verdict")
        self.assertIn("rule shell*", replay["message"])
        self.assertEqual(self.http.count("/agent/policy/check"), 1, "the repeat still rode the cache")

    def test_a_no_opinion_degraded_answer_is_cached_but_teaches_no_mode(self):
        # The resolved verdict is cached (the cache is what keeps per-call
        # checks off an already-degraded plane), but the hardcoded audit mode
        # of the no-opinion flavor never becomes a learned mode.
        self.write_credentials()
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_fixture("agent-policy-check-degraded.json")))
        self.assertIsNotNone(self.run_check(tool="web_search", tool_input={"q": "x"}))
        self.assertIsNotNone(self.run_check(tool="web_search", tool_input={"q": "x"}))
        self.assertEqual(self.http.count("/agent/policy/check"), 1, "the identical repeat rides the cached verdict")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertIsNone(state["snapshot"], "no mode stub is learned from a no-opinion body")

    def test_a_recording_degraded_enforce_deny_is_honored_and_arms_the_snapshot(self):
        # An event_write_failed body is a real verdict whose RECORDING failed:
        # the deny stands even with no local rules, and the learned stub is
        # what makes the next enforce call fetch the rules (the only flip
        # detector while the event table is down).
        self.write_credentials()
        self.http.queue(
            "/agent/policy/check",
            harness.FakeResponse(200, check_fixture("agent-policy-check-degraded-deny.json")),
        )
        payload = self.run_check(tool="shell.run")
        self.assertEqual(payload["effective"], "deny")
        self.assertEqual(policy_hook.hook_response(payload)["action"], "block")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"], {"mode": "enforce", "rules": [], "fetched_at": 0})
        # The replay-classification pin: the cached body must classify exactly
        # as its first arrival did, so the replayed verdict stays a deny
        # instead of being re-resolved as a shrug against the local rules.
        replay = self.run_check(tool="shell.run")
        self.assertEqual(replay["effective"], "deny", "the replay is the verdict it was, not a re-resolved shrug")
        self.assertEqual(self.http.count("/agent/policy/check"), 1, "and it rode the cache")

    def test_a_recording_degraded_audit_answer_teaches_the_flip_out_of_enforce(self):
        # The audit flavor of the same failure: a real answer whose mode is
        # truth, distinguishable from the no-opinion body only by
        # degraded_reason. Before the field existed, an owner's
        # enforce-to-audit flip was invisible for the length of an event-write
        # outage and their un-enforced rules kept blocking tools.
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        self.http.queue(
            "/agent/policy/check",
            harness.FakeResponse(200, check_fixture("agent-policy-check-degraded.json", degraded_reason="event_write_failed")),
        )
        payload = self.run_check(tool="web_search", tool_input={"q": "x"})
        self.assertEqual(payload["effective"], "allow")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"]["mode"], "audit", "the owner's flip out of enforce landed")

    def test_a_degraded_grant_consumed_allow_is_the_owners_yes(self):
        # The vendored degraded-grant fixture (cache_ttl 0): the single-use
        # grant was burned server-side BEFORE the event insert failed. Local
        # rules naming the tool require_approval must not override the owner's
        # explicit approval into a second page, and single-use is never cached.
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "payment*", "action": "require_approval"}])
        self.http.queue(
            "/agent/policy/check",
            harness.FakeResponse(200, check_fixture("agent-policy-check-degraded-grant.json")),
            harness.FakeResponse(200, check_fixture("agent-policy-check-degraded-grant.json")),
        )
        payload = self.run_check(tool="payment.charge", tool_input={"amount": 5})
        self.assertEqual(payload["effective"], "allow")
        self.assertIsNone(policy_hook.hook_response(payload))
        self.assertIsNotNone(self.run_check(tool="payment.charge", tool_input={"amount": 5}))
        self.assertEqual(self.http.count("/agent/policy/check"), 2, "a burned grant is never replayed from cache")

    def test_an_audit_degraded_answer_for_an_approval_ruled_tool_still_caches(self):
        # The single-use belt is for the enforce fall-open ONLY: in audit mode
        # the same outage enforced nothing, and stripping the ttl there would
        # send one uncached check per repeated call against the
        # already-degraded plane.
        self.write_credentials()
        body = check_fixture(
            "agent-policy-check-degraded.json",
            degraded_reason="event_write_failed",
            decision="require_approval",
            matched_pattern="payment*",
        )
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, body))
        self.assertEqual(self.run_check(tool="payment.charge", tool_input={"amount": 5})["effective"], "allow")
        self.assertEqual(self.run_check(tool="payment.charge", tool_input={"amount": 5})["effective"], "allow")
        self.assertEqual(self.http.count("/agent/policy/check"), 1, "the identical repeat rode the cache")


class BudgetAndBackoffDiscipline(Base):
    def test_a_dead_install_stats_the_credentials_file_once_per_backoff_window(self):
        self.write_credentials(access_token_expires_at=self.clock - 10)
        calls = []
        real = policy_hook.load_token

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with mock.patch.object(policy_hook, "load_token", side_effect=counting):
            self.run_check()
            self.run_check(tool_input={"command": "second"})
            self.run_check(tool_input={"command": "third"})
        self.assertEqual(len(calls), 1, "the backoff gate must come before any file I/O")

    def test_the_check_rechecks_the_budget_after_the_snapshot_spend(self):
        # At one slot remaining, the enforce snapshot fetch takes it and the
        # check must be SKIPPED, not allowed to go one over the window.
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[], fetched_at=0)
        state_path = self.home / "ai-passport-policy-state.json"
        state = json.loads(state_path.read_text())
        state["budget"] = [self.clock] * (policy_hook.BUDGET_MAX - 1)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        self.queue_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        self.run_check(tool="web_search", tool_input={"q": "x"})
        self.assertEqual(self.http.count("/agent/policy/snapshot"), 1)
        self.assertEqual(self.http.count("/agent/policy/check"), 0, "the snapshot spent the window's last slot")

    def test_an_audit_snapshot_is_never_refreshed(self):
        self.write_credentials()
        self.seed_snapshot(mode="audit", rules=[], fetched_at=self.clock - 3600)
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok()))
        self.run_check()
        self.assertEqual(self.http.count("/agent/policy/snapshot"), 0, "audit installs never pay for rules they never read")

    def test_an_enforce_stub_fetches_its_rules_before_the_check(self):
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[], fetched_at=0)
        self.queue_snapshot(mode="enforce", rules=[{"tool_pattern": "shell*", "action": "deny"}])
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, check_ok(mode="enforce")))
        self.run_check(tool="web_search", tool_input={"q": "x"})
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"]["rules"], [{"tool_pattern": "shell*", "action": "deny"}])
        self.assertGreater(state["snapshot"]["fetched_at"], 0)

    def test_an_enforce_stub_without_rules_fails_open_not_closed_on_empty_rules(self):
        # Mode learned from a check answer, rules never fetched, plane dark:
        # evaluating the EMPTY list would silently allow constrained tools
        # while pretending the rules decided, so the stub stays silent.
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[], fetched_at=0)
        self.http.queue("/agent/policy/snapshot", urllib.error.URLError("refused"))
        self.http.queue("/agent/policy/check", urllib.error.URLError("refused"))
        self.assertIsNone(self.run_check(tool="shell.run"))

    def test_a_single_use_answer_is_never_cached(self):
        # cache_ttl 0 marks a grant-consumed allow: the next identical call
        # must post its own check rather than riding the burned grant.
        self.write_credentials()
        self.http.queue(
            "/agent/policy/check",
            harness.FakeResponse(200, check_ok(cache_ttl=0)),
            harness.FakeResponse(200, check_ok(cache_ttl=0)),
        )
        self.assertIsNotNone(self.run_check(tool="payments.charge", tool_input={"amount": 5}))
        self.assertIsNotNone(self.run_check(tool="payments.charge", tool_input={"amount": 5}))
        self.assertEqual(self.http.count("/agent/policy/check"), 2)


class HookResponse(unittest.TestCase):
    def test_allow_and_none_stay_silent(self):
        self.assertIsNone(policy_hook.hook_response(None))
        self.assertIsNone(policy_hook.hook_response({"effective": "allow", "decision": "deny"}))

    def test_enforced_answers_translate_to_the_block_wire_shape(self):
        deny = policy_hook.hook_response({"effective": "deny"})
        self.assertEqual(deny["action"], "block")
        approval = policy_hook.hook_response({"effective": "require_approval", "approval_url": "https://my.ego.ist/inbox"})
        self.assertEqual(approval["action"], "block")
        self.assertIn("https://my.ego.ist/inbox", approval["message"])


class Selftest(Base):
    def test_back_to_back_selftests_both_probe_for_real(self):
        # The first draft used constant probe args, so a re-run within the
        # dedupe TTL exited 1 on a healthy install. Unique args per run keep
        # every selftest a real probe.
        self.write_credentials()
        probes = []

        def fake_run_check(home, tool, tool_input, session_id, **kwargs):
            probes.append(tool_input)
            return {"mode": "audit", "decision": "allow", "event_id": "e-1"}

        stdout = io.StringIO()
        with mock.patch.object(policy_hook, "run_check", fake_run_check):
            with mock.patch.object(policy_hook.sys, "stdout", stdout):
                first = policy_hook.selftest(str(self.home))
                second = policy_hook.selftest(str(self.home))
        self.assertEqual((first, second), (0, 0))
        self.assertNotEqual(probes[0], probes[1], "each selftest sends a novel digest, so dedupe cannot eat it")
        self.assertEqual(stdout.getvalue().count("event recorded"), 2)

    def test_selftest_names_the_credentials_path_it_checked(self):
        self.write_config({"credentials_path": str(self.home / "nowhere.json")})
        stdout = io.StringIO()
        with mock.patch.object(policy_hook.sys, "stdout", stdout):
            code = policy_hook.selftest(str(self.home))
        self.assertEqual(code, 1)
        self.assertIn("nowhere.json", stdout.getvalue())


class Main(Base):
    def run_main(self, stdin_text, argv=None):
        stdout = io.StringIO()
        with mock.patch.object(policy_hook.sys, "stdin", io.StringIO(stdin_text)):
            with mock.patch.object(policy_hook.sys, "stdout", stdout):
                code = policy_hook.main(argv or ["--home", str(self.home)])
        return code, stdout.getvalue()

    def test_exits_zero_and_silent_on_the_audit_path(self):
        # An unreachable loopback port: the POST fails, and failing open means
        # exit 0 with no stdout, which is Hermes' silent no-op.
        self.write_credentials(token_url="http://127.0.0.1:9/token")
        code, out = self.run_main(json.dumps({
            "hook_event_name": "pre_tool_call",
            "tool_name": "terminal",
            "tool_input": {"command": "ls"},
            "session_id": "sess-1",
        }))
        self.assertEqual((code, out), (0, ""))

    def test_exits_zero_on_garbage_stdin(self):
        self.assertEqual(self.run_main("not json at all")[0], 0)
        self.assertEqual(self.run_main("")[0], 0)
        self.assertEqual(self.run_main(json.dumps(["a", "list"]))[0], 0)


if __name__ == "__main__":
    unittest.main()
