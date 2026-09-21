"""The vendored /agent/* contract fixtures (issue #425 phase 6).

``tests/fixtures/`` holds byte-identical copies of ``shared/agent-contract/``
from the backend repo (``scripts/sync_shared.mjs`` syncs them, and the
backend's ``agent_backend_smoke.mjs`` pins its live routes to the same files).
These tests are this plugin's half of the handshake: every fixture body must
drive the real parsers to the documented outcome, so a contract change that
would break this plugin fails a test on whichever side drifted first.

The scaffolding is the policy-hook and client suites' own ``Base`` classes,
subclassed rather than restated, so a change to the hook invocation surface or
the client constructor cannot leave this suite exercising a stale copy.
"""

from __future__ import annotations

import json
import unittest

import harness
import test_client
import test_policy_hook

policy_hook = harness.submodule("policy_hook")

# One loader for both Hermes suites: the path and any future validation live
# in harness (test_policy_hook.check_fixture reads through the same one).
fixture_text = harness.fixture_text
fixture = harness.fixture_json


class CheckFixtures(test_policy_hook.Base):
    """Each check fixture, fed through run_check and hook_response verbatim."""

    def queue_check_fixture(self, name: str) -> None:
        self.http.queue("/agent/policy/check", harness.FakeResponse(200, fixture_text(name)))

    def test_the_allow_fixture_stays_silent(self):
        self.write_credentials()
        self.queue_check_fixture("agent-policy-check-allow.json")
        payload = self.run_check(tool="web_search", tool_input={"q": "hi"})
        self.assertEqual(payload["effective"], "allow")
        self.assertIsNone(policy_hook.hook_response(payload))

    def test_the_deny_fixture_blocks(self):
        self.write_credentials()
        self.queue_check_fixture("agent-policy-check-deny.json")
        payload = self.run_check(tool="shell.run")
        self.assertEqual(payload["effective"], "deny")
        block = policy_hook.hook_response(payload)
        self.assertEqual(block["action"], "block")

    def test_the_approval_fixture_blocks_with_the_owners_link(self):
        body = fixture("agent-policy-check-approval.json")
        self.write_credentials()
        self.queue_check_fixture("agent-policy-check-approval.json")
        payload = self.run_check(tool="payment.charge", tool_input={"amount": 5})
        self.assertEqual(payload["effective"], "require_approval")
        block = policy_hook.hook_response(payload)
        self.assertEqual(block["action"], "block")
        self.assertIn(body["approval_url"], block["message"])
        self.assertIn("Retry after they approve", block["message"])

    def test_the_degraded_fixture_defers_to_the_last_known_rules(self):
        # The degraded body is the plane's rules-unreachable shrug. With no
        # local rules it passes through as an allow; with an enforce snapshot
        # the last-known rules answer instead, so a store outage blocks
        # exactly the tools the owner constrained.
        self.write_credentials()
        self.queue_check_fixture("agent-policy-check-degraded.json")
        payload = self.run_check(tool="shell.run")
        self.assertEqual(payload["effective"], "allow")
        self.assertIsNone(policy_hook.hook_response(payload))
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertIsNone(state["snapshot"], "a degraded body never teaches a mode")

        snapshot = fixture("agent-policy-snapshot.json")
        self.seed_snapshot(mode=snapshot["mode"], rules=snapshot["rules"])
        self.queue_check_fixture("agent-policy-check-degraded.json")
        # A fresh tool_input: the first call's resolved verdict is cached by
        # design, and this leg is about the snapshot, not the cache.
        blocked = self.run_check(tool="shell.run", tool_input={"command": "rm"})
        self.assertEqual(blocked["effective"], "deny")
        self.assertIn("rule shell*", blocked["message"])

    def test_the_degraded_deny_fixture_stays_a_deny(self):
        # mode enforce + degraded is a real verdict whose recording failed:
        # honored from the wire even for a tool no local rule constrains.
        self.write_credentials()
        self.queue_check_fixture("agent-policy-check-degraded-deny.json")
        payload = self.run_check(tool="browser.open", tool_input={"url": "https://x.test"})
        self.assertEqual(payload["effective"], "deny")
        self.assertEqual(policy_hook.hook_response(payload)["action"], "block")

    def test_the_degraded_grant_fixture_is_the_owners_yes(self):
        # A grant-consumed allow (cache_ttl 0) with the degraded flag: the
        # owner already approved this exact call, so a local require_approval
        # rule must not page them a second time.
        body = fixture("agent-policy-check-degraded-grant.json")
        self.assertEqual(body["cache_ttl"], 0, "the grant fixture must be marked single-use")
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "payment*", "action": "require_approval"}])
        self.queue_check_fixture("agent-policy-check-degraded-grant.json")
        payload = self.run_check(tool="payment.charge", tool_input={"amount": 5})
        self.assertEqual(payload["effective"], "allow")
        self.assertIsNone(policy_hook.hook_response(payload))

    def test_the_degraded_approval_fall_open_admits_one_call_and_is_never_cached(self):
        # An enforced require_approval whose event insert failed is an
        # approval nobody could ever resolve, so the backend lets exactly ONE
        # call through (decision require_approval, effective allow, cache_ttl
        # 0). The repeat must mint its own check, not ride the first fall-open
        # from the cache into a minute of unapproved calls.
        body = fixture("agent-policy-check-degraded-approval.json")
        self.assertEqual(body["cache_ttl"], 0, "the fall-open fixture must be marked single-use")
        self.assertEqual(body["degraded_reason"], "event_write_failed")
        self.write_credentials()
        self.seed_snapshot(mode="enforce", rules=[{"tool_pattern": "payment*", "action": "require_approval"}])
        self.queue_check_fixture("agent-policy-check-degraded-approval.json")
        self.queue_check_fixture("agent-policy-check-degraded-approval.json")
        payload = self.run_check(tool="payment.charge", tool_input={"amount": 5})
        self.assertEqual(payload["effective"], "allow")
        self.assertIsNone(policy_hook.hook_response(payload))
        self.assertIsNotNone(self.run_check(tool="payment.charge", tool_input={"amount": 5}))
        self.assertEqual(self.http.count("/agent/policy/check"), 2, "the repeat minted its own check")

    def test_the_snapshot_fixture_parses_into_the_local_rule_store(self):
        body = fixture("agent-policy-snapshot.json")
        self.assertTrue(body["rules"], "the snapshot fixture must exercise a populated rule list")
        self.write_credentials()
        # An enforce stub with no rules is exactly the state that forces a
        # snapshot fetch before the check, so the fixture body is what lands
        # in the persisted state file.
        self.seed_snapshot(mode="enforce", rules=[], fetched_at=0)
        self.http.queue("/agent/policy/snapshot", harness.FakeResponse(200, fixture_text("agent-policy-snapshot.json")))
        # The check answer must agree on the mode: a check reply keeps the
        # snapshot mode fresh, so an audit-mode body here would overwrite the
        # very mode this test is pinning.
        self.queue_check_fixture("agent-policy-check-deny.json")
        self.run_check(tool="shell.run")
        state = json.loads((self.home / "ai-passport-policy-state.json").read_text())
        self.assertEqual(state["snapshot"]["mode"], body["mode"])
        self.assertEqual(state["snapshot"]["rules"], body["rules"])


class FixtureCoverage(unittest.TestCase):
    # The reverse gate on tests/fixtures membership: a fixture the sync lands
    # here without a test parsing it is a published shape this plugin silently
    # ignores. Kept adjacent to the suites that exercise each name; the
    # decision fixtures are deliberately absent (Hermes never polls the
    # decisions leg, so scripts/sync_shared.mjs does not vendor them here).
    EXERCISED = {
        "agent-policy-check-allow.json",
        "agent-policy-check-deny.json",
        "agent-policy-check-approval.json",
        "agent-policy-check-degraded.json",
        "agent-policy-check-degraded-deny.json",
        "agent-policy-check-degraded-grant.json",
        "agent-policy-check-degraded-approval.json",
        "agent-policy-snapshot.json",
        "agent-prefetch-response.json",
        "structured-recall-response.json",
    }

    def test_every_synced_fixture_is_exercised(self):
        synced = {path.name for path in harness.FIXTURES_DIR.glob("*.json")}
        self.assertEqual(synced, self.EXERCISED)


class StructuredRecallFixture(unittest.TestCase):
    def test_the_structured_recall_fixture_keeps_its_closed_shape(self):
        # The backend pins this shape in structured_recall_contract_smoke.mjs;
        # this side asserts the vendored copy still carries the fields a
        # consumer parses.
        body = fixture("structured-recall-response.json")
        self.assertIn(
            body["outcome"],
            {"ok", "approval_required", "locked", "unavailable", "account_unavailable", "rate_limited", "partial"},
        )
        for entry in body["categories"]:
            self.assertIsInstance(entry["category"], str)
            self.assertTrue(entry["outcome"])
            if entry["outcome"] == "locked":
                self.assertTrue(entry["unlock_url"].endswith("/memory-lock"))
            else:
                self.assertNotIn("unlock_url", entry)
            for row in entry.get("rows", []):
                for key in ("memory_id", "content", "category", "source", "created_at"):
                    self.assertIn(key, row)


class PrefetchFixture(test_client.Base):
    def test_the_prefetch_fixture_parses_into_rows(self):
        body = fixture("agent-prefetch-response.json")
        self.http.queue(test_client.PREFETCH, harness.FakeResponse(200, fixture_text("agent-prefetch-response.json")))
        result = self.read()
        self.assertEqual([row["memory_id"] for row in result.rows], [row["memory_id"] for row in body["rows"]])
        self.assertEqual([row["content"] for row in result.rows], [row["content"] for row in body["rows"]])
        self.assertEqual(result.skipped, body["skipped_categories"])
        self.assertEqual(result.approval_url, body["approval_url"])


if __name__ == "__main__":
    unittest.main()
