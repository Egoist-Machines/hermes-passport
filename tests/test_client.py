"""The /agent/prefetch client: caching, backoff, budget and degradation."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import harness

client_mod = harness.submodule("client")
config_mod = harness.submodule("config")
credentials_mod = harness.submodule("credentials")

PREFETCH = "/agent/prefetch"
TOKEN = "/oauth/token"


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.home / "ai-passport-refresh.json"
        self.now = [1_000.0]
        self.http = harness.FakeHttp()
        harness.write_credentials(self.path, access_token="live", access_token_expires_at=10 ** 12)
        self.config = config_mod.ProviderConfig({}, str(self.home), env={})
        self.credentials = credentials_mod.Credentials(
            self.path, opener=self.http, clock=lambda: 0.0
        )
        self.client = client_mod.PassportClient(
            config=self.config,
            credentials=self.credentials,
            opener=self.http,
            clock=lambda: self.now[0],
        )

    def read(self, query=None, timeout_s=1.0, session_key=None):
        return self.client.prefetch(
            categories=self.config.categories,
            query=query,
            limit=self.config.context_limit,
            session_key=session_key,
            timeout_s=timeout_s,
        )

    def ok(self, rows=(), skipped=(), approval_url="https://my.ego.ist/passes"):
        return harness.FakeResponse(200, harness.prefetch_body(rows, skipped, approval_url))


class HappyPath(Base):
    def test_rows_are_normalized(self):
        self.http.queue(
            PREFETCH,
            self.ok(
                rows=[
                    harness.row("m1", "likes oat milk"),
                    {"memory_id": "m2", "content": "ships on fridays", "category": 7, "created_at": 5},
                    {"memory_id": None, "content": "dropped"},
                    {"content": "no id"},
                    "nonsense",
                ],
                skipped=[{"category": "event", "reason": "no_pass"}, {"category": "x"}],
            ),
        )
        result = self.read()
        self.assertEqual([row["memory_id"] for row in result.rows], ["m1", "m2"])
        # A category the backend answered outside the closed vocabulary of types
        # falls back rather than being forwarded into a prompt.
        self.assertEqual(result.rows[1]["category"], "other")
        self.assertIsNone(result.rows[1]["created_at"])
        self.assertEqual(result.skipped, [{"category": "event", "reason": "no_pass"}])
        self.assertEqual(result.approval_url, "https://my.ego.ist/passes")
        self.assertFalse(result.stale)

    def test_the_request_carries_the_declared_shape(self):
        self.http.queue(PREFETCH, self.ok())
        self.read(query="  what   do  I like?  ", session_key="session-9")
        body = self.http.bodies(PREFETCH)[0]
        self.assertEqual(body["categories"], list(self.config.categories))
        self.assertEqual(body["query"], "what do I like?")
        self.assertEqual(body["session_key"], "session-9")
        self.assertEqual(body["limit"], self.config.context_limit)
        headers = self.http.calls[0]["headers"]
        self.assertEqual(headers["authorization"], "Bearer live")

    def test_query_and_session_key_are_clamped_to_the_backends_bounds(self):
        self.http.queue(PREFETCH, self.ok())
        self.read(query="x" * 400, session_key="s" * 300)
        body = self.http.bodies(PREFETCH)[0]
        self.assertEqual(len(body["query"]), client_mod.QUERY_MAX_LENGTH)
        self.assertEqual(len(body["session_key"]), client_mod.SESSION_KEY_MAX_LENGTH)

    def test_the_clamp_counts_utf16_units_like_the_backend(self):
        # The backend rejects query.length > 256 in UTF-16 code units, and an
        # astral character (an emoji) is two of them. A code-point clamp would
        # send up to 512 units and the 400 latches the five-minute backoff.
        self.http.queue(PREFETCH, self.ok())
        self.read(query="🦄" * 300)
        query = self.http.bodies(PREFETCH)[0]["query"]
        units = sum(2 if ord(ch) > 0xFFFF else 1 for ch in query)
        self.assertLessEqual(units, client_mod.QUERY_MAX_LENGTH)
        self.assertEqual(len(query), client_mod.QUERY_MAX_LENGTH // 2)

    def test_empty_query_is_omitted_rather_than_sent_blank(self):
        self.http.queue(PREFETCH, self.ok())
        self.read(query="   ")
        self.assertNotIn("query", self.http.bodies(PREFETCH)[0])

    def test_a_repeat_read_is_served_from_cache(self):
        self.http.queue(PREFETCH, self.ok(rows=[harness.row("m1", "one")]))
        self.read()
        self.read()
        self.assertEqual(self.http.count(PREFETCH), 1)

    def test_the_session_key_is_not_part_of_the_cache_key(self):
        # The backend validates it and then ignores it, so keying on it would turn
        # one read into a cache miss per concurrent session.
        self.http.queue(PREFETCH, self.ok())
        self.read(session_key="a")
        self.read(session_key="b")
        self.assertEqual(self.http.count(PREFETCH), 1)

    def test_a_different_query_is_a_different_read(self):
        self.http.queue(PREFETCH, self.ok(), self.ok())
        self.read(query="one")
        self.read(query="two")
        self.assertEqual(self.http.count(PREFETCH), 2)

    def test_a_redirect_is_refused_and_treated_as_an_outage(self):
        # The transport never follows a redirect (it would hand the bearer to
        # whatever a middlebox points at); it surfaces as its 3xx status.
        self.http.queue(PREFETCH, harness.FakeResponse(302, "", headers={"location": "https://evil.example/"}))
        self.assertIsNone(self.read())
        self.assertEqual(self.client.state()["backoff_reason"], "unavailable")


class Degradation(Base):
    def prime(self):
        self.http.queue(PREFETCH, self.ok(rows=[harness.row("m1", "cached")]))
        self.read()
        self.now[0] += self.config.cache_ttl_ms / 1000.0 + 1

    def test_a_failure_serves_the_last_good_answer(self):
        self.prime()
        self.http.queue(PREFETCH, harness.FakeResponse(503, "{}"))
        result = self.read()
        self.assertTrue(result.stale)
        self.assertEqual(result.rows[0]["memory_id"], "m1")

    def test_a_failure_without_cache_answers_none(self):
        self.http.queue(PREFETCH, harness.FakeResponse(503, "{}"))
        self.assertIsNone(self.read())

    def test_a_transport_error_is_not_raised(self):
        self.http.queue(PREFETCH, OSError("connection reset"))
        self.assertIsNone(self.read())
        self.assertEqual(self.client.state()["backoff_reason"], "unavailable")

    def test_a_malformed_200_is_treated_as_an_outage(self):
        # A 200 whose body is not the plane's JSON is a middlebox answering in
        # the backend's place, not an empty Passport.
        self.prime()
        self.http.queue(PREFETCH, harness.FakeResponse(200, "<html>captive portal</html>"))
        result = self.read()
        self.assertTrue(result.stale)
        self.assertEqual(self.client.state()["backoff_reason"], "unavailable")

    def test_a_json_array_200_is_treated_as_an_outage(self):
        self.http.queue(PREFETCH, harness.FakeResponse(200, "[]"))
        self.assertIsNone(self.read())
        self.assertEqual(self.client.state()["backoff_reason"], "unavailable")

    def test_backoff_stops_further_requests(self):
        self.http.queue(PREFETCH, harness.FakeResponse(503, "{}"))
        self.read()
        self.read(query="another")
        self.assertEqual(self.http.count(PREFETCH), 1)

    def test_backoff_expires(self):
        self.http.queue(PREFETCH, harness.FakeResponse(503, "{}"), self.ok())
        self.read()
        self.now[0] += client_mod.BACKOFF_S["unavailable"] + 1
        self.assertIsNotNone(self.read())
        self.assertEqual(self.http.count(PREFETCH), 2)

    def test_a_429_backs_off_for_a_cache_window(self):
        self.http.queue(PREFETCH, harness.FakeResponse(429, json.dumps({"error": "rate_limited"})))
        self.read()
        self.assertEqual(self.client.state()["backoff_reason"], "rate_limited")

    def test_a_400_is_a_version_skew_and_backs_off_hard(self):
        self.http.queue(PREFETCH, harness.FakeResponse(400, json.dumps({"error": "invalid_categories"})))
        self.read()
        self.now[0] += client_mod.BACKOFF_S["invalid_request"] - 1
        self.read(query="later")
        self.assertEqual(self.http.count(PREFETCH), 1)

    def test_one_403_is_probed_past_but_a_repeat_latches(self):
        self.http.queue(
            PREFETCH,
            harness.FakeResponse(403, "{}"),
            harness.FakeResponse(403, "{}"),
        )
        self.read()
        # A single 403 can be one transient store blip on the backend's
        # verification path, so the first only pauses for the probe window.
        self.now[0] += client_mod.BACKOFF_S["forbidden_probe"] + 1
        self.read()
        self.assertEqual(self.http.count(PREFETCH), 2)
        self.now[0] += client_mod.BACKOFF_S["forbidden_probe"] + 1
        self.read()
        self.assertEqual(self.http.count(PREFETCH), 2)

    def test_one_turns_two_concurrent_403s_count_as_one_blip(self):
        # A turn fires the query read and the recent read in parallel; both
        # answering 403 on one transient store blip must not read as the repeat
        # that latches the five-minute window.
        started = []
        release = threading.Event()

        def slow_403(request):
            started.append(1)
            release.wait(2.0)
            return harness.FakeResponse(403, "{}")

        self.http.queue(PREFETCH, slow_403, slow_403)
        threads = [
            threading.Thread(target=lambda q=q: self.read(query=q, timeout_s=5.0)) for q in ("one", None)
        ]
        for thread in threads:
            thread.start()
        for _ in range(100):
            if len(started) == 2:
                break
            threading.Event().wait(0.01)
        release.set()
        for thread in threads:
            thread.join(5.0)
        # Still on the probe window: past it, the next read goes through.
        self.now[0] += client_mod.BACKOFF_S["forbidden_probe"] + 1
        self.http.queue(PREFETCH, self.ok())
        self.assertIsNotNone(self.read(query="after the probe window"))

    def test_a_404_is_the_dark_flag_not_an_outage(self):
        # While the plane is dark Express answers its default 404.
        self.http.queue(PREFETCH, harness.FakeResponse(404, "not found"))
        self.read()
        self.assertEqual(self.client.state()["backoff_reason"], "forbidden")

    def test_a_success_clears_the_backoff(self):
        self.http.queue(PREFETCH, harness.FakeResponse(503, "{}"), self.ok())
        self.read()
        self.now[0] += client_mod.BACKOFF_S["unavailable"] + 1
        self.read()
        self.assertIsNone(self.client.state()["backoff_reason"])


class Authentication(Base):
    def test_a_401_forces_one_refresh_and_one_retry(self):
        self.http.queue(PREFETCH, harness.FakeResponse(401, "{}"), self.ok(rows=[harness.row("m1", "after")]))
        self.http.queue(TOKEN, harness.FakeResponse(200, json.dumps({"access_token": "fresh", "expires_in": 3600})))
        result = self.read()
        self.assertEqual(result.rows[0]["memory_id"], "m1")
        self.assertEqual(self.http.count(TOKEN), 1)
        self.assertEqual(self.http.calls[-1]["headers"]["authorization"], "Bearer fresh")

    def test_a_401_after_the_refresh_gives_up_for_the_window(self):
        self.http.queue(PREFETCH, harness.FakeResponse(401, "{}"), harness.FakeResponse(401, "{}"))
        self.http.queue(TOKEN, harness.FakeResponse(200, json.dumps({"access_token": "fresh", "expires_in": 3600})))
        self.assertIsNone(self.read())
        self.assertEqual(self.client.state()["backoff_reason"], "auth")

    def test_a_missing_install_latches_terminal_and_stops_the_network(self):
        self.path.unlink()
        self.credentials._reset_for_tests()
        self.assertIsNone(self.read())
        self.assertEqual(self.client.state()["terminal_reason"], "not_installed")
        self.assertEqual(self.http.calls, [])

    def test_a_terminal_verdict_is_re_probed_after_the_recheck_window(self):
        self.path.unlink()
        self.credentials._reset_for_tests()
        self.read()
        # The documented recovery is the owner rewriting the file, and a
        # long-running gateway must pick that up without a restart.
        harness.write_credentials(self.path, access_token="live", access_token_expires_at=10 ** 12)
        self.credentials._reset_for_tests()
        self.now[0] += client_mod.TERMINAL_RECHECK_S + 1
        self.http.queue(PREFETCH, self.ok(rows=[harness.row("m1", "back")]))
        result = self.read()
        self.assertIsNotNone(result)
        self.assertIsNone(self.client.state()["terminal_reason"])


class Budget(Base):
    def test_the_client_spends_at_most_half_the_backends_throttle(self):
        for index in range(client_mod.REQUEST_BUDGET_MAX + 5):
            self.http.queue(PREFETCH, self.ok())
        for index in range(client_mod.REQUEST_BUDGET_MAX):
            self.read(query=f"q{index}")
        self.assertEqual(self.http.count(PREFETCH), client_mod.REQUEST_BUDGET_MAX)
        # Novel prompt text makes most reads cache misses, so an uncapped process
        # would trip the backend throttle and black out every surface.
        self.assertIsNone(self.read(query="one too many"))
        self.assertEqual(self.http.count(PREFETCH), client_mod.REQUEST_BUDGET_MAX)

    def test_the_window_slides(self):
        for index in range(client_mod.REQUEST_BUDGET_MAX + 2):
            self.http.queue(PREFETCH, self.ok())
        for index in range(client_mod.REQUEST_BUDGET_MAX):
            self.read(query=f"q{index}")
        self.now[0] += client_mod.REQUEST_BUDGET_WINDOW_S + 1
        self.assertIsNotNone(self.read(query="after the window"))


class SingleFlight(Base):
    def test_concurrent_identical_reads_share_one_request(self):
        started = threading.Event()
        release = threading.Event()

        def slow(request):
            started.set()
            release.wait(2.0)
            return self.ok(rows=[harness.row("m1", "shared")])

        self.http.queue(PREFETCH, slow, self.ok())
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.read(timeout_s=5.0))) for _ in range(4)]
        for thread in threads:
            thread.start()
        started.wait(2.0)
        release.set()
        for thread in threads:
            thread.join(5.0)
        self.assertEqual(self.http.count(PREFETCH), 1)
        self.assertEqual({result.rows[0]["memory_id"] for result in results}, {"m1"})

    def test_a_follower_that_runs_out_of_budget_serves_stale(self):
        self.http.queue(PREFETCH, self.ok(rows=[harness.row("m1", "cached")]))
        self.read()
        self.now[0] += self.config.cache_ttl_ms / 1000.0 + 1

        started = threading.Event()
        release = threading.Event()

        def slow(request):
            started.set()
            release.wait(2.0)
            return self.ok(rows=[harness.row("m2", "fresh")])

        self.http.queue(PREFETCH, slow)
        leader_result = []
        leader = threading.Thread(target=lambda: leader_result.append(self.read(timeout_s=5.0)))
        leader.start()
        started.wait(2.0)
        # The follower will not hold the turn open for the leader's slower read.
        follower = self.read(timeout_s=0.05)
        self.assertTrue(follower.stale)
        self.assertEqual(follower.rows[0]["memory_id"], "m1")
        release.set()
        leader.join(5.0)
        self.assertEqual(leader_result[0].rows[0]["memory_id"], "m2")


if __name__ == "__main__":
    unittest.main()
