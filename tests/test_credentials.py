"""Credential reads and the single-use refresh-token rotation."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import harness

credentials_mod = harness.submodule("credentials")
Credentials = credentials_mod.Credentials
PassportAuthError = credentials_mod.PassportAuthError


def token_response(access_token="access-1", refresh_token="refresh-2", expires_in=3600):
    payload = {"access_token": access_token, "expires_in": expires_in}
    if refresh_token is not None:
        payload["refresh_token"] = refresh_token
    return harness.FakeResponse(200, json.dumps(payload))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.home / "ai-passport-refresh.json"
        self.http = harness.FakeHttp()
        self.now = [1_000_000.0]
        self.announced = []

    def build(self, **kwargs):
        return Credentials(
            self.path,
            opener=self.http,
            clock=lambda: self.now[0],
            on_access_token=self.announced.append,
            **kwargs,
        )

    def stored(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class Reading(Base):
    def test_missing_file_is_terminal(self):
        credentials = self.build()
        with self.assertRaises(PassportAuthError) as caught:
            credentials.access_token()
        self.assertTrue(caught.exception.terminal)
        self.assertEqual(caught.exception.code, "not_installed")
        self.assertEqual(self.http.calls, [])

    def test_unparseable_file_is_not_terminal(self):
        self.path.write_text("{oops", encoding="utf-8")
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertFalse(caught.exception.terminal)
        self.assertEqual(caught.exception.code, "invalid_file")

    def test_json_array_is_rejected(self):
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "invalid_file")

    def test_missing_fields_are_rejected(self):
        harness.write_credentials(self.path, refresh_token="")
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "invalid_file")

    def test_a_file_that_stats_but_cannot_be_read_is_a_plain_error(self):
        # A directory at the path stats fine and then fails the read; the CLI
        # promises a plain error line, never a traceback.
        self.path.mkdir()
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "unreadable")
        self.assertFalse(caught.exception.terminal)

    def test_base_url_comes_from_the_token_url(self):
        harness.write_credentials(self.path, token_url="https://passport.test:8443/oauth/token")
        self.assertEqual(self.build().base_url(), "https://passport.test:8443")

    def test_unusable_token_url_is_rejected(self):
        harness.write_credentials(self.path, token_url="not-a-url")
        with self.assertRaises(PassportAuthError):
            self.build().base_url()

    def test_installed_is_a_file_check(self):
        credentials = self.build()
        self.assertFalse(credentials.installed())
        harness.write_credentials(self.path)
        self.assertTrue(credentials.installed())

    def test_summary_never_carries_the_token(self):
        harness.write_credentials(self.path, access_token="secret-token", access_token_expires_at=self.now[0] + 3600)
        summary = self.build().summary()
        self.assertNotIn("secret-token", json.dumps(summary))
        self.assertTrue(summary["has_access_token"])
        self.assertEqual(summary["client_id"], "client-abc")


class Rotation(Base):
    def test_stored_token_is_reused_inside_the_margin(self):
        harness.write_credentials(
            self.path, access_token="live", access_token_expires_at=self.now[0] + 3600
        )
        credentials = self.build()
        self.assertEqual(credentials.access_token(), "live")
        self.assertEqual(self.http.calls, [])
        # Reconciled on every read so the MCP bearer cannot drift.
        self.assertEqual(self.announced, ["live"])

    def test_token_inside_the_margin_is_served_while_rotating_in_the_background(self):
        # Blocking a reader on a network POST for a token it already holds
        # would stall every turn's prefetch during the pre-expiry margin.
        harness.write_credentials(
            self.path, access_token="still-valid", access_token_expires_at=self.now[0] + 60
        )
        self.http.queue("/oauth/token", token_response())
        credentials = self.build()
        self.assertEqual(credentials.access_token(), "still-valid")
        credentials._join_background_rotation()
        self.assertEqual(self.stored()["refresh_token"], "refresh-2")
        self.assertEqual(self.stored()["access_token"], "access-1")
        # The next read serves the rotation's answer.
        self.assertEqual(credentials.access_token(), "access-1")
        self.assertEqual(self.http.count("/oauth/token"), 1)

    def test_an_expired_token_blocks_on_the_rotation(self):
        harness.write_credentials(
            self.path, access_token="dead", access_token_expires_at=self.now[0] - 1
        )
        self.http.queue("/oauth/token", token_response())
        self.assertEqual(self.build().access_token(), "access-1")

    def test_a_failed_background_rotation_is_swallowed(self):
        harness.write_credentials(
            self.path, access_token="still-valid", access_token_expires_at=self.now[0] + 60
        )
        self.http.queue("/oauth/token", OSError("connection reset"))
        credentials = self.build()
        self.assertEqual(credentials.access_token(), "still-valid")
        credentials._join_background_rotation()
        # The token being rotated was still valid, so nothing was lost.
        self.assertEqual(self.stored()["refresh_token"], "refresh-1")

    def test_concurrent_forced_rotations_spend_one_refresh_token(self):
        import time

        harness.write_credentials(
            self.path, access_token="rejected", access_token_expires_at=self.now[0] + 3600
        )
        started = threading.Event()
        release = threading.Event()

        def slow(request):
            started.set()
            release.wait(2.0)
            return token_response()

        self.http.queue("/oauth/token", slow, token_response(access_token="second"))
        credentials = self.build()
        results = []

        def forced():
            results.append(credentials.access_token(force=True))

        first = threading.Thread(target=forced)
        first.start()
        started.wait(2.0)
        # The second recovery races in while the first's POST is still in
        # flight, so both observed the same rejected token.
        second = threading.Thread(target=forced)
        second.start()
        time.sleep(0.15)  # let it read the observed token and park at the gate
        release.set()
        first.join(5.0)
        second.join(5.0)
        # Concurrent 401 recoveries collapse into one rotation: whoever waits
        # on the gate finds a fresher token than the one it watched fail.
        self.assertEqual(self.http.count("/oauth/token"), 1)
        self.assertEqual(set(results), {"access-1"})

    def test_rotation_persists_and_keeps_the_file_owner_only(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", token_response())
        self.build().access_token()
        stored = self.stored()
        self.assertEqual(stored["access_token"], "access-1")
        self.assertEqual(stored["refresh_token"], "refresh-2")
        self.assertEqual(stored["access_token_expires_at"], self.now[0] + 3600)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.announced, ["access-1"])

    def test_rotation_leaves_no_temp_file_behind(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", token_response())
        self.build().access_token()
        leftovers = sorted(p.name for p in self.home.iterdir())
        self.assertEqual(leftovers, ["ai-passport-refresh.json"])

    def test_unknown_fields_in_the_file_survive_a_rotation(self):
        harness.write_credentials(self.path, app="hermes", note="written by the guide")
        self.http.queue("/oauth/token", token_response())
        self.build().access_token()
        stored = self.stored()
        self.assertEqual(stored["app"], "hermes")
        self.assertEqual(stored["note"], "written by the guide")

    def test_a_response_without_a_new_refresh_token_keeps_the_old_one(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", token_response(refresh_token=None))
        self.build().access_token()
        self.assertEqual(self.stored()["refresh_token"], "refresh-1")

    def test_a_response_without_an_access_token_is_an_error(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", harness.FakeResponse(200, json.dumps({"expires_in": 60})))
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "token_error")

    def test_transport_failure_is_not_terminal(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", OSError("connection reset"))
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "unreachable")
        self.assertFalse(caught.exception.terminal)

    def test_a_severed_registration_is_terminal(self):
        # invalid_client means the client registration is gone, which no retry
        # fixes. The backend answers a 500 when its store cannot be read, so this
        # is never a transient blip.
        harness.write_credentials(self.path)
        self.http.queue(
            "/oauth/token",
            harness.FakeResponse(400, json.dumps({"error": "invalid_client", "error_description": "Invalid client_id"})),
        )
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertTrue(caught.exception.terminal)
        self.assertEqual(caught.exception.code, "invalid_client")
        self.assertIn("connect code", caught.exception.args[0])
        # One POST and done: nothing about a terminal verdict is retried.
        self.assertEqual(self.http.count("/oauth/token"), 1)

    def test_a_401_invalid_client_is_also_terminal(self):
        # The OAuth spec allows invalid_client on a 401 as well.
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", harness.FakeResponse(401, json.dumps({"error": "invalid_client"})))
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertTrue(caught.exception.terminal)

    def test_a_redirect_is_refused_not_followed(self):
        # Following it would hand the refresh token to the redirect target.
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", harness.FakeResponse(302, "", headers={"location": "https://evil.example/"}))
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "unreachable")
        self.assertFalse(caught.exception.terminal)

    def test_server_error_is_not_terminal(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", harness.FakeResponse(503, "{}"))
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertEqual(caught.exception.code, "token_error")
        self.assertFalse(caught.exception.terminal)

    def test_one_rotation_at_a_time(self):
        harness.write_credentials(self.path)
        started = threading.Event()
        release = threading.Event()

        def slow(request):
            started.set()
            release.wait(2.0)
            return token_response()

        self.http.queue("/oauth/token", slow, token_response(access_token="second"))
        credentials = self.build()
        results = []
        threads = [threading.Thread(target=lambda: results.append(credentials.access_token())) for _ in range(3)]
        for thread in threads:
            thread.start()
        started.wait(2.0)
        release.set()
        for thread in threads:
            thread.join(5.0)
        # The refresh token is single-use: two concurrent rotations would spend it
        # twice and revoke the install. Later callers see the fresh stored token.
        self.assertEqual(self.http.count("/oauth/token"), 1)
        self.assertEqual(set(results), {"access-1"})


class TwoRefresherRace(Base):
    """The agent's own curl and this provider share one rotating token.

    The backend settles that race now (five-minute sealed recovery copy,
    returned to a replay from the same client address), so the provider does
    exactly one POST per rotation and treats what comes back as the answer.
    Retrying a spent token past that window is classified as theft and revokes
    the whole family, so these tests pin "never retry" as the contract.
    """

    def test_a_dead_token_is_terminal_immediately(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", harness.FakeResponse(400, json.dumps({"error": "invalid_grant"})))
        with self.assertRaises(PassportAuthError) as caught:
            self.build().access_token()
        self.assertTrue(caught.exception.terminal)
        self.assertEqual(caught.exception.code, "invalid_grant")
        self.assertEqual(self.http.count("/oauth/token"), 1)

    def test_a_recovered_replay_is_an_ordinary_success(self):
        # The winner rotated first; the backend hands us the sealed copy of the
        # response it already issued, so both refreshers land on one chain.
        harness.write_credentials(self.path)
        credentials = self.build()
        self.http.queue("/oauth/token", token_response(access_token="the-winners-token", refresh_token="refresh-2"))
        self.assertEqual(credentials.access_token(), "the-winners-token")
        self.assertEqual(self.http.count("/oauth/token"), 1)
        self.assertEqual(self.stored()["refresh_token"], "refresh-2")

    def test_a_spent_token_is_never_re_posted_after_the_file_moves(self):
        # The old behaviour re-read the file and spent the token it found. Past
        # the recovery window that is a replay against an already-revoked
        # family, so the file moving must NOT produce a second POST.
        harness.write_credentials(self.path)
        credentials = self.build()

        def rewrite_then_fail(request):
            harness.write_credentials(self.path, refresh_token="refresh-from-the-winner")
            return harness.FakeResponse(400, json.dumps({"error": "invalid_grant"}))

        self.http.queue("/oauth/token", rewrite_then_fail, token_response(access_token="must-not-be-reached"))
        with self.assertRaises(PassportAuthError) as caught:
            credentials.access_token()
        self.assertTrue(caught.exception.terminal)
        self.assertEqual(self.http.count("/oauth/token"), 1)

    def test_the_refresh_post_never_sends_a_resource_parameter(self):
        # Sending it cannot change which audience the rotation binds (the
        # server uses the one already stored on the chain) and a value that is
        # not byte-equal to the canonical issuer answers invalid_grant, which
        # this provider latches as terminal. See _request_token.
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", token_response())
        self.build().access_token()
        self.assertNotIn("resource", self.http.calls[0]["body"])


class DiskFailure(Base):
    def test_a_rotated_token_survives_a_failed_write(self):
        harness.write_credentials(self.path)
        original = credentials_mod._write_atomically
        failures = {"count": 0}

        def failing(path, payload):
            failures["count"] += 1
            raise OSError("disk full")

        credentials_mod._write_atomically = failing
        self.addCleanup(lambda: setattr(credentials_mod, "_write_atomically", original))

        self.http.queue("/oauth/token", token_response())
        credentials = self.build()
        self.assertEqual(credentials.access_token(), "access-1")
        self.assertEqual(failures["count"], 1)
        # Disk still holds the SPENT token; memory holds the live one. Re-reading
        # the file here would lose the only live copy and brick the install.
        self.assertEqual(self.stored()["refresh_token"], "refresh-1")
        self.assertEqual(credentials.summary()["pending_disk_write"], True)
        self.assertEqual(credentials.access_token(), "access-1")
        self.assertEqual(self.http.count("/oauth/token"), 1)

    def test_the_file_heals_on_a_later_read(self):
        harness.write_credentials(self.path)
        original = credentials_mod._write_atomically
        state = {"fail": True}

        def sometimes(path, payload):
            if state["fail"]:
                raise OSError("disk full")
            return original(path, payload)

        credentials_mod._write_atomically = sometimes
        self.addCleanup(lambda: setattr(credentials_mod, "_write_atomically", original))

        self.http.queue("/oauth/token", token_response())
        credentials = self.build()
        credentials.access_token()
        state["fail"] = False
        credentials.access_token()
        self.assertEqual(self.stored()["refresh_token"], "refresh-2")
        self.assertFalse(credentials.summary()["pending_disk_write"])
        self.assertEqual(self.http.count("/oauth/token"), 1)


class FileWatching(Base):
    def test_an_external_rewrite_is_picked_up(self):
        harness.write_credentials(self.path, access_token="first", access_token_expires_at=self.now[0] + 3600)
        credentials = self.build()
        self.assertEqual(credentials.access_token(), "first")
        # Another process (the agent's own 401 path) rewrites the file.
        harness.write_credentials(self.path, access_token="second", access_token_expires_at=self.now[0] + 3600)
        os.utime(self.path, (self.now[0] + 10, self.now[0] + 10))
        self.assertEqual(credentials.access_token(), "second")
        self.assertEqual(self.http.calls, [])

    def test_force_skips_the_stored_token(self):
        harness.write_credentials(self.path, access_token="live", access_token_expires_at=self.now[0] + 3600)
        self.http.queue("/oauth/token", token_response(access_token="forced"))
        self.assertEqual(self.build().access_token(force=True), "forced")


class Sink(Base):
    def test_a_raising_sink_does_not_fail_the_refresh(self):
        harness.write_credentials(self.path)
        self.http.queue("/oauth/token", token_response())

        def boom(token):
            raise RuntimeError("no")

        credentials = Credentials(
            self.path, opener=self.http, clock=lambda: self.now[0], on_access_token=boom
        )
        self.assertEqual(credentials.access_token(), "access-1")
        self.assertEqual(self.stored()["access_token"], "access-1")


if __name__ == "__main__":
    unittest.main()
