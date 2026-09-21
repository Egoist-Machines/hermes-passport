"""The rewrite-only .env sink that keeps the MCP bearer current."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import harness

env_sink_mod = harness.submodule("env_sink")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.env = self.home / ".env"

    def sink(self, var="AI_PASSPORT_TOKEN", enabled=True):
        return env_sink_mod.EnvTokenSink(hermes_home=str(self.home), var=var, enabled=enabled)

    def write_env(self, text: str, mode: int = 0o600):
        self.env.write_text(text, encoding="utf-8")
        os.chmod(self.env, mode)


class RewriteOnly(Base):
    def test_a_missing_env_file_is_never_created(self):
        self.assertFalse(self.sink().sync("token-1"))
        self.assertFalse(self.env.exists())

    def test_a_missing_line_is_the_owners_opt_out(self):
        # They run MCP with OAuth, or a hand-managed header, or not at all. A
        # plugin that appends a bearer token to their .env is adding a credential
        # to a file they deliberately kept clean.
        self.write_env("OTHER=1\n")
        self.assertFalse(self.sink().sync("token-1"))
        self.assertEqual(self.env.read_text(encoding="utf-8"), "OTHER=1\n")

    def test_an_existing_line_is_updated_in_place(self):
        self.write_env("A=1\nAI_PASSPORT_TOKEN=old\nB=2\n")
        self.assertTrue(self.sink().sync("token-1"))
        self.assertEqual(self.env.read_text(encoding="utf-8"), "A=1\nAI_PASSPORT_TOKEN=token-1\nB=2\n")

    def test_an_export_prefix_and_indentation_survive(self):
        self.write_env("  export AI_PASSPORT_TOKEN=old\n")
        self.assertTrue(self.sink().sync("token-1"))
        self.assertEqual(self.env.read_text(encoding="utf-8"), "  export AI_PASSPORT_TOKEN=token-1\n")

    def test_a_file_without_a_trailing_newline_stays_that_way(self):
        self.write_env("AI_PASSPORT_TOKEN=old")
        self.assertTrue(self.sink().sync("token-1"))
        self.assertEqual(self.env.read_text(encoding="utf-8"), "AI_PASSPORT_TOKEN=token-1")

    def test_an_unchanged_token_writes_nothing(self):
        self.write_env("AI_PASSPORT_TOKEN=token-1\n")
        before = self.env.stat().st_mtime_ns
        self.assertFalse(self.sink().sync("token-1"))
        self.assertEqual(self.env.stat().st_mtime_ns, before)

    def test_a_repeat_call_with_the_same_token_writes_nothing(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        sink = self.sink()
        self.assertTrue(sink.sync("token-1"))
        before = self.env.stat().st_mtime_ns
        self.assertFalse(sink.sync("token-1"))
        self.assertEqual(self.env.stat().st_mtime_ns, before)

    def test_a_drifted_line_is_healed_on_the_next_sync(self):
        # Level-triggered on purpose: rotation is not the only way the file
        # drifts (owner edit, backup restore), and an edge-triggered memo would
        # leave the MCP bearer wrong for up to an hour of 401s.
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        sink = self.sink()
        self.assertTrue(sink.sync("token-1"))
        self.write_env("AI_PASSPORT_TOKEN=hand-edited\n")
        self.assertTrue(sink.sync("token-1"))
        self.assertIn("AI_PASSPORT_TOKEN=token-1", self.env.read_text(encoding="utf-8"))

    def test_a_disabled_sink_does_nothing(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        self.assertFalse(self.sink(enabled=False).sync("token-1"))
        self.assertIn("old", self.env.read_text(encoding="utf-8"))

    def test_an_empty_token_is_ignored(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        self.assertFalse(self.sink().sync(""))

    def test_a_custom_var_name_is_honored(self):
        self.write_env("PASSPORT_BEARER=old\n")
        self.assertTrue(self.sink(var="PASSPORT_BEARER").sync("token-1"))
        self.assertIn("PASSPORT_BEARER=token-1", self.env.read_text(encoding="utf-8"))

    def test_a_similar_variable_name_is_not_matched(self):
        self.write_env("AI_PASSPORT_TOKEN_BACKUP=old\n")
        self.assertFalse(self.sink().sync("token-1"))
        self.assertIn("AI_PASSPORT_TOKEN_BACKUP=old", self.env.read_text(encoding="utf-8"))

    def test_a_differently_cased_variable_is_someone_elses(self):
        # dotenv keys are case-sensitive: a lowercase twin is a distinct
        # variable owned by another consumer, and the canonical-cased line's
        # absence is the owner's opt-out. Rewriting it would break the other
        # consumer AND add a credential the rewrite-only design promises not to.
        self.write_env("ai_passport_token=value-used-by-another-script\n")
        self.assertFalse(self.sink().sync("token-1"))
        self.assertEqual(
            self.env.read_text(encoding="utf-8"),
            "ai_passport_token=value-used-by-another-script\n",
        )

    def test_every_occurrence_is_updated(self):
        # A duplicated line means the last one wins at read time, so updating
        # only the first would leave the stale token in charge.
        self.write_env("AI_PASSPORT_TOKEN=one\nX=1\nAI_PASSPORT_TOKEN=two\n")
        self.assertTrue(self.sink().sync("token-1"))
        self.assertEqual(
            self.env.read_text(encoding="utf-8"),
            "AI_PASSPORT_TOKEN=token-1\nX=1\nAI_PASSPORT_TOKEN=token-1\n",
        )


class Safety(Base):
    def test_a_newline_in_a_token_cannot_inject_a_variable(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        self.sink().sync("evil\nMALICIOUS=1")
        text = self.env.read_text(encoding="utf-8")
        self.assertEqual(text, "AI_PASSPORT_TOKEN=evilMALICIOUS=1\n")
        self.assertNotIn("\nMALICIOUS", text)

    def test_the_file_mode_is_preserved(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n", mode=0o600)
        self.sink().sync("token-1")
        self.assertEqual(self.env.stat().st_mode & 0o777, 0o600)

    def test_no_temp_file_is_left_behind(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        self.sink().sync("token-1")
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), [".env"])

    def test_the_rest_of_the_file_survives_the_rewrite(self):
        self.write_env("# comment\nANTHROPIC_API_KEY=secret\nAI_PASSPORT_TOKEN=old\n\nTRAILING=1\n")
        self.sink().sync("token-1")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn("# comment", text)
        self.assertIn("ANTHROPIC_API_KEY=secret", text)
        self.assertIn("\n\nTRAILING=1\n", text)

    def test_the_environ_is_left_alone_without_a_host_to_ask(self):
        # Under a multiplexed gateway a token in os.environ leaks to sibling
        # profiles, so with no host answer the sink must not guess.
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        os.environ.pop("AI_PASSPORT_TOKEN", None)
        self.addCleanup(lambda: os.environ.pop("AI_PASSPORT_TOKEN", None))
        real = sys.modules.get("agent.secret_scope")
        sys.modules["agent.secret_scope"] = types.ModuleType("agent.secret_scope")
        self.addCleanup(lambda: sys.modules.__setitem__("agent.secret_scope", real))
        self.sink().sync("token-1")
        self.assertNotIn("AI_PASSPORT_TOKEN", os.environ)

    def test_the_environ_is_updated_for_a_single_profile_install(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        os.environ.pop("AI_PASSPORT_TOKEN", None)
        self.addCleanup(lambda: os.environ.pop("AI_PASSPORT_TOKEN", None))
        harness.install_host_stubs()
        self.sink().sync("token-1")
        self.assertEqual(os.environ.get("AI_PASSPORT_TOKEN"), "token-1")

    def test_a_multiplexed_gateway_never_exports(self):
        self.write_env("AI_PASSPORT_TOKEN=old\n")
        os.environ.pop("AI_PASSPORT_TOKEN", None)
        self.addCleanup(lambda: os.environ.pop("AI_PASSPORT_TOKEN", None))
        real = sys.modules.get("agent.secret_scope")
        stub = types.ModuleType("agent.secret_scope")
        stub.is_multiplex_active = lambda: True
        sys.modules["agent.secret_scope"] = stub
        self.addCleanup(lambda: sys.modules.__setitem__("agent.secret_scope", real))
        self.assertTrue(self.sink().sync("token-1"))
        # The FILE is still updated: it is per profile. Only the shared environ
        # is off limits.
        self.assertIn("AI_PASSPORT_TOKEN=token-1", self.env.read_text(encoding="utf-8"))
        self.assertNotIn("AI_PASSPORT_TOKEN", os.environ)


if __name__ == "__main__":
    unittest.main()
