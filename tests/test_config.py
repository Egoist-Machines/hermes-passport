"""Config normalization, and the vocabulary drift gate against the backend."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import harness

config_mod = harness.submodule("config")


class CategoryVocabulary(unittest.TestCase):
    """The provider's category list must not drift from the backend's.

    The wire rejects an unknown category with a 400 that backs the whole
    provider off for five minutes, and a category the backend gained but the
    provider never asks for is memory the owner approved and never sees.
    """

    def _contracts_source(self) -> str:
        path = harness.REPO_ROOT / "shared" / "contracts.js"
        self.assertTrue(path.exists(), f"expected the backend contract at {path}")
        return path.read_text(encoding="utf-8")

    def _js_array(self, source: str, name: str) -> list:
        match = re.search(rf"export const {name} = Object\.freeze\(\[(.*?)\]\)", source, re.DOTALL)
        self.assertIsNotNone(match, f"could not find {name} in shared/contracts.js")
        return re.findall(r'"([a-z_]+)"', match.group(1))

    def test_governed_categories_match_the_backend(self):
        expected = self._js_array(self._contracts_source(), "MEMORY_CATEGORIES")
        self.assertEqual(list(config_mod.GOVERNED_CATEGORIES), expected)

    def test_derived_categories_match_the_backend(self):
        source = self._contracts_source()
        expected = re.search(r'export const DERIVED_MEMORY_CATEGORIES = Object\.freeze\(\[(.*?)\]\)', source)
        self.assertIsNotNone(expected)
        self.assertEqual(list(config_mod.DERIVED_CATEGORIES), re.findall(r'"([a-z_]+)"', expected.group(1)))

    def test_writable_excludes_the_derived_ones(self):
        self.assertNotIn("claim", config_mod.WRITABLE_CATEGORIES)
        self.assertEqual(
            list(config_mod.WRITABLE_CATEGORIES),
            [c for c in config_mod.GOVERNED_CATEGORIES if c not in config_mod.DERIVED_CATEGORIES],
        )

    def test_defaults_are_all_governed(self):
        for category in config_mod.DEFAULT_CATEGORIES:
            self.assertIn(category, config_mod.GOVERNED_CATEGORIES)


class Normalization(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def config(self, raw, env=None):
        return config_mod.ProviderConfig(raw, self.home, env=env or {})

    def test_garbage_falls_back_to_defaults(self):
        for raw in (None, [], "nope", 7):
            config = self.config(raw)
            self.assertEqual(config.categories, config_mod.DEFAULT_CATEGORIES)
            self.assertEqual(config.context_limit, config_mod.DEFAULTS["context"]["limit"])
            self.assertTrue(config.context_enabled)

    def test_out_of_range_values_are_clamped(self):
        config = self.config({"cache_ttl_ms": 10, "context": {"limit": 900, "timeout_ms": -5}})
        self.assertEqual(config.cache_ttl_ms, 1_000)
        self.assertEqual(config.context_limit, 50)
        self.assertEqual(config.context_timeout_ms, 200)

    def test_unknown_categories_are_dropped(self):
        config = self.config({"categories": ["preference", "nonsense", "fact", "fact"]})
        self.assertEqual(config.categories, ("preference", "fact"))

    def test_all_unknown_categories_fall_back_to_the_default_set(self):
        config = self.config({"categories": ["nonsense"]})
        self.assertEqual(config.categories, config_mod.DEFAULT_CATEGORIES)

    def test_string_booleans_are_coerced(self):
        # `hermes memory setup` prompts free text and ignores schema types.
        config = self.config({"context": {"enabled": "false", "send_prompt_as_query": "no"}})
        self.assertFalse(config.context_enabled)
        self.assertFalse(config.send_prompt_as_query)
        config = self.config({"context": {"enabled": "TRUE"}})
        self.assertTrue(config.context_enabled)

    def test_string_numbers_are_coerced(self):
        config = self.config({"context": {"limit": "9", "max_chars": "1500"}})
        self.assertEqual(config.context_limit, 9)
        self.assertEqual(config.context_max_chars, 1500)

    def test_booleans_are_not_numbers(self):
        config = self.config({"context": {"limit": True}})
        self.assertEqual(config.context_limit, config_mod.DEFAULTS["context"]["limit"])

    def test_credentials_path_defaults_beside_the_config(self):
        config = self.config({})
        self.assertEqual(config.credentials_path, Path(self.home) / config_mod.CREDENTIALS_FILE_NAME)

    def test_credentials_path_override(self):
        config = self.config({"credentials_path": " /tmp/elsewhere.json "})
        self.assertEqual(config.credentials_path, Path("/tmp/elsewhere.json"))

    def test_base_url_from_env_when_config_is_silent(self):
        config = self.config({}, env={"AI_PASSPORT_BASE_URL": "http://localhost:3020/"})
        self.assertEqual(config.base_url, "http://localhost:3020")

    def test_config_base_url_wins_over_env(self):
        config = self.config({"base_url": "https://passport.test"}, env={"AI_PASSPORT_BASE_URL": "http://other"})
        self.assertEqual(config.base_url, "https://passport.test")

    def test_base_url_none_means_derive_from_credentials(self):
        self.assertIsNone(self.config({}).base_url)


class BaseUrlNormalization(unittest.TestCase):
    def test_rejects_non_http_schemes(self):
        for value in ("file:///etc/passwd", "ftp://host", "javascript:alert(1)", "", "   ", None, 5):
            self.assertIsNone(config_mod.normalize_base_url(value))

    def test_strips_trailing_slashes_and_keeps_a_path_prefix(self):
        self.assertEqual(config_mod.normalize_base_url("https://h.test/base/"), "https://h.test/base")
        self.assertEqual(config_mod.normalize_base_url(" https://h.test "), "https://h.test")

    def test_drops_query_and_fragment(self):
        self.assertEqual(config_mod.normalize_base_url("https://h.test/x?y=1#z"), "https://h.test/x")


class ConfigFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(config_mod.read_config_file(self.home), {})

    def test_unparseable_file_reads_as_empty(self):
        config_mod.config_path(self.home).write_text("{not json", encoding="utf-8")
        self.assertEqual(config_mod.read_config_file(self.home), {})

    def test_non_object_file_reads_as_empty(self):
        config_mod.config_path(self.home).write_text("[1,2]", encoding="utf-8")
        self.assertEqual(config_mod.read_config_file(self.home), {})

    def test_write_merges_rather_than_replaces(self):
        config_mod.write_config_file(self.home, {"categories": ["fact"], "context": {"limit": 3}})
        config_mod.write_config_file(self.home, {"categories": ["preference"]})
        saved = json.loads(config_mod.config_path(self.home).read_text(encoding="utf-8"))
        self.assertEqual(saved["categories"], ["preference"])
        # A hand-edited budget survives a re-run of setup.
        self.assertEqual(saved["context"], {"limit": 3})

    def test_written_file_is_owner_only(self):
        config_mod.write_config_file(self.home, {"categories": ["fact"]})
        mode = config_mod.config_path(self.home).stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_written_file_leaves_no_temp_behind(self):
        config_mod.write_config_file(self.home, {"categories": ["fact"]})
        leftovers = [p.name for p in Path(self.home).iterdir() if p.name != config_mod.CONFIG_FILE_NAME]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
