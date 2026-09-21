"""Compatibility with the way Hermes discovers and loads this plugin.

These are contract tests against the host loader's behaviour, not against our
own code, and they exist because the loader has two sharp edges:

- discovery is a TEXT SCAN of ``__init__.py`` for ``MemoryProvider`` or
  ``register_memory_provider`` in the first 8 KiB, so a refactor that moves the
  class out of that file makes the plugin invisible with no error anywhere;
- the CLI is loaded under a package SHELL whose ``__init__.py`` never executes,
  so ``cli.py`` may import real submodules and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from unittest import mock

import harness

PLUGIN_DIR = harness.PLUGIN_DIR


class HostDiscovery(unittest.TestCase):
    def test_an_unusable_hermes_agent_dir_is_an_error_not_a_fallback(self):
        # HERMES_AGENT_DIR is the canary's and the release evidence's "test
        # against THIS host" switch. Falling back silently once meant a typo'd
        # path (or a host that moved agent/memory_provider.py) still printed
        # the "real Hermes" banner while the suite tested a different install
        # or the stubs.
        with mock.patch.dict("os.environ", {"HERMES_AGENT_DIR": "/nonexistent/hermes-agent"}):
            with self.assertRaisesRegex(RuntimeError, "HERMES_AGENT_DIR"):
                harness.hermes_agent_dir()


def manifest() -> dict:
    """Read plugin.yaml without a yaml dependency (it is a flat file)."""
    values: dict = {}
    for line in (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith(" ") or line.startswith("-"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip().strip('"')
    return values


class Discovery(unittest.TestCase):
    def test_the_text_scan_finds_this_plugin(self):
        source = (PLUGIN_DIR / "__init__.py").read_text(encoding="utf-8")[:8192]
        self.assertTrue("register_memory_provider" in source or "MemoryProvider" in source)

    def test_the_manifest_name_matches_the_provider_name(self):
        # `hermes plugins install` names the installed directory after this, and
        # `memory.provider` resolves a provider by directory name, so a mismatch
        # makes the provider unselectable.
        provider = harness.load_provider().AiPassportMemoryProvider()
        self.assertEqual(manifest()["name"], provider.name)

    def test_the_manifest_name_is_the_literal_the_guide_hard_codes(self):
        # marketing/lib/hermes-install.js and the sync workflow's mirror both
        # hard-code ai_passport (the install command, `memory.provider`, the
        # CLI namespace, and the status examples). Renaming the provider means
        # changing the guide too; this is the one place that pins the literal,
        # so the sync workflow can run this suite instead of restating the
        # check in shell.
        self.assertEqual(manifest()["name"], "ai_passport")

    def test_the_installed_directory_name_is_import_safe(self):
        name = manifest()["name"]
        self.assertTrue(name.isidentifier(), f"{name} must be a valid Python identifier")

    def test_the_manifest_declares_no_pip_dependencies(self):
        # A sealed venv (Docker, a `hermes update` rebuild) must not be able to
        # break memory, so this plugin stays on the standard library.
        declarations = [
            line
            for line in (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8").splitlines()
            if line.startswith("pip_dependencies")
        ]
        self.assertEqual(declarations, [])

    def test_the_module_exposes_register(self):
        self.assertTrue(callable(harness.load_provider().register))

    def test_submodules_import_cleanly_in_any_order(self):
        # The host pre-loads *.py in glob order and only debug-logs a failure, so
        # an import-order dependency would silently disable the plugin.
        for name in ("cache", "client", "config", "credentials", "env_sink", "formatting", "mcp", "status", "transport"):
            self.assertIsNotNone(harness.submodule(name))


class CliOnlyLoad(unittest.TestCase):
    """The path where the package's __init__.py is never executed."""

    def test_cli_imports_without_the_package_body(self):
        module = harness.load_cli()
        self.assertTrue(callable(module.register_cli))
        # Nothing from __init__.py was needed to get here.
        shell = sys.modules[f"{harness.CLI_NAMESPACE}.{harness.PLUGIN_NAME}"]
        self.assertIsNone(getattr(shell, "__file__", None))

    def test_the_handler_is_named_the_way_the_host_looks_it_up(self):
        # plugins/memory: getattr(cli_mod, f"{active_provider}_command").
        module = harness.load_cli()
        self.assertTrue(callable(getattr(module, f"{manifest()['name']}_command", None)))

    def test_register_cli_builds_the_subcommands(self):
        module = harness.load_cli()
        parser = argparse.ArgumentParser(prog="hermes ai_passport")
        module.register_cli(parser)
        for command in ("status", "probe", "refresh", "setup"):
            args = parser.parse_args([command])
            self.assertEqual(args.ai_passport_command, command)

    def test_probe_accepts_a_query(self):
        module = harness.load_cli()
        parser = argparse.ArgumentParser(prog="hermes ai_passport")
        module.register_cli(parser)
        args = parser.parse_args(["probe", "--query", "codeword", "--json"])
        self.assertEqual(args.query, "codeword")
        self.assertTrue(args.json)

    def test_cli_imports_only_real_submodules(self):
        source = (PLUGIN_DIR / "cli.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from ."):
                continue
            # `from . import x` and `from .. import x` resolve through the shell's
            # never-executed __init__.py and fail at runtime.
            self.assertFalse(
                stripped.startswith("from . import") or stripped.startswith("from .. "),
                f"cli.py cannot use {stripped!r}: the package body is not executed on this path",
            )


if __name__ == "__main__":
    unittest.main()
