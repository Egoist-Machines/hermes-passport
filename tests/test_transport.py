"""The shared no-redirect opener."""

from __future__ import annotations

import io
import unittest
import urllib.error
import urllib.request

import harness

transport = harness.submodule("transport")


class RefuseRedirects(unittest.TestCase):
    def test_the_handler_refuses_every_redirect_class(self):
        handler = transport._RefuseRedirects()
        request = urllib.request.Request("https://passport.test/agent/prefetch", data=b"{}", method="POST")
        for code in (301, 302, 303, 307, 308):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                handler.redirect_request(request, io.BytesIO(b""), code, "Moved", {}, "https://evil.example/")
            # The error carries the ORIGINAL url and the 3xx status, so callers
            # map it like any other middlebox answer; nothing was re-sent.
            self.assertEqual(caught.exception.code, code)
            self.assertIn("passport.test", caught.exception.filename)

    def test_the_opener_has_the_handler_installed(self):
        handlers = [type(h).__name__ for h in transport._OPENER.handlers]
        self.assertIn("_RefuseRedirects", handlers)
        # And the permissive default is displaced, not merely joined: the
        # opener resolves one redirect handler and it must be ours.
        self.assertEqual(handlers.count("HTTPRedirectHandler"), 0)


if __name__ == "__main__":
    unittest.main()
