"""The one HTTP opener every module in this plugin uses.

urllib's DEFAULT opener auto-follows 301/302/303 redirects, rewriting a POST to
a GET and forwarding the request's Authorization header to the redirect target
with no same-host check (verified in urllib/request.py redirect_request: only
content-length and content-type are dropped). Every request this plugin makes
carries a live credential, either a bearer header or the refresh token in the
body, so following a redirect means handing that credential to whatever a
captive portal, misconfigured proxy, or hostile middlebox points at.

No Passport endpoint ever answers a redirect, so the honest behaviour is to
refuse: the redirect surfaces as an HTTPError carrying its 3xx status and each
caller's error mapping treats it like the middlebox answer it is.
"""

from __future__ import annotations

import urllib.error
import urllib.request


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"refusing to follow a {code} redirect", headers, fp)


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def open_request(request, timeout=None):
    """Drop-in for urllib.request.urlopen, minus redirect following."""
    return _OPENER.open(request, timeout=timeout)
