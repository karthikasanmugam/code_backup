#!/usr/bin/env python3
"""Static file server for the data-validation frontend with caching disabled.

Drop-in replacement for ``python -m http.server`` that adds no-store cache
headers, so edits to index.html / CSS / JS are always picked up by the browser
without a hard refresh.

Usage:
    python serve_frontend.py <port> [directory] [bind_host]
"""
from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    bind = sys.argv[3] if len(sys.argv) > 3 else "0.0.0.0"
    handler = partial(NoCacheHandler, directory=directory)
    with ThreadingHTTPServer((bind, port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
