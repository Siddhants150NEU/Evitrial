from __future__ import annotations

import contextlib

try:
    from langfuse import Langfuse

    _client = Langfuse()
except Exception:
    _client = None

@contextlib.contextmanager
def span(name: str):
    if _client is None:
        yield None
        return

    s = None
    try:
        s = _client.span(name=name)
    except Exception:
        s = None
    try:
        yield s
    finally:
        try:
            if s is not None:
                s.end()
        except Exception:
            pass
