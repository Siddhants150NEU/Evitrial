"""Where things live. Deliberately dependency-free.

This exists so serveDemo and runCache can agree on the cache directory without one
importing the other — runCache pulls in runStages, which pulls in src.retrieval, which
pulls in torch. Importing that chain just to learn a directory name cost the server
3.2 seconds of startup and loaded a deep learning stack it may never use.
"""
from __future__ import annotations

from pathlib import Path

WEBAPP = Path(__file__).resolve().parent.parent
CACHE = WEBAPP / "cache"
RUNS = WEBAPP.parent / "reports" / "runs"

# Which logged eval run the Matchers tab quotes. Pinned rather than "newest", because
# newest silently changes what's on screen the moment anyone runs eval.py — and the
# numbers in a talk should be the ones you checked. Override per-request with ?runId=.
EVAL_RUN_ID = "20260803T013354Z_286af69"
