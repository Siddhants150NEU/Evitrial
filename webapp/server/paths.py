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
