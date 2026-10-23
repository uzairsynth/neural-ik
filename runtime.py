"""Keep plotting caches in a writable, repo-local directory."""
import os
from pathlib import Path


def configure_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
