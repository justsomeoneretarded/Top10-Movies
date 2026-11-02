"""Thread-safe reservations for optional downloads; never touch model caches."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import shutil
import threading

class DownloadBudget:
    def __init__(self, reserve_bytes=800_000_000, headroom_bytes=2_000_000_000):
        self.reserve_bytes = reserve_bytes
        self.headroom_bytes = headroom_bytes
        self.pending = 0
        self.lock = threading.Lock()

    @contextmanager
    def claim(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            accepted = shutil.disk_usage(root).free - self.pending >= self.reserve_bytes + self.headroom_bytes
            if accepted:
                self.pending += self.reserve_bytes
        try:
            yield accepted
        finally:
            if accepted:
                with self.lock:
                    self.pending -= self.reserve_bytes
