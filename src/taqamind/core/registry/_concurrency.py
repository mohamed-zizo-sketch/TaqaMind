"""Thread-safe and cross-process advisory file locking for local storage and registries."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _get_thread_lock(lock_path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(lock_path)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[lock_path] = lock
        return lock


@contextmanager
def locked_path(
    lock_path: Path,
    *,
    timeout_seconds: float = 15.0,
    poll_interval: float = 0.005,
) -> Iterator[None]:
    """Acquire both an in-process threading lock and an atomic O_EXCL lock file."""
    resolved = Path(os.path.abspath(lock_path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _get_thread_lock(resolved)
    deadline = time.monotonic() + timeout_seconds

    with thread_lock:
        fd: int | None = None
        while True:
            try:
                fd = os.open(
                    str(resolved),
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                break
            except (FileExistsError, PermissionError) as exc:
                if time.monotonic() >= deadline:
                    if isinstance(exc, PermissionError):
                        raise
                    raise TimeoutError(
                        f"Timed out waiting for lock '{resolved.name}'."
                    ) from None
                time.sleep(min(poll_interval, max(0.001, deadline - time.monotonic())))
        try:
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            yield
        finally:
            os.close(fd)
            resolved.unlink(missing_ok=True)
