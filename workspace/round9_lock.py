"""OS-owned exclusive batch lock; process exit releases it even after cancellation.

This is execution protection only. It does not alter frozen participant sources,
seed identities, action ordering, scoring, API settings, or sample selection.
"""
from __future__ import annotations
import os
from pathlib import Path


class ExclusiveBatch:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+b')
        try:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError('Another round9 batch owns the execution lock; do not start or retry a duplicate') from exc
        self.stream = stream
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        return False
