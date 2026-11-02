from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar

_L = TypeVar("_L", bound=AbstractContextManager[Any])


class ProcessLock(Generic[_L]):
    """One lock for each process, by process ID.

    A fork copies a lock as it is at that moment, and when another thread of the parent held it,
    no thread of the child can release the copy. An ``os.register_at_fork`` hook can replace the
    copy, but a fork that runs no hooks, as uWSGI forks its workers by default, does not run it.
    The child has a new process ID in either case, so it makes its own lock.

    The first use in a process drops the locks of other processes. Unlike the prompt cache, no
    fork hook clears them, so a process forked before that first use in its parent, which then
    gets the ID of an ended process whose lock the copy still holds, uses that lock.

    Use ``with lock():``, so that the release is on the lock that the acquire took, also when a
    fork in the block, as in a signal handler, changes the process ID in between.
    """

    def __init__(self, factory: Callable[[], _L]) -> None:
        self._factory = factory
        self._locks: dict[int, _L] = {}

    def __call__(self) -> _L:
        pid = os.getpid()
        lock = self._locks.get(pid)
        if lock is None:
            # list() copies the keys at once: other threads can change the dict.
            for other in list(self._locks):
                if other != pid:
                    self._locks.pop(other, None)
            # setdefault, so that threads that get here at the same time share one lock.
            lock = self._locks.setdefault(pid, self._factory())
        return lock
