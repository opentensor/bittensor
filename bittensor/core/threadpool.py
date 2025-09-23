# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements `ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_."""

__author__ = "Brian Quinlan (brian@sweetapp.com)"

import argparse
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
import weakref
from concurrent.futures import _base
from typing import Callable

from bittensor.core.config import Config
from bittensor.core.settings import BLOCKTIME
from bittensor.utils.btlogging.defines import BITTENSOR_LOGGER_NAME

# Workers are created as daemon threads. This is done to allow the interpreter
# to exit when there are still idle threads in a ThreadPoolExecutor's thread
# pool (i.e. shutdown() was not called). However, allowing workers to die with
# the interpreter has two undesirable properties:
#   - The workers would still be running during interpreter shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads finish.

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)

_threads_queues = weakref.WeakKeyDictionary()
_shutdown = False


class _WorkItem(object):
    def __init__(self, future, fn, start_time, args, kwargs):
        self.future = future
        self.fn = fn
        self.start_time = start_time
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (
            time.time() - self.start_time > BLOCKTIME
        ):
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)


NULL_ENTRY = (sys.maxsize, _WorkItem(None, None, time.time(), (), {}))


def _worker(executor_reference, work_queue, initializer, initargs):
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical("Exception in initializer:", exc_info=True)
            executor = executor_reference()
            if executor is not None:
                executor._initializer_failed()
            return
    try:
        while True:
            work_item = work_queue.get(block=True)
            priority = work_item[0]
            item = work_item[1]
            if priority == sys.maxsize:
                del item
            elif item is not None:
                item.run()
                # Delete references to object. See issue16284
                del item
                continue

            executor = executor_reference()
            # Exit if:
            #   - The interpreter is shutting down OR
            #   - The executor that owns the worker has been collected OR
            #   - The executor that owns the worker has been shutdown.
            if _shutdown or executor is None or executor._shutdown:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown = True
                # Notice other workers
                work_queue.put(NULL_ENTRY)
                return
            del executor
    except BaseException:
        logger.error("work_item", work_item)
        _base.LOGGER.critical("Exception in worker", exc_info=True)


class BrokenThreadPool(_base.BrokenExecutor):
    """
    Raised when a worker thread in a `ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_ failed initializing.
    """


class PriorityThreadPoolExecutor(_base.Executor):
    """Base threadpool executor with a priority queue."""

    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__

    def __init__(
        self,
        maxsize=-1,
        max_workers=None,
        thread_name_prefix="",
        initializer=None,
        initargs=(),
    ):
        """Initializes a new `ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_ instance.

        Parameters:
            max_workers: The maximum number of threads that can be used to execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
            initializer: An callable used to initialize worker threads.
            initargs: A tuple of arguments to pass to the initializer.
        """
        if max_workers is None:
            # Use this number because ThreadPoolExecutor is often
            # used to overlap I/O instead of CPU work.
            max_workers = (os.cpu_count() or 1) * 5
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")

        self._max_workers = max_workers
        self._work_queue = queue.PriorityQueue(maxsize=maxsize)
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = set()
        self._broken = False
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._thread_name_prefix = thread_name_prefix or (
            "ThreadPoolExecutor-%d" % self._counter()
        )
        self._initializer = initializer
        self._initargs = initargs

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments from parser"""
        prefix_str = "" if prefix is None else prefix + "."
        try:
            default_max_workers = (
                os.getenv("BT_PRIORITY_MAX_WORKERS")
                if os.getenv("BT_PRIORITY_MAX_WORKERS") is not None
                else 5
            )
            default_maxsize = (
                os.getenv("BT_PRIORITY_MAXSIZE")
                if os.getenv("BT_PRIORITY_MAXSIZE") is not None
                else 10
            )
            parser.add_argument(
                "--" + prefix_str + "priority.max_workers",
                type=int,
                help="""maximum number of threads in thread pool""",
                default=default_max_workers,
            )
            parser.add_argument(
                "--" + prefix_str + "priority.maxsize",
                type=int,
                help="""maximum size of tasks in priority queue""",
                default=default_maxsize,
            )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def config(cls) -> "Config":
        """Get config from the argument parser.

        Return: :func:`bittensor.Config` object.
        """
        parser = argparse.ArgumentParser()
        PriorityThreadPoolExecutor.add_args(parser)
        return Config(parser)

    @property
    def is_empty(self):
        return self._work_queue.empty()

    def submit(self, fn: Callable, *args, **kwargs) -> _base.Future:
        with self._shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

            priority = kwargs.get("priority", random.randint(0, 1000000))
            if priority == 0:
                priority = random.randint(1, 100)
            epsilon = random.uniform(0, 0.01) * priority
            start_time = time.time()
            if "priority" in kwargs:
                del kwargs["priority"]

            f = _base.Future()
            w = _WorkItem(f, fn, start_time, args, kwargs)
            self._work_queue.put((-float(priority + epsilon), w), block=False)
            self._adjust_thread_count()
            return f

    submit.__doc__ = _base.Executor.submit.__doc__

    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(NULL_ENTRY)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            t.daemon = True
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue

    def _initializer_failed(self):
        with self._shutdown_lock:
            self._broken = (
                "A thread initializer failed, the thread pool is not usable anymore"
            )
            # Drain work queue and mark pending futures failed
            while True:
                try:
                    work_item = self._work_queue.get_nowait()
                except queue.Empty:
                    break
                if work_item is not None:
                    work_item.future.set_exception(BrokenThreadPool(self._broken))

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown = True
            self._work_queue.put(NULL_ENTRY)

        if wait:
            for t in self._threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass

    shutdown.__doc__ = _base.Executor.shutdown.__doc__
