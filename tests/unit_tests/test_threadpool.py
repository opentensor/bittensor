"""
Unit tests for bittensor.core.threadpool module.

Tests the PriorityThreadPoolExecutor and related components.
"""

import argparse
import os
import pytest
import queue
import threading
import time
from concurrent.futures import Future
from unittest.mock import Mock, patch

from bittensor.core.threadpool import (
    PriorityThreadPoolExecutor,
    _WorkItem,
    BrokenThreadPool,
)
from bittensor.core.settings import BLOCKTIME
from bittensor.core.config import Config


class TestWorkItem:
    """Tests for _WorkItem class functionality."""

    def test_work_item_execution(self):
        """Test _WorkItem creation and execution."""
        result_value = 42
        test_fn = Mock(return_value=result_value)
        future = Future()
        work_item = _WorkItem(future, test_fn, time.time(), (1, 2), {"key": "value"})

        work_item.run()

        test_fn.assert_called_once_with(1, 2, key="value")
        assert future.result() == result_value

    def test_work_item_exception_handling(self):
        """Test _WorkItem handles exceptions correctly."""
        test_fn = Mock(side_effect=ValueError("Test error"))
        future = Future()
        work_item = _WorkItem(future, test_fn, time.time(), (), {})

        work_item.run()

        with pytest.raises(ValueError, match="Test error"):
            future.result()

    def test_work_item_cancelled_future(self):
        """Test _WorkItem skips execution if future is cancelled."""
        test_fn = Mock(return_value=42)
        future = Future()
        future.cancel()
        work_item = _WorkItem(future, test_fn, time.time(), (), {})

        work_item.run()

        test_fn.assert_not_called()

    def test_stale_task_detection(self):
        """Test _WorkItem skips execution if task is stale."""
        test_fn = Mock(return_value=42)
        future = Future()
        start_time = time.time() - BLOCKTIME - 1
        work_item = _WorkItem(future, test_fn, start_time, (), {})

        work_item.run()

        test_fn.assert_not_called()


class TestExecutorInitialization:
    """Tests for PriorityThreadPoolExecutor initialization."""

    def test_default_initialization(self):
        """Test executor creation with default parameters."""
        executor = PriorityThreadPoolExecutor()

        try:
            assert executor._max_workers == (os.cpu_count() or 1) * 5
            assert isinstance(executor._work_queue, queue.PriorityQueue)
            assert executor._broken is False
            assert executor._shutdown is False
        finally:
            executor.shutdown(wait=True)

    def test_custom_initialization(self):
        """Test executor creation with custom parameters."""
        executor = PriorityThreadPoolExecutor(
            max_workers=3, maxsize=5, thread_name_prefix="TestPool"
        )

        try:
            assert executor._max_workers == 3
            assert executor._thread_name_prefix == "TestPool"
        finally:
            executor.shutdown(wait=True)

    def test_invalid_max_workers(self):
        """Test that invalid max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be greater than 0"):
            PriorityThreadPoolExecutor(max_workers=0)

    def test_invalid_initializer(self):
        """Test that non-callable initializer raises TypeError."""
        with pytest.raises(TypeError, match="initializer must be a callable"):
            PriorityThreadPoolExecutor(initializer="not_callable")


class TestTaskSubmission:
    """Tests for task submission and execution."""

    def test_submit_basic(self):
        """Test basic task submission."""
        executor = PriorityThreadPoolExecutor(max_workers=2)

        try:
            future = executor.submit(lambda: 42)
            assert future.result(timeout=2) == 42
        finally:
            executor.shutdown(wait=True)

    def test_priority_ordering(self):
        """Test that tasks execute in priority order."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        execution_order = []
        lock = threading.Lock()

        def task(name):
            with lock:
                execution_order.append(name)
            time.sleep(0.01)

        try:
            # Submit low priority task first (will start immediately)
            executor.submit(lambda: task("low"), priority=100)
            time.sleep(0.01)
            # Then submit higher priority tasks
            executor.submit(lambda: task("high"), priority=1000)
            executor.submit(lambda: task("medium"), priority=500)

            time.sleep(0.5)

            # Verify execution order
            assert execution_order[0] == "low"
            assert execution_order[1] == "high"
            assert execution_order[2] == "medium"
        finally:
            executor.shutdown(wait=True)

    def test_task_with_args_and_kwargs(self):
        """Test task execution with args and kwargs."""
        executor = PriorityThreadPoolExecutor(max_workers=1)

        def task(a, b, c=0, d=0):
            return a + b + c + d

        try:
            future = executor.submit(task, 1, 2, c=3, d=4)
            assert future.result(timeout=2) == 10
        finally:
            executor.shutdown(wait=True)

    def test_concurrent_submissions(self):
        """Test thread safety of concurrent submit operations."""
        executor = PriorityThreadPoolExecutor(max_workers=4)

        try:
            futures = [executor.submit(lambda x=i: x * 2) for i in range(50)]
            results = [f.result(timeout=5) for f in futures]
            assert len(results) == 50
        finally:
            executor.shutdown(wait=True)


class TestExecutorShutdown:
    """Tests for executor shutdown behavior."""

    def test_graceful_shutdown(self):
        """Test graceful shutdown waits for tasks to complete."""
        executor = PriorityThreadPoolExecutor(max_workers=2)
        completed = []

        def task(value):
            time.sleep(0.1)
            completed.append(value)

        futures = [executor.submit(task, i) for i in range(3)]
        executor.shutdown(wait=True)

        assert len(completed) == 3
        assert executor._shutdown is True

    def test_immediate_shutdown(self):
        """Test immediate shutdown doesn't wait for tasks."""
        executor = PriorityThreadPoolExecutor(max_workers=1)

        executor.submit(lambda: time.sleep(2))

        start_time = time.time()
        executor.shutdown(wait=False)
        shutdown_time = time.time() - start_time

        assert shutdown_time < 0.5
        assert executor._shutdown is True

    def test_submit_after_shutdown(self):
        """Test that submitting after shutdown raises RuntimeError."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        executor.shutdown(wait=True)

        with pytest.raises(
            RuntimeError, match="cannot schedule new futures after shutdown"
        ):
            executor.submit(lambda: 42)

    def test_context_manager(self):
        """Test executor as context manager."""
        with PriorityThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(lambda: 42)
            assert future.result(timeout=2) == 42

        assert executor._shutdown is True


class TestBrokenThreadPool:
    """Tests for BrokenThreadPool exception handling."""

    def test_broken_pool_exception(self):
        """Test BrokenThreadPool exception."""
        assert issubclass(BrokenThreadPool, Exception)

        with pytest.raises(BrokenThreadPool):
            raise BrokenThreadPool("Test error")

    def test_submit_to_broken_pool(self):
        """Test submitting to broken pool raises BrokenThreadPool."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        executor._broken = "Pool is broken"

        with pytest.raises(BrokenThreadPool):
            executor.submit(lambda: 42)

        executor.shutdown(wait=False)

    def test_initializer_failure(self):
        """Test handling of failed initializers."""

        def failing_initializer():
            raise RuntimeError("Initializer failed")

        executor = PriorityThreadPoolExecutor(
            max_workers=1, initializer=failing_initializer
        )

        try:
            executor.submit(lambda: 42)
            time.sleep(0.5)

            assert executor._broken is not False

            with pytest.raises(BrokenThreadPool):
                executor.submit(lambda: 99)
        finally:
            executor.shutdown(wait=False)


class TestConfiguration:
    """Tests for configuration."""

    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ, {"BT_PRIORITY_MAX_WORKERS": "7", "BT_PRIORITY_MAXSIZE": "15"}
        ):
            parser = argparse.ArgumentParser()
            PriorityThreadPoolExecutor.add_args(parser)
            args = parser.parse_args([])

            assert getattr(args, "priority.max_workers") == 7
            assert getattr(args, "priority.maxsize") == 15

    def test_config_method(self):
        """Test config() class method."""
        config = PriorityThreadPoolExecutor.config()

        assert isinstance(config, Config)
        assert hasattr(config, "priority")
        assert hasattr(config.priority, "max_workers")
        assert hasattr(config.priority, "maxsize")
