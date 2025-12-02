"""
Unit tests for bittensor.core.threadpool module.

Tests the PriorityThreadPoolExecutor and related components including:
- Executor initialization and configuration
- WorkItem creation and execution
- Priority queue ordering
- Task submission with various priorities
- Stale task detection
- Thread pool scaling
- Worker thread lifecycle
- Executor shutdown
- Error handling
- Initializer functions
- Concurrent operations
- Environment variable configuration
"""

import argparse
import os
import pytest
import queue
import sys
import threading
import time
from concurrent.futures import Future
from unittest.mock import Mock, patch, MagicMock

from bittensor.core.threadpool import (
    PriorityThreadPoolExecutor,
    _WorkItem,
    _worker,
    BrokenThreadPool,
    NULL_ENTRY,
)
from bittensor.core.settings import BLOCKTIME
from bittensor.core.config import Config


class TestWorkItem:
    """Tests for _WorkItem class functionality."""

    def test_work_item_creation_and_execution(self):
        """Test _WorkItem class creation and successful execution."""
        # Create a simple function to execute
        result_value = 42
        test_fn = Mock(return_value=result_value)
        
        # Create a future and work item
        future = Future()
        start_time = time.time()
        work_item = _WorkItem(future, test_fn, start_time, (1, 2), {"key": "value"})
        
        # Verify work item attributes
        assert work_item.future == future
        assert work_item.fn == test_fn
        assert work_item.start_time == start_time
        assert work_item.args == (1, 2)
        assert work_item.kwargs == {"key": "value"}
        
        # Run the work item
        work_item.run()
        
        # Verify execution
        test_fn.assert_called_once_with(1, 2, key="value")
        assert future.result() == result_value

    def test_work_item_exception_handling(self):
        """Test _WorkItem handles exceptions correctly."""
        # Create a function that raises an exception
        test_exception = ValueError("Test error")
        test_fn = Mock(side_effect=test_exception)
        
        # Create a future and work item
        future = Future()
        start_time = time.time()
        work_item = _WorkItem(future, test_fn, start_time, (), {})
        
        # Run the work item
        work_item.run()
        
        # Verify exception is captured in future
        with pytest.raises(ValueError, match="Test error"):
            future.result()

    def test_work_item_cancelled_future(self):
        """Test _WorkItem skips execution if future is cancelled."""
        test_fn = Mock(return_value=42)
        future = Future()
        future.cancel()
        
        start_time = time.time()
        work_item = _WorkItem(future, test_fn, start_time, (), {})
        
        # Run the work item
        work_item.run()
        
        # Verify function was not called
        test_fn.assert_not_called()

    def test_stale_task_detection(self):
        """Test _WorkItem skips execution if task is stale (older than BLOCKTIME)."""
        test_fn = Mock(return_value=42)
        future = Future()
        
        # Create a work item with old start time
        start_time = time.time() - BLOCKTIME - 1
        work_item = _WorkItem(future, test_fn, start_time, (), {})
        
        # Run the work item
        work_item.run()
        
        # Verify function was not called due to staleness
        test_fn.assert_not_called()


class TestPriorityThreadPoolExecutor:
    """Tests for PriorityThreadPoolExecutor initialization and basic operations."""

    def test_priority_thread_pool_executor_initialization(self):
        """Test executor creation with default parameters."""
        executor = PriorityThreadPoolExecutor()
        
        try:
            # Verify default initialization
            assert executor._max_workers == (os.cpu_count() or 1) * 5
            assert isinstance(executor._work_queue, queue.PriorityQueue)
            assert isinstance(executor._threads, set)
            assert executor._broken is False
            assert executor._shutdown is False
            assert executor._initializer is None
            assert executor._initargs == ()
        finally:
            executor.shutdown(wait=True)

    def test_priority_thread_pool_executor_initialization_with_params(self):
        """Test executor creation with custom parameters."""
        max_workers = 3
        maxsize = 5
        thread_name_prefix = "TestPool"
        
        executor = PriorityThreadPoolExecutor(
            max_workers=max_workers,
            maxsize=maxsize,
            thread_name_prefix=thread_name_prefix
        )
        
        try:
            assert executor._max_workers == max_workers
            assert executor._thread_name_prefix == thread_name_prefix
        finally:
            executor.shutdown(wait=True)

    def test_initialization_invalid_max_workers(self):
        """Test that invalid max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be greater than 0"):
            PriorityThreadPoolExecutor(max_workers=0)
        
        with pytest.raises(ValueError, match="max_workers must be greater than 0"):
            PriorityThreadPoolExecutor(max_workers=-1)

    def test_initialization_invalid_initializer(self):
        """Test that non-callable initializer raises TypeError."""
        with pytest.raises(TypeError, match="initializer must be a callable"):
            PriorityThreadPoolExecutor(initializer="not_callable")

    def test_empty_queue_detection(self):
        """Test is_empty property correctly reports queue state."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        try:
            # Queue should be empty initially
            assert executor.is_empty is True
            
            # Submit a task
            future = executor.submit(lambda: time.sleep(0.1))
            
            # Wait for task to complete
            future.result(timeout=2)
            
            # Queue should be empty after task completion
            time.sleep(0.1)  # Give time for queue to clear
            assert executor.is_empty is True
        finally:
            executor.shutdown(wait=True)


class TestTaskSubmission:
    """Tests for task submission with various priorities."""

    def test_submit_with_priority_levels(self):
        """Test submitting tasks with various priority levels."""
        executor = PriorityThreadPoolExecutor(max_workers=2)
        
        try:
            # Submit tasks with different priorities
            future_high = executor.submit(lambda: "high", priority=1000)
            future_medium = executor.submit(lambda: "medium", priority=500)
            future_low = executor.submit(lambda: "low", priority=100)
            
            # All tasks should complete
            assert future_high.result(timeout=2) == "high"
            assert future_medium.result(timeout=2) == "medium"
            assert future_low.result(timeout=2) == "low"
        finally:
            executor.shutdown(wait=True)

    def test_priority_queue_ordering(self):
        """Test that tasks execute in priority order (higher priority first)."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        execution_order = []
        lock = threading.Lock()
        
        def task(name):
            with lock:
                execution_order.append(name)
            time.sleep(0.01)
            return name
        
        try:
            # Submit tasks in reverse priority order
            # Higher priority values should execute first
            future_low = executor.submit(lambda: task("low"), priority=100)
            time.sleep(0.01)  # Ensure first task starts
            future_high = executor.submit(lambda: task("high"), priority=1000)
            future_medium = executor.submit(lambda: task("medium"), priority=500)
            
            # Wait for all tasks
            future_low.result(timeout=2)
            future_high.result(timeout=2)
            future_medium.result(timeout=2)
            
            # Verify execution order (first task runs immediately, then by priority)
            assert execution_order[0] == "low"  # Already running
            assert execution_order[1] == "high"  # Highest priority
            assert execution_order[2] == "medium"  # Medium priority
        finally:
            executor.shutdown(wait=True)

    def test_submit_without_priority(self):
        """Test submitting tasks without explicit priority uses random priority."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        try:
            # Submit task without priority
            future = executor.submit(lambda: 42)
            result = future.result(timeout=2)
            
            assert result == 42
        finally:
            executor.shutdown(wait=True)

    def test_priority_epsilon_randomization(self):
        """Test that priority tie-breaking uses epsilon randomization."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        try:
            # Submit multiple tasks with same priority
            futures = []
            for i in range(5):
                future = executor.submit(lambda x=i: x, priority=100)
                futures.append(future)
            
            # All tasks should complete (epsilon prevents deadlock)
            results = [f.result(timeout=2) for f in futures]
            assert len(results) == 5
        finally:
            executor.shutdown(wait=True)

    def test_concurrent_task_submission(self):
        """Test thread safety of concurrent submit operations."""
        executor = PriorityThreadPoolExecutor(max_workers=4)
        num_tasks = 50
        
        def submit_tasks(task_range):
            futures = []
            for i in task_range:
                future = executor.submit(lambda x=i: x * 2, priority=i)
                futures.append(future)
            return futures
        
        try:
            # Submit tasks from multiple threads
            threads = []
            all_futures = []
            
            for i in range(5):
                start = i * 10
                end = start + 10
                thread = threading.Thread(
                    target=lambda r=range(start, end): all_futures.extend(submit_tasks(r))
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all submission threads
            for thread in threads:
                thread.join()
            
            # Verify all tasks complete
            results = [f.result(timeout=5) for f in all_futures]
            assert len(results) == num_tasks
        finally:
            executor.shutdown(wait=True)


class TestThreadPoolScaling:
    """Tests for thread pool scaling and worker lifecycle."""

    def test_thread_pool_scaling(self):
        """Test thread creation up to max_workers."""
        max_workers = 3
        executor = PriorityThreadPoolExecutor(max_workers=max_workers)
        
        # Create a barrier to hold tasks
        barrier = threading.Barrier(max_workers + 1)
        
        def blocking_task():
            barrier.wait(timeout=5)
            return True
        
        try:
            # Submit enough tasks to require all workers
            futures = []
            for _ in range(max_workers):
                future = executor.submit(blocking_task)
                futures.append(future)
            
            # Give time for threads to be created
            time.sleep(0.2)
            
            # Verify thread count
            assert len(executor._threads) <= max_workers
            
            # Release barrier
            barrier.wait(timeout=5)
            
            # Wait for tasks to complete
            for future in futures:
                assert future.result(timeout=2) is True
        finally:
            executor.shutdown(wait=True)

    def test_worker_thread_lifecycle(self):
        """Test worker creation, execution, and termination."""
        executor = PriorityThreadPoolExecutor(max_workers=2)
        
        try:
            # Initially no threads
            assert len(executor._threads) == 0
            
            # Submit a task to create a worker
            future = executor.submit(lambda: 42)
            result = future.result(timeout=2)
            
            assert result == 42
            # At least one thread should have been created
            assert len(executor._threads) >= 1
        finally:
            executor.shutdown(wait=True)
            
            # After shutdown, threads should terminate
            time.sleep(0.5)
            for thread in executor._threads:
                assert not thread.is_alive() or thread.daemon


class TestExecutorShutdown:
    """Tests for executor shutdown behavior."""

    def test_executor_shutdown_graceful(self):
        """Test graceful shutdown waits for tasks to complete."""
        executor = PriorityThreadPoolExecutor(max_workers=2)
        completed = []
        
        def task(value):
            time.sleep(0.1)
            completed.append(value)
            return value
        
        # Submit tasks
        futures = [executor.submit(task, i) for i in range(3)]
        
        # Shutdown and wait
        executor.shutdown(wait=True)
        
        # Verify all tasks completed
        assert len(completed) == 3
        assert executor._shutdown is True

    def test_executor_shutdown_immediate(self):
        """Test immediate shutdown doesn't wait for tasks."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        def long_task():
            time.sleep(2)
            return 42
        
        # Submit a long task
        future = executor.submit(long_task)
        
        # Shutdown immediately
        start_time = time.time()
        executor.shutdown(wait=False)
        shutdown_time = time.time() - start_time
        
        # Shutdown should be quick
        assert shutdown_time < 0.5
        assert executor._shutdown is True

    def test_submit_after_shutdown(self):
        """Test that submitting after shutdown raises RuntimeError."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        executor.shutdown(wait=True)
        
        with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
            executor.submit(lambda: 42)


class TestBrokenThreadPool:
    """Tests for BrokenThreadPool exception handling."""

    def test_broken_thread_pool_exception(self):
        """Test BrokenThreadPool error handling."""
        # BrokenThreadPool should be an exception
        assert issubclass(BrokenThreadPool, Exception)
        
        # Test raising the exception
        with pytest.raises(BrokenThreadPool):
            raise BrokenThreadPool("Test error")

    def test_submit_to_broken_pool(self):
        """Test that submitting to broken pool raises BrokenThreadPool."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        # Manually break the pool
        executor._broken = "Pool is broken"
        
        with pytest.raises(BrokenThreadPool):
            executor.submit(lambda: 42)
        
        executor.shutdown(wait=False)


class TestInitializer:
    """Tests for initializer function functionality."""

    def test_initializer_function(self):
        """Test custom initializer with initargs."""
        init_called = []
        
        def initializer(value):
            init_called.append(value)
        
        executor = PriorityThreadPoolExecutor(
            max_workers=2,
            initializer=initializer,
            initargs=(42,)
        )
        
        try:
            # Submit a task to trigger worker creation
            future = executor.submit(lambda: "done")
            future.result(timeout=2)
            
            # Give time for initializer to run
            time.sleep(0.2)
            
            # Initializer should have been called at least once
            assert len(init_called) >= 1
            assert 42 in init_called
        finally:
            executor.shutdown(wait=True)

    def test_initializer_failure_handling(self):
        """Test handling of failed initializers."""
        def failing_initializer():
            raise RuntimeError("Initializer failed")
        
        executor = PriorityThreadPoolExecutor(
            max_workers=1,
            initializer=failing_initializer,
            initargs=()
        )
        
        try:
            # Submit a task
            future = executor.submit(lambda: 42)
            
            # Give time for initializer to fail
            time.sleep(0.5)
            
            # Pool should be marked as broken
            assert executor._broken is not False
            
            # New submissions should fail
            with pytest.raises(BrokenThreadPool):
                executor.submit(lambda: 99)
        finally:
            executor.shutdown(wait=False)


class TestConfiguration:
    """Tests for configuration and environment variables."""

    def test_config_from_environment_variables(self):
        """Test BT_PRIORITY_MAX_WORKERS and BT_PRIORITY_MAXSIZE environment variables."""
        with patch.dict(os.environ, {
            'BT_PRIORITY_MAX_WORKERS': '7',
            'BT_PRIORITY_MAXSIZE': '15'
        }):
            parser = argparse.ArgumentParser()
            PriorityThreadPoolExecutor.add_args(parser)
            
            # Parse with defaults from environment
            args = parser.parse_args([])
            
            # Access using getattr since args uses dot notation
            assert getattr(args, 'priority.max_workers') == 7
            assert getattr(args, 'priority.maxsize') == 15

    def test_config_without_environment_variables(self):
        """Test default configuration without environment variables."""
        # Clear environment variables
        env_backup = {}
        for key in ['BT_PRIORITY_MAX_WORKERS', 'BT_PRIORITY_MAXSIZE']:
            if key in os.environ:
                env_backup[key] = os.environ[key]
                del os.environ[key]
        
        try:
            parser = argparse.ArgumentParser()
            PriorityThreadPoolExecutor.add_args(parser)
            
            args = parser.parse_args([])
            
            # Should use defaults
            assert getattr(args, 'priority.max_workers') == 5
            assert getattr(args, 'priority.maxsize') == 10
        finally:
            # Restore environment
            for key, value in env_backup.items():
                os.environ[key] = value

    def test_add_args_parser_integration(self):
        """Test argparse argument addition."""
        parser = argparse.ArgumentParser()
        PriorityThreadPoolExecutor.add_args(parser)
        
        # Test with custom values
        args = parser.parse_args([
            '--priority.max_workers', '8',
            '--priority.maxsize', '20'
        ])
        
        assert getattr(args, 'priority.max_workers') == 8
        assert getattr(args, 'priority.maxsize') == 20

    def test_add_args_with_prefix(self):
        """Test add_args with custom prefix."""
        parser = argparse.ArgumentParser()
        PriorityThreadPoolExecutor.add_args(parser, prefix="custom")
        
        args = parser.parse_args([
            '--custom.priority.max_workers', '6',
            '--custom.priority.maxsize', '12'
        ])
        
        assert getattr(args, 'custom.priority.max_workers') == 6
        assert getattr(args, 'custom.priority.maxsize') == 12

    def test_config_method(self):
        """Test config() class method returns Config object."""
        config = PriorityThreadPoolExecutor.config()
        
        assert isinstance(config, Config)
        assert hasattr(config, 'priority')
        assert hasattr(config.priority, 'max_workers')
        assert hasattr(config.priority, 'maxsize')


class TestWorkerFunction:
    """Tests for _worker function behavior."""

    def test_worker_with_null_entry(self):
        """Test worker exits on NULL_ENTRY."""
        work_queue = queue.PriorityQueue()
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        try:
            # Mark executor as shutdown to allow worker to exit
            executor._shutdown = True
            
            # Put NULL_ENTRY to signal shutdown
            work_queue.put(NULL_ENTRY)
            
            # Create a worker thread
            import weakref
            executor_ref = weakref.ref(executor)
            thread = threading.Thread(
                target=_worker,
                args=(executor_ref, work_queue, None, ())
            )
            thread.daemon = True
            thread.start()
            
            # Wait for thread to exit
            thread.join(timeout=2)
            
            # Thread should have exited
            assert not thread.is_alive()
        finally:
            executor.shutdown(wait=False)

    def test_worker_executes_work_items(self):
        """Test worker correctly executes work items from queue."""
        work_queue = queue.PriorityQueue()
        executor = PriorityThreadPoolExecutor(max_workers=1)
        results = []
        
        def test_task():
            results.append("executed")
            return 42
        
        try:
            # Create work item
            future = Future()
            work_item = _WorkItem(future, test_task, time.time(), (), {})
            
            # Put work item in queue
            work_queue.put((1, work_item))
            
            # Put NULL_ENTRY to stop worker
            work_queue.put(NULL_ENTRY)
            
            # Create and start worker
            executor_ref = lambda: executor
            thread = threading.Thread(
                target=_worker,
                args=(executor_ref, work_queue, None, ())
            )
            thread.start()
            thread.join(timeout=2)
            
            # Verify execution
            assert future.result(timeout=1) == 42
            assert "executed" in results
        finally:
            executor.shutdown(wait=False)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_submit_with_zero_priority(self):
        """Test that priority=0 is converted to random value."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        try:
            # Submit with priority 0
            future = executor.submit(lambda: 42, priority=0)
            result = future.result(timeout=2)
            
            assert result == 42
        finally:
            executor.shutdown(wait=True)

    def test_multiple_shutdown_calls(self):
        """Test that multiple shutdown calls are safe."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        executor.shutdown(wait=True)
        # Second shutdown should not raise
        executor.shutdown(wait=True)
        
        assert executor._shutdown is True

    def test_executor_context_manager_compatibility(self):
        """Test executor can be used as context manager."""
        with PriorityThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(lambda: 42)
            result = future.result(timeout=2)
            assert result == 42
        
        # After context exit, should be shutdown
        assert executor._shutdown is True

    def test_large_number_of_tasks(self):
        """Test handling large number of tasks."""
        executor = PriorityThreadPoolExecutor(max_workers=4)
        num_tasks = 100
        
        try:
            futures = [executor.submit(lambda x=i: x, priority=i) for i in range(num_tasks)]
            results = [f.result(timeout=10) for f in futures]
            
            assert len(results) == num_tasks
            assert set(results) == set(range(num_tasks))
        finally:
            executor.shutdown(wait=True)

    def test_task_with_args_and_kwargs(self):
        """Test task execution with both args and kwargs."""
        executor = PriorityThreadPoolExecutor(max_workers=1)
        
        def task(a, b, c=0, d=0):
            return a + b + c + d
        
        try:
            future = executor.submit(task, 1, 2, c=3, d=4, priority=100)
            result = future.result(timeout=2)
            
            assert result == 10
        finally:
            executor.shutdown(wait=True)
