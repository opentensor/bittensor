import threading
import _thread
import time
from contextlib import contextmanager


@contextmanager
def timeout(timeout):
    timer = threading.Timer(timeout, _thread.interrupt_main)
    timer.start()

    try:
        yield
    except KeyboardInterrupt as e:
        print(f"timeout exception {e} after {timeout} seconds")
        raise TimeoutError
    finally:
        timer.cancel()
