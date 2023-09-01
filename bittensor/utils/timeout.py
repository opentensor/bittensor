import signal
from contextlib import contextmanager


@contextmanager
def timeout(seconds):
    def raise_timeout(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
