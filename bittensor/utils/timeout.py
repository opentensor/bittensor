import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    def __init__(self, func_name, locals):
        params = ', '.join(f"{k}={v}" for k, v in locals.items() if not k.startswith('_'))
        super().__init__(f"{func_name}() timed out with parameters: {params}")

@contextmanager
def timeout(seconds, func_name, locals):
    def raise_timeout(signum, frame):
        raise TimeoutException(func_name, locals)

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)