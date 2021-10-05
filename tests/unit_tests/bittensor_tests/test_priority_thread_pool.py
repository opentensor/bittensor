import bittensor
from unittest.mock import MagicMock

priority_pool = bittensor.prioritythreadpool(max_workers=1)

def test_priority_thread_pool():
    save = []
    def save_number(number,save):
        save += [number]
    with priority_pool:
        for x in range(10):
            priority_pool.submit(save_number, x,save,priority=x)
    
    assert save[0] == 0
    assert save[1] == 9


if __name__ == "__main__":
    test_priority_thread_pool()