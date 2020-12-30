import time

# A moving average that updates values based on time since last update.
class timed_rolling_avg():
    def __init__(self, initial_value, alpha):
        self.value = initial_value
        self.alpha = alpha
        self.last_update = time.time()

    def update(self, new_value):
        now = time.time()
        time_delta = now - self.last_update
        self.last_update = now
        new_value = new_value / time_delta
        self.value = (1 - self.alpha) * self.value + self.alpha * new_value