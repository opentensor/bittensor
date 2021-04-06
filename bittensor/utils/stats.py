# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

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