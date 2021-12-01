""" A exponential moving average that updates values based on time since last update.
"""
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

class timed_rolling_avg():
    """ A exponential moving average that updates values based on time since last update.
    """
    def __init__(self, initial_value, alpha):
        self.value = initial_value
        self.alpha = alpha
        self.last_update = time.time()

    def update(self, new_value):
        """ Update self.value (the moving average) with the new_value 
        """
        now = time.time()
        time_delta = now - self.last_update
        self.last_update = now
        new_value = new_value / time_delta
        self.value = (1 - self.alpha) * self.value + self.alpha * new_value

class AmountPerSecondRollingAverage():
    """ A exponential moving average that counts quantity per second.
    """
    def __init__(self, initial_value=0, alpha=0.1):
        self.value = initial_value
        self.alpha = alpha
        self.last_update = None

    def event(self, amount):
        """ Update self.value (the moving average) with the new_value 
        """
        if self.last_update == None:
            self.last_update = time.time()
        else:
            now = time.time()
            time_delta = now - self.last_update
            self.last_update = now
            new_value = amount / time_delta
            self.value = (1 - self.alpha) * self.value + self.alpha * new_value

    def get(self) -> float:
        return float(self.value)

        
class EventsPerSecondRollingAverage():
    """ A exponential moving average that counts the number of events per second.
    """
    def __init__(self, initial_value, alpha):
        self.value = initial_value
        self.alpha = alpha
        self.last_update = None

    def event(self):
        """ Update self.value (the moving average) with the new_value 
        """
        if self.last_update == None:
            self.last_update = time.time()
        else:
            now = time.time()
            time_delta = now - self.last_update
            self.last_update = now
            new_value = 1 / time_delta
            self.value = (1 - self.alpha) * self.value + self.alpha * new_value
    
    def get(self) -> float:
        return float(self.value)


class Running_Average(object):
    def __init__(self, buffer_size=10):
        """
        Running average object which takes an average over the last `buffer_size` items

        This object allows the efficient calculation of the average of the last
        `buffer_size` numbers added to it.

        Examples
        --------
        >>> a = Running_Average(2)
        >>> a.add(1)
        >>> a.get()
        1.0
        >>> a.add(1)  # there are two 1 in buffer
        >>> a.get()
        1.0
        >>> a.add(2)  # there's a 1 and a 2 in the buffer
        >>> a.get()
        1.5
        >>> a.add(2)
        >>> a.get()  # now there's only two 2 in the buffer
        2.0
        """
        self._buffer_size = int(buffer_size)  # make sure it's an int
        self.reset()

    def add(self, new):
        """
        Add a new number to the buffer, or replaces the oldest one there.
        """
        new = float(new)  # make sure it's a float
        n = len(self._buffer)
        if n < self.buffer_size:  # still have to had numbers to the buffer.
            self._buffer.append(new)
            if self._average != self._average:  # ~ if isNaN().
                self._average = new  # no previous numbers, so it's new.
            else:
                self._average *= n  # so it's only the sum of numbers.
                self._average += new  # add new number.
                self._average /= (n+1)  # divide by new number of numbers.
        else:  # buffer full, replace oldest value.
            old = self._buffer[self._index]  # the previous oldest number.
            self._buffer[self._index] = new  # replace with new one.
            self._index += 1  # update the index and make sure it's...
            self._index %= self.buffer_size  # ... smaller than buffer_size.
            self._average -= old/self.buffer_size  # remove old one...
            self._average += new/self.buffer_size  # ...and add new one...
            # ... weighted by the number of elements.

    def __call__(self):
        """
        Return the moving average value, for the lazy ones who don't want
        to write .get .
        """
        return self._average

    def get(self):
        """
        Return the moving average value.
        """
        return self()

    def reset(self):
        """
        Reset the moving average.

        If for some reason you don't want to just create a new one.
        """
        self._buffer = []  # could use np.empty(self.buffer_size)...
        self._index = 0  # and use this to keep track of how many numbers.
        self._average = float('nan')  # could use np.NaN .

    def get_buffer_size(self):
        """
        Return current buffer_size.
        """
        return self._buffer_size

    def set_buffer_size(self, buffer_size):
        """
        >>> a = Running_Average(10)
        >>> for i in range(15):
        ...     a.add(i)
        ...
        >>> a()
        9.5
        >>> a._buffer  # should not access this!!
        [10.0, 11.0, 12.0, 13.0, 14.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        Decreasing buffer size:
        >>> a.buffer_size = 6
        >>> a._buffer  # should not access this!!
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        >>> a.buffer_size = 2
        >>> a._buffer
        [13.0, 14.0]

        Increasing buffer size:
        >>> a.buffer_size = 5
        Warning: no older data available!
        >>> a._buffer
        [13.0, 14.0]

        Keeping buffer size:
        >>> a = Running_Average(10)
        >>> for i in range(15):
        ...     a.add(i)
        ...
        >>> a()
        9.5
        >>> a._buffer  # should not access this!!
        [10.0, 11.0, 12.0, 13.0, 14.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        >>> a.buffer_size = 10  # reorders buffer!
        >>> a._buffer
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        """
        buffer_size = int(buffer_size)
        # order the buffer so index is zero again:
        new_buffer = self._buffer[self._index:]
        new_buffer.extend(self._buffer[:self._index])
        self._index = 0
        if self._buffer_size < buffer_size:
            print('Warning: no older data available!')  # should use Warnings!
        else:
            diff = self._buffer_size - buffer_size
            print(diff)
            new_buffer = new_buffer[diff:]
        self._buffer_size = buffer_size
        self._buffer = new_buffer

    buffer_size = property(get_buffer_size, set_buffer_size)
