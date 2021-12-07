
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


import threading
import time
import random
import multiprocessing as mp
import queue

class ProducerThread(threading.Thread):
    def __init__(self, queue, target=None, arg = None, name=None):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name
        self.arg = arg
        self.queue = queue 

    def run(self):
        while True:
            if not self.queue.full():
                item = self.target(*self.arg)
                print(item)
                self.queue.put(item)
                print(f"\n\nQUEUE PUT {item} \t QUEUE SIZE {self.queue.qsize()}\n\n", item)
                time.sleep(10)
        return

class ThreadQueue():
    def __init__(self, producer_target, producer_arg, buffer_size = 5):
        self.buffer_size = buffer_size
        self.queue = queue.Queue(buffer_size)
        self.producer = ProducerThread(name='producer', queue = self.queue, target = producer_target, arg = producer_arg)
        self.producer.start()