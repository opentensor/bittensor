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
import queue
from loguru import logger

class ProducerThread(threading.Thread):
    r""" This producer thread runs in backgraound to fill the queue with the result of the target function.
    """
    def __init__(self, queue, target, arg, name=None):
        r"""Initialization.
        Args:
            queue (:obj:`queue.Queue`, `required`)
                The queue to be filled.
                
            target (:obj:`function`, `required`)
                The target function to run when the queue is not full.

            arg (:type:`tuple`, `required`)
                The arguments to be passed to the target function.

            name (:type:`str`, `optional`)
                The name of this threading object. 
        """
        super(ProducerThread,self).__init__()
        self.name = name
        self.target = target
        self.arg = arg
        self.queue = queue 
        self._stop_event = threading.Event()

    def run(self):
        r""" Work of the thread. Keep checking if the queue is full, if it is not full, run the target function to fill the queue.
        """
        while not self.stopped():
            if not self.queue.full():
                item = self.target(*self.arg, self.queue.qsize()+1 )
                self.queue.put(item)
            time.sleep(2)
        return

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

class ThreadQueue():
    r""" Manages the queue the producer thread that monitor and fills the queue.
    """
    def __init__(self, producer_target, producer_arg, buffer_size = 2):
        """ Setup the queue and start the producer thread.
        
        Args:
                
            producer_target (:obj:`function`, `required`)
                The target function to run when the queue is not full.

            producer_arg (:type:`tuple`, `required`)
                The arguments to be passed to the target function.

            buffer_size (:type:`int`, `optional`)
                The size of the queue.
        """
        self.buffer_size = buffer_size
        self.queue = queue.Queue(buffer_size)
        self.producer = ProducerThread(name='producer', queue = self.queue, target = producer_target, arg = producer_arg)
        self.producer.start()

    def __del__(self):
        self.close()

    def close(self):
        self.producer.stop()
        self.producer.join()
        logger.success('Dataset Thread Queue Closed')
