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

from loguru import logger
import threading
from multiprocessing.managers import BaseManager

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
        

class ManagerServer(BaseManager):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connected_count = 0
        self.server = self.get_server()
        self.manager_thread = StoppableThread(target=self.server.serve_forever,daemon=True)
        self.manager_thread.start()
        self.register('add_connection_count', callable=self.add_connection_count)     
        self.register('deduct_connection_count', callable=self.deduct_connection_count)     
        
    def close(self):
        if self.server:
            self.server.stop_event.set()
            self.manager_thread.stop()
            self.manager_thread.join()
        logger.success('Manager Server Closed')
    
    def add_connection_count(self):
        self.connected_count += 1
        logger.success(f'Manager Server: Added 1 connection, total connections: {self.connected_count}')
        return

    def deduct_connection_count(self):
        self.connected_count -= 1
        logger.success(f'Manager Server: Removed 1 connection, total connections:  {self.connected_count}')

        if self.connected_count == 0:
            logger.success(f'Manager Server: No one is connecting, killing this server {self.connected_count}')
            self.close()

        return