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

import asyncio
import concurrent

class Asyncio:
    loop = None

    @staticmethod
    def init():
        Asyncio.loop = asyncio.get_event_loop()

    @staticmethod
    def add_task(method):
        loop = asyncio.get_event_loop()
        loop.create_task(method)

    @staticmethod
    def stop():
        loop = asyncio.get_event_loop()
        loop.stop()

    @staticmethod
    def start_in_thread(method, args):
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=3,
        )

        Asyncio.loop.run_in_executor(executor, method, args)

    @staticmethod
    def run_forever():
        Asyncio.loop.run_forever()

    @staticmethod
    def run_in_new_loop(task):
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(task)
        loop.stop()
        return result
