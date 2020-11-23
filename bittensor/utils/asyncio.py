import asyncio
import concurrent

class Asyncio:

    @staticmethod
    def add_task(method):
        loop = asyncio.get_event_loop()
        loop.create_task(method)

    @staticmethod
    def start_in_thread(method, args):
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=3,
        )

        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, method, args)

    @staticmethod
    def run_forever():
        loop = asyncio.get_event_loop()
        loop.run_forever()
