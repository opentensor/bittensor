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
