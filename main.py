import contextlib
import time
import threading
import uvicorn

from fastapi import FastAPI, APIRouter

# class Hello:

#     def __init__(self, name: str):
#         self.name = name
#         self.router = APIRouter()
#         self.router.add_api_route("/hello", self.hello, methods=["GET"])

#     def hello(self):
#         return {"Hello": self.name}

# app = FastAPI()
# hello = Hello("World")
# app.include_router(hello.router)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="debug" )

class Server(uvicorn.Server):
    should_exit: bool = False
    is_running: bool = False

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


    def _wrapper_run(self):
        with self.run_in_thread():
            while not self.should_exit:
                time.sleep(1e-3)

    def start(self):
        if not self.is_running:
            self.should_exit = False
            thread = threading.Thread(target=self._wrapper_run, daemon=True)
            thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.should_exit = True

config = uvicorn.Config(app, host="0.0.0.0", port=8001, reload=False, log_level="debug" )
server = Server(config=config)

server.start()
time.sleep(10)
server.stop()

# with server.run_in_thread():
#     print ('started')
#     time.sleep(10)
#     print ('stopping')
#     # server.should_exit = True
# ee = server.run_in_thread().__enter__()
# print(ee)
# print ('started')
# time.sleep(10)
# print ('stopping')
# server.should_exit = True
# server.run_in_thread().__exit__(None, None, None)

# server.should_exit = True

# print('done')

