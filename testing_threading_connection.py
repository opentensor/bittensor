import threading
import websocket
import queue
import time


class WebSocketManager:
    def __init__(self, url):
        self.url = url
        self.ws = websocket.WebSocket()
        self.ws.connect(
            self.url,
            **{
                "max_size": 2**32,
                "write_limit": 2**16,
            },
        )
        self.lock = threading.Lock()
        self.response_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.listener_thread = threading.Thread(target=self._listener, daemon=True)
        self.listener_thread.start()

    def _listener(self):
        """Поток для получения данных из веб-сокета."""
        while not self.stop_event.is_set():
            try:
                message = self.ws.recv()
                self.response_queue.put(message)
            except Exception as e:
                print(f"Error in listener thread: {e}")
                break

    def send(self, message):
        """Отправка сообщения через веб-сокет."""
        with self.lock:
            self.ws.send(message)

    def recv(self, timeout=None):
        """Получение сообщения из очереди."""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        """Закрытие веб-сокета и остановка потоков."""
        self.stop_event.set()
        with self.lock:
            self.ws.close()


# Пример использования
def worker(ws_manager: WebSocketManager, message, worker_id):
    ws_manager.send(message)
    response = ws_manager.recv(timeout=10)
    print(f"Worker {worker_id} received: {response}")


def main():
    # Создаем WebSocketManager
    ws_manager = WebSocketManager("wss://entrypoint-finney.opentensor.ai:443")

    # Запускаем несколько потоков
    threads = []
    for i in range(50):
        t = threading.Thread(target=worker, args=(ws_manager, f"Message {i}", i))
        threads.append(t)
        t.start()

    # Ждем завершения потоков
    for t in threads:
        t.join()

    # Закрываем соединение
    ws_manager.close()


if __name__ == "__main__":
    main()
