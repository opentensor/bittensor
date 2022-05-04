import socket
from random import randint

max_tries = 10


def get_random_unused_port():
    def port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    tries = 0
    while tries < max_tries:
        tries += 1
        port = randint(2**14, 2**16 - 1)

        if not port_in_use(port):
            return port

    raise RuntimeError(f"Tried {max_tries} random ports and could not find an open one")
