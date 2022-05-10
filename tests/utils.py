import socket
from random import randint
from typing import Set

max_tries = 10


def get_random_unused_port(allocated_ports: Set = set()):
    def port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    tries = 0
    while tries < max_tries:
        tries += 1
        port = randint(2**14, 2**16 - 1)

        if port not in allocated_ports and not port_in_use(port):
            allocated_ports.add(port)
            return port

    raise RuntimeError(f"Tried {max_tries} random ports and could not find an open one")
