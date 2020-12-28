import socket
import struct

class IP:
    ip: int
    ip_type: int

    def __init__(self, ip, ip_type):
        self.ip = ip
        self.ip_type = ip_type

    def __str__(self):
        return "/ipv%i/%s" % (self.ip_type, IP.int2ip(self.ip))

    @staticmethod
    def ip2int(addr):
        return struct.unpack("!I", socket.inet_aton(addr))[0]

    @staticmethod
    def int2ip(addr):
        return socket.inet_ntoa(struct.pack("!I", addr))