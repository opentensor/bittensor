from .ip import IP

from loguru import logger



class Neuron:
    uid: int
    hotkey : str
    ip: IP
    coldkey : str

    def __init__(self, uid, hotkey, ip, ip_type, coldkey):
        self.uid = uid
        self.hotkey = hotkey
        self.ip = IP(ip, ip_type)
        self.coldkey = coldkey
        self.stake = int

    @staticmethod
    def from_dict(attrs : dict):
        return Neuron(attrs['uid'], attrs['hotkey'], attrs['ip'], attrs['ip_type'], attrs['coldkey'])

    def __str__(self):
        return "<neuron uid: %s hotkey: %s ip: %s coldkey: %s>" % (self.uid, self.hotkey, self.ip, self.coldkey)


class Neurons(list):
    @staticmethod
    def from_list(input : list):
        output = Neurons()

        if not input:
            return output

        for row in input:
            data = row[1]  # Attributes of the neuron are stored in the second element of the list
            data['hotkey'] = row[0] # the hotkey pub key is stored in the first element
            output.append(Neuron.from_dict(data))

        return output

    def __str__(self):
        y = map(lambda x : x.__str__(), self)

        return "".join(y)





