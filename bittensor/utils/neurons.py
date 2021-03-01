
from loguru import logger
import bittensor.utils.networking as net

class Neuron:
    uid: int
    hotkey : str
    ip: str
    ip_type: int
    modality: int
    coldkey : str

    def __init__(self, uid, hotkey, ip, ip_type, modality, coldkey):
        self.uid = uid
        self.hotkey = hotkey
        self.ip = net.int_to_ip (ip)
        self.ip_type = ip_type
        self.coldkey = coldkey
        self.modality = modality
        self.stake = int

    @staticmethod
    def from_dict(attrs : dict):
        return Neuron(attrs['uid'], attrs['hotkey'], attrs['ip'], attrs['ip_type'], attrs['modality'], attrs['coldkey'])

    def __str__(self):
        return "<neuron uid: %s hotkey: %s ip: %s  modality: %s coldkey: %s>" % (self.uid, self.hotkey, net.ip__str__(self.ip_type, self.ip), self.modality, self.coldkey)

class Neurons(list):
    @staticmethod
    def from_list(input : list):
        output = Neurons()

        if not input:
            return output

        for row in input:
            data = row[1]  # Attributes of the neuron are stored in the second element of the list
            output.append(Neuron.from_dict(data))

        return output

    def has_uid(self,uid):
        neurons = filter(lambda x: x.uid == uid, self)
        return len(list(neurons)) > 0

    def get_by_uid(self, uid):
        neurons = Neurons(filter(lambda x: x.uid == uid, self))
        return None if len(neurons) == 0 else neurons[0]

    def __str__(self):
        y = map(lambda x : x.__str__(), self)
        return "".join(y)





