import bittensor
from bittensor.session import BTSession

class Neuron(object):
    """ 
    """
    def __init__(   
                self,
                config,
        ):
        self.config = config

    def start(self, session: BTSession):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError