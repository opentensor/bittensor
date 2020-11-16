import bittensor
from bittensor.session import BTSession

class Neuron(object):
    """ 
    """
    def __init__(   
                self,
                config,
                session: BTSession
        ):
        self.config = config
        self.session = session

    def start(self):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError