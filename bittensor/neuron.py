import bittensor

class Neuron(object):
    """ 
    """
    def __init__(   
                self,
                config: bittensor.Config,
                session: bittensor.BTSession
        ):
        self.config = config
        self.session = session

    def start(self):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError