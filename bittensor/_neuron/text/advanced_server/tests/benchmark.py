import bittensor
from multiprocessing import Process

def benchmark1():
    d = bittensor.dataset( _mock = True )
    config = bittensor.neurons.advanced_server.neuron.config()
    config.wallet.name = 'mock'
    config.subtensor.network = 'mock'
    config.dataset.network = 'mock'
    config.logging.debug = True
    neuron = bittensor.neurons.advanced_server.neuron( config )

    p = Process(target = neuron.run)
    p.start()
    print ('started')
    p.join()


if __name__ == '__main__':
    benchmark1()

