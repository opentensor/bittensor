from threading import Thread
import bittensor
import time
from multiprocessing import Process

from bittensor._dendrite import dendrite

def benchmark1():
    config = bittensor.neurons.template_miner.neuron.config()
    config.wallet.name = 'mock'
    config.subtensor.network = 'mock'
    config.dataset._mock = True
    config.logging.debug = True
    config.neuron.n_epochs = 1
    neuron = bittensor.neurons.template_miner.neuron( config )

    d = bittensor.dataset( _mock = True )
    dendrite = bittensor.dendrite( wallet = neuron.wallet )
    graph = bittensor.metagraph( subtensor = bittensor.subtensor.mock() )

    t = Thread(target = neuron.run)
    t.start()
    while t.is_alive():
        print (graph)
        if graph.n.item() != 1:
            graph.sync()
            time.sleep(1)
            print ('waiting for subscription')
        else:
            print( graph.endpoint_objs[0] )
            resp, qtime, codes = dendrite.forward_text( endpoints = graph.endpoint_objs[0], inputs = next( d ) )
            print ( codes )

    t.join()


if __name__ == '__main__':
    benchmark1()

