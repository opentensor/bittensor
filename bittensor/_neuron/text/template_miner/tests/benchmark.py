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

        if neuron.wallet.hotkey.ss58_address in graph.hotkeys:
            endpoint = graph.endpoint_objs[ graph.hotkeys.index( neuron.wallet.hotkey.ss58_address ) ]
            print ( endpoint )
            
            if endpoint.ip != '0.0.0.0':
                print ( 'registered and served ')
                resp, qtime, codes = dendrite.forward_text( endpoints = endpoint, inputs = next( d ) )
                print ( codes )

            else:
                print ( 'registered but not served')
                graph.sync()
                time.sleep(1)

        else:
            print ('not served.')
            graph.sync()
            time.sleep(1)

    t.join()


if __name__ == '__main__':
    benchmark1()

