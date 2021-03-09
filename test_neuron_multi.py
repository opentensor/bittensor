import bittensor
from multiprocessing import Process

config = bittensor.Neuron.default_config()
config.subtensor.network = 'kusanagi'
bittensor.init(config = config)
# bittensor.neuron.start()

# bittensor.neuron.metagraph.sync()

def function():
    print (bittensor.neuron.metagraph)

p1 = Process(target=function)
p1.start()
p1.join()

p2 = Process(target=function)
p2.start()
p2.join()
