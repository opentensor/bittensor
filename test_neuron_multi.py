import bittensor
from multiprocessing import Process

config = bittensor.Neuron.default_config()
config.subtensor.network = 'kusanagi'
bittensor.init(config = config)
print(bittensor.neuron.metagraph.S)

# def function():
#     print (bittensor.neuron.subtensor.connect())

# p1 = Process(target=function)
# p1.start()
# p1.join()

# p2 = Process(target=function)
# p2.start()
# p2.join()
