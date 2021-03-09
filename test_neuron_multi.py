import bittensor
from multiprocessing import Process

config = bittensor.Neuron.default_config()
config.subtensor.network = 'kusanagi'

bittensor.init( config = config )
bittensor.neuron.subtensor.connect()

print ( bittensor.neuron.metagraph.S() )
# print ( bittensor.neuron.metagraph.state.index_for_uid[0] )

# def function():
#     print (bittensor.neuron.subtensor.connect())

# p1 = Process(target=function)
# p1.start()
# p1.join()

# p2 = Process(target=function)
# p2.start()
# p2.join()
