import bittensor
from multiprocessing import Process

config = bittensor.Neuron.default_config()
config.subtensor.network = 'kusanagi'
bittensor.init( config = config )
bittensor.neuron.metagraph.sync()

# print (bittensor.neuron.metagraph.neurons())
# bittensor.forward_text(

# )



# bittensor.neuron.subtensor.connect()

# print ( bittensor.neuron.metagraph.toString() )
# print ( bittensor.neuron.axon.toString() )
# print ( bittensor.neuron.dendrite.toString() )
# print ( bittensor.neuron.axon.fullToString() )
# print ( bittensor.neuron.dendrite.fullToString() )


# # print ( bittensor.neuron.axon.toTensorboard(None, None) )
# # print ( bittensor.neuron.metagraph.state.index_for_uid[0] )

# def function():
#     print ( bittensor.neuron.metagraph.toString() )
#     print ( bittensor.neuron.axon.toString() )
#     print ( bittensor.neuron.dendrite.toString() )
#     print ( bittensor.neuron.axon.fullToString() )
#     print ( bittensor.neuron.dendrite.fullToString() )

# p1 = Process(target=function)
# p1.start()
# p1.join()

# p2 = Process(target=function)
# p2.start()
# p2.join()
