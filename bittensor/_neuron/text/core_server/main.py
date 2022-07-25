import bittensor
if __name__ == "__main__":
    bittensor.utils.version_checking()
    template = bittensor.neurons.core_server.neuron().run()
