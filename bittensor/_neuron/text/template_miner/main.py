import bittensor
if __name__ == "__main__":

    neuron = bittensor.neurons.template_miner.neuron()
    with neuron:
        neuron.run()