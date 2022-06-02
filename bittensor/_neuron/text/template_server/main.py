import bittensor
if __name__ == "__main__":
    bittensor.utils.check_version()
    template = bittensor.neurons.template_server.neuron().run()
