import bittensor

def test_axon():
    axon = bittensor.axon()
    axon.to_wandb()

def test_dendrite():
    dendrite = bittensor.dendrite()
    dendrite.to_wandb()

def test_metagraph():
    metagraph = bittensor.metagraph()
    metagraph.to_wandb()