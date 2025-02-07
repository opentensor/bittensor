from bittensor import Subtensor


sub = Subtensor("test")

meta = sub.get_metagraph_info(netuid=200)
