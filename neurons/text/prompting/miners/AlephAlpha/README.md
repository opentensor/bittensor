## AlephAlpha Miner
AlephAlpha Language Model Serving with BitTensor
This code is for running a language model powered by AlephAlpha through the BitTensor framework. 

# Example Usage
```
python3 -m pip install -r neurons/text/prompting/miners/AlephAlpha/requirements.txt
python3 neurons/text/prompting/miners/AlephAlpha/miner.py --client.api_key <your AlephAlpha api_key>
```

# Full Usage
```
usage: miner.py [-h] [--netuid NETUID] [--config CONFIG] [--client.api_key CLIENT.API_KEY] [--client.model CLIENT.MODEL] [--client.maximum_tokens CLIENT.MAXIMUM_TOKENS] [--client.temperature CLIENT.TEMPERATURE]
                [--client.stop_sequences CLIENT.STOP_SEQUENCES] [--client.top_k CLIENT.TOP_K] [--client.top_p CLIENT.TOP_P] [--neuron.name NEURON.NAME] [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH]
                [--neuron.no_set_weights NEURON.NO_SET_WEIGHTS] [--neuron.max_batch_size NEURON.MAX_BATCH_SIZE] [--neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN] [--neuron.blacklist.hotkeys NEURON.BLACKLIST.HOTKEYS]


optional arguments:
  -h, --help            show this help message and exit
  --netuid NETUID       Subnet netuid
  --config CONFIG       If set, defaults are overridden by passed file.
  --client.api_key CLIENT.API_KEY
                        AlephAlpha API key. (required)
  --client.model CLIENT.MODEL
                        Model name to use. (default: 'luminous-base')
  --client.maximum_tokens CLIENT.MAXIMUM_TOKENS
                        The maximum number of tokens to be generated. (default: 64)
  --client.temperature CLIENT.TEMPERATURE
                        A non-negative float that tunes the degree of randomness in generation. (default: 0.0)
  --client.stop_sequences CLIENT.STOP_SEQUENCES
                        Stop tokens. (default: ['user: ', 'bot: ', 'system: '])
  --client.top_k CLIENT.TOP_K
                        Number of most likely tokens to consider at each step. (default: 0)
  --client.top_p CLIENT.TOP_P
                        Total probability mass of tokens to consider at each step. (default: 0.0)
  --neuron.name NEURON.NAME
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name (default: 'AI21_server')
  --neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH
                        Blocks until the miner sets weights on chain (default: 100)
  --neuron.no_set_weights NEURON.NO_SET_WEIGHTS
                        If True, the model does not set weights. (default: False)
  --neuron.max_batch_size NEURON.MAX_BATCH_SIZE
                        The maximum batch size for forward requests. (default: -1)
  --neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN
                        The maximum sequence length for forward requests. (default: -1)
  --neuron.blacklist.hotkeys NEURON.BLACKLIST.HOTKEYS
                        To blacklist certain hotkeys (default: [])
```