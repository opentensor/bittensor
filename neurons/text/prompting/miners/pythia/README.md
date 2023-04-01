## AlephAlpha Miner
togethercomputer/Pythia-7B Language Model Serving with BitTensor
This code is for running a language model powered by togethercomputer through the BitTensor framework. 

# Example Usage
```
python3 -m pip install -r neurons/text/prompting/miners/pythia/requirements.txt
python3 neurons/text/prompting/miners/pythia/miner.py
```

# Full Usage
```
usage: miner.py [-h] [--netuid NETUID] [--config CONFIG] [--client.max_new_tokens CLIENT.MAX_NEW_TOKENS] 
                [--client.temperature CLIENT.TEMPERATURE] [--client.do_sample CLIENT.DO_SAMPLE] 
                [--neuron.name NEURON.NAME] [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH]
                [--neuron.no_set_weights NEURON.NO_SET_WEIGHTS] [--neuron.max_batch_size NEURON.MAX_BATCH_SIZE] [--neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN] [--neuron.blacklist.hotkeys NEURON.BLACKLIST.HOTKEYS]


optional arguments:
  -h, --help            show this help message and exit
  --netuid NETUID       Subnet netuid
  --config CONFIG       If set, defaults are overridden by passed file.
  --client.max_new_tokens CLIENT.MAX_NEW_TOKENS
                        The maximum number of tokens to be generated. (default: 64)
  --client.temperature  Model temperature setting (default: 0.8)
  --client.do_sample    Whether to use sampling or not (if not, uses greedy decoding). (default: false)
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