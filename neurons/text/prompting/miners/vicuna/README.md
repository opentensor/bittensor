## Vicuna Miner
Vicuna Language Model Serving with BitTensor
This code is for running the Vicuna model through the BitTensor framework.

# Overview

## Contents

- [Installing Dependencies](#installing-Dependencies)
- [Converting Weights Into Model](#converting-weights-into-model)
- [Starting Miner](#starting-miner)


# Installing Dependencies

```
python3 -m pip install -r neurons/text/prompting/miners/vicuna/requirements.txt
```

# Converting Weights Into Model
If you already have a converted checkpoint of the model, you can skip this step.

## Vicuna Weights
The [Vicuna](https://vicuna.lmsys.org/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the Vicuna weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get Vicuna weights by applying our delta. They will automatically download delta weights from our Hugging Face [account](https://huggingface.co/lmsys).

**NOTE**:
Weights v1.1 are only compatible with the latest main branch of huggingface/transformers and ``fschat >= 0.2.0``.
Please update your local packages accordingly. If you follow the above commands to do a fresh install, then you should get all the correct versions.

Depending on which conversion script was used to create the Huggingface checkpoint of Llama, you might get an error that the tokenizer can not be found when loading the tokenizer. You can then replace all AutoTokenizers command with the correct tokenizer (in the exampel "LlamaTokenizer"), using this command:
```
find /path/to/fastchat -type f -name '*.py' -exec sed -i 's/AutoTokenizer/LlamaTokenizer/g' {} +
```

### Vicuna-7B
This conversion command needs around 30 GB of CPU RAM.
If you do not have enough memory, you can create a large swap file that allows the operating system to automatically utilize the disk as virtual memory.
```bash
python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/vicuna-7b \
    --delta lmsys/vicuna-7b-delta-v1.1
```

### Vicuna-13B
This conversion command needs around 60 GB of CPU RAM.
If you do not have enough memory, you can create a large swap file that allows the operating system to automatically utilize the disk as virtual memory.
```bash
python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-13b \
    --target /output/path/to/vicuna-13b \
    --delta lmsys/vicuna-13b-delta-v1.1
```


# Starting Miner
```
python3 neurons/text/prompting/miners/vicuna/neuron.py
```

# Full Usage
```
usage: neuron.py [-h] [--vicuna.model_name VICUNA.MODEL_NAME] [--vicuna.device VICUNA.DEVICE] [--vicuna.max_new_tokens VICUNA.MAX_NEW_TOKENS]
                 [--vicuna.temperature VICUNA.TEMPERATURE] [--vicuna.do_sample] [--netuid NETUID] [--neuron.name NEURON.NAME]
                 [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH] [--neuron.no_set_weights]
                 [--neuron.max_batch_size NEURON.MAX_BATCH_SIZE] [--neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN]
                 [--neuron.blacklist.hotkeys [NEURON.BLACKLIST.HOTKEYS ...]] [--neuron.blacklist.allow_non_registered]
                 [--neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE] [--neuron.default_priority NEURON.DEFAULT_PRIORITY]
                 [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--wallet._mock]
                 [--wallet.reregister WALLET.REREGISTER] [--axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS]
                 [--axon.priority.maxsize AXON.PRIORITY.MAXSIZE] [--axon.port AXON.PORT] [--axon.ip AXON.IP]
                 [--axon.external_port AXON.EXTERNAL_PORT] [--axon.external_ip AXON.EXTERNAL_IP] [--axon.max_workers AXON.MAX_WORKERS]
                 [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS] [--subtensor.network SUBTENSOR.NETWORK]
                 [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock]
                 [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES]
                 [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL] [--subtensor.register.no_output_in_place]
                 [--subtensor.register.verbose] [--subtensor.register.cuda.use_cuda] [--subtensor.register.cuda.no_cuda]
                 [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]]
                 [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB] [--logging.debug] [--logging.trace] [--logging.record_log]
                 [--logging.logging_dir LOGGING.LOGGING_DIR] [--metagraph._mock] [--config CONFIG] [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --neoxt.model_name NEOXT.MODEL_NAME
                        Name/path of model to load of model to load
  --vicuna.device VICUNA.DEVICE
                        Device to load model
  --vicuna.max_new_tokens VICUNA.MAX_NEW_TOKENS
                        Max tokens for model output.
  --vicuna.temperature VICUNA.TEMPERATURE
                        Sampling temperature of model
  --vicuna.do_sample    Whether to use sampling or not (if not, uses greedy decoding).
  --netuid NETUID       Subnet netuid
  --neuron.name NEURON.NAME
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH
                        Blocks until the miner sets weights on chain
  --neuron.no_set_weights
                        If True, the model does not set weights.
  --neuron.max_batch_size NEURON.MAX_BATCH_SIZE
                        The maximum batch size for forward requests.
  --neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN
                        The maximum sequence length for forward requests.
  --neuron.blacklist.hotkeys [NEURON.BLACKLIST.HOTKEYS ...]
                        To blacklist certain hotkeys
  --neuron.blacklist.allow_non_registered
                        If True, the miner will allow non-registered hotkeys to mine.
  --neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE
                        Set default stake for miners.
  --neuron.default_priority NEURON.DEFAULT_PRIORITY
                        Set default priority for miners.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --wallet._mock        To turn on wallet mocking for testing purposes.
  --wallet.reregister WALLET.REREGISTER
                        Whether to reregister the wallet if it is not already registered.
  --axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS
                        maximum number of threads in thread pool
  --axon.priority.maxsize AXON.PRIORITY.MAXSIZE
                        maximum size of tasks in priority queue
  --axon.port AXON.PORT
                        The local port this axon endpoint is bound to. i.e. 8091
  --axon.ip AXON.IP     The local ip this axon binds to. ie. [::]
  --axon.external_port AXON.EXTERNAL_PORT
                        The public port this axon broadcasts to the network. i.e. 8091
  --axon.external_ip AXON.EXTERNAL_IP
                        The external ip this axon broadcasts to the network to. ie. [::]
  --axon.max_workers AXON.MAX_WORKERS
                        The maximum number connection handler threads working simultaneously on this endpoint. The grpc server distributes
                        new worker threads to service requests up to this number.
  --axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS
                        Maximum number of allowed active connections
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney (main network) -- local (local running network) --
                        mock (creates a mock connection (for testing)) If this option is set it overloads subtensor.chain_endpoint with an
                        entry point node from that network.
  --subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT
                        The subtensor endpoint flag. If set, overrides the --network flag.
  --subtensor._mock     To turn on subtensor mocking for testing purposes.
  --subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES, -n SUBTENSOR.REGISTER.NUM_PROCESSES
                        Number of processors to use for registration
  --subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, --subtensor.register.cuda.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, --cuda.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, -u SUBTENSOR.REGISTER.UPDATE_INTERVAL
                        The number of nonces to process before checking for next block during registration
  --subtensor.register.no_output_in_place, --no_output_in_place
                        Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.
  --subtensor.register.verbose
                        Whether to ouput the registration statistics verbosely.
  --subtensor.register.cuda.use_cuda, --cuda, --cuda.use_cuda
                        Set flag to use CUDA to register.
  --subtensor.register.cuda.no_cuda, --no_cuda, --cuda.no_cuda
                        Set flag to not use CUDA for registration
  --subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...], --cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]
                        Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).
  --subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB, --cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB
                        Set the number of Threads Per Block for CUDA.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --metagraph._mock     To turn on metagraph mocking for testing purposes.
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguemnts have been set.
```