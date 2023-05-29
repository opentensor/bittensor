# Raven RWKV Miner
BlinkDL/Raven-RWKV-7B Language Model Serving with BitTensor
This code is for running a language model powered by BlinkDL through the BitTensor framework. 

## Setup
Go to the huggingface repo for more information: [rwkv-4-raven](https://huggingface.co/BlinkDL/rwkv-4-raven)

NOTE: You need to pass the path to the tokenizer.json from the command line.
- Find it [here](https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B/resolve/main/20B_tokenizer.json)

NOTE: You will want to browse and see what Raven model you wish to load [here](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main)
e.g. `RWKV-4-Raven-7B-v11-Eng99%25-Other1%25-20230427-ctx8192` for Engligh 99% and Other languages 1%=
e.g. `RWKV-4-Raven-7B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230430-ctx8192` for 49% English, 49% Chinese, 1% Japanese

These percentages refer to the amount of training data from that particular language.

# Usage
```
wget https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B/resolve/main/20B_tokenizer.json
python3 -m pip install -r neurons/text/prompting/miners/huggingface/raven-rwkv/requirements.txt
python3 neurons/text/prompting/miners/huggingface/raven-rwkv/neuron.py --raven.tokenizer_path /home/jason/bittensor/20B_tokenizer.json \
 --raven.model_name RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192 \
 --raven.repetition-penalty 0.2 --raven.top_p 0.0 --raven.temperature 1.0
```

# Full Usage
```
usage: neuron.py [-h] [--raven.model_name RAVEN.MODEL_NAME] [--raven.repo_id RAVEN.REPO_ID] [--raven.tokenizer_path RAVEN.TOKENIZER_PATH] [--raven.device RAVEN.DEVICE] [--raven.ctx_limit RAVEN.CTX_LIMIT] [--raven.max_new_tokens RAVEN.MAX_NEW_TOKENS]
                 [--raven.temperature RAVEN.TEMPERATURE] [--raven.top_p RAVEN.TOP_P] [--raven.do_prompt_injection] [--raven.system_prompt RAVEN.SYSTEM_PROMPT] [--raven.jit_on] [--raven.cuda_on] [--raven.strategy RAVEN.STRATEGY]
                 [--raven.pad_tokens RAVEN.PAD_TOKENS [RAVEN.PAD_TOKENS ...]] [--raven.repetition_penalty RAVEN.REPETITION_PENALTY] [--netuid NETUID] [--neuron.name NEURON.NAME] [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH] [--neuron.no_set_weights]
                 [--neuron.max_batch_size NEURON.MAX_BATCH_SIZE] [--neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN] [--neuron.blacklist.hotkeys [NEURON.BLACKLIST.HOTKEYS [NEURON.BLACKLIST.HOTKEYS ...]]] [--neuron.blacklist.allow_non_registered]
                 [--neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE] [--neuron.blacklist.vpermit_required] [--neuron.default_priority NEURON.DEFAULT_PRIORITY] [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--wallet._mock]
                 [--wallet.reregister WALLET.REREGISTER] [--axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS] [--axon.priority.maxsize AXON.PRIORITY.MAXSIZE] [--axon.port AXON.PORT] [--axon.ip AXON.IP] [--axon.external_port AXON.EXTERNAL_PORT]
                 [--axon.external_ip AXON.EXTERNAL_IP] [--axon.max_workers AXON.MAX_WORKERS] [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS] [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock]
                 [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES] [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL] [--subtensor.register.no_output_in_place] [--subtensor.register.verbose] [--subtensor.register.cuda.use_cuda]
                 [--subtensor.register.cuda.no_cuda] [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]] [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB] [--logging.debug] [--logging.trace] [--logging.record_log]
                 [--logging.logging_dir LOGGING.LOGGING_DIR] [--config CONFIG] [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --raven.model_name RAVEN.MODEL_NAME
                        Name/path of model to load
  --raven.repo_id RAVEN.REPO_ID
                        Repo id of model to load
  --raven.tokenizer_path RAVEN.TOKENIZER_PATH
                        Path to tokenizer json file
  --raven.device RAVEN.DEVICE
                        Device to load model
  --raven.ctx_limit RAVEN.CTX_LIMIT
                        Max context length for model input.
  --raven.max_new_tokens RAVEN.MAX_NEW_TOKENS
                        Max tokens for model output.
  --raven.temperature RAVEN.TEMPERATURE
                        Sampling temperature of model
  --raven.top_p RAVEN.TOP_P
                        Top p sampling of model
  --raven.do_prompt_injection
                        Whether to use a custom "system" prompt instead of the one sent by bittensor.
  --raven.system_prompt RAVEN.SYSTEM_PROMPT
                        What prompt to replace the system prompt with
  --raven.jit_on        Whether to use Just-In-Time complication (JIT)
  --raven.cuda_on       Whether to use CUDA kernel for seq mode (much faster). [Requires CUDA_HOME env_variable to be set]
  --raven.strategy RAVEN.STRATEGY
                        Strategy to use for RWKV model
  --raven.pad_tokens RAVEN.PAD_TOKENS [RAVEN.PAD_TOKENS ...]
                        A list of integers separated by spaces for the pad_tokens.
  --raven.repetition_penalty RAVEN.REPETITION_PENALTY
                        Repetition penalty for RWKV model
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
  --neuron.blacklist.hotkeys [NEURON.BLACKLIST.HOTKEYS [NEURON.BLACKLIST.HOTKEYS ...]]
                        To blacklist certain hotkeys
  --neuron.blacklist.allow_non_registered
                        If True, this miner will allow non-registered hotkeys to query it.
  --neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE
                        Set default stake for miners.
  --neuron.blacklist.vpermit_required
                        Require vpermit to query this miner.
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
                        The maximum number connection handler threads working simultaneously on this endpoint. The grpc server distributes new worker threads to service requests up to this number.
  --axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS
                        Maximum number of allowed active connections
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney (main network) -- local (local running network) -- mock (creates a mock connection (for testing)) If this option is set it overloads subtensor.chain_endpoint with an entry point node from that
                        network.
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
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguemnts have been set.
  ```