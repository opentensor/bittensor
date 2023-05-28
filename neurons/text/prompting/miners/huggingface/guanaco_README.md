# Guanaco Miner
timdettmers Guanaco Language Model Serving with BitTensor
This code is for running a language model powered by togethercomputer through the BitTensor framework. 
Reference: [code](https://github.com/artidoro/qlora)

# Example Usage
```
python3 -m pip install -r neurons/text/prompting/miners/huggingface/guanaco_requirements.txt
python neurons/text/prompting/miners/huggingface/guanaco_miner.py --guanaco.model_name timdettmers/guanaco-33b-merged --guanaco.device_map auto
```

# Full Usage
```
usage: guanaco_miner.py [-h] [--guanaco.model_name GUANACO.MODEL_NAME] [--guanaco.api_key GUANACO.API_KEY]
                        [--guanaco.device GUANACO.DEVICE] [--guanaco.max_new_tokens GUANACO.MAX_NEW_TOKENS]
                        [--guanaco.temperature GUANACO.TEMPERATURE] [--guanaco.do_sample]
                        [--guanaco.repetition_penalty GUANACO.REPETITION_PENALTY] [--guanaco.do_prompt_injection]
                        [--guanaco.system_prompt GUANACO.SYSTEM_PROMPT]
                        [--guanaco.repetition-penalty GUANACO.REPETITION_PENALTY] [--guanaco.top_p GUANACO.TOP_P]
                        [--guanaco.top_k GUANACO.TOP_K] [--guanaco.load_in_8bit GUANACO.LOAD_IN_8BIT]
                        [--guanaco.device_map GUANACO.DEVICE_MAP]
                        [--guanaco.pad_tokens GUANACO.PAD_TOKENS [GUANACO.PAD_TOKENS ...]] [--netuid NETUID]
                        [--neuron.name NEURON.NAME] [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH]
                        [--neuron.no_set_weights] [--neuron.max_batch_size NEURON.MAX_BATCH_SIZE]
                        [--neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN]
                        [--neuron.blacklist.hotkeys [NEURON.BLACKLIST.HOTKEYS [NEURON.BLACKLIST.HOTKEYS ...]]]
                        [--neuron.blacklist.allow_non_registered]
                        [--neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE]
                        [--neuron.default_priority NEURON.DEFAULT_PRIORITY] [--wallet.name WALLET.NAME]
                        [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--wallet._mock]
                        [--wallet.reregister WALLET.REREGISTER] [--axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS]
                        [--axon.priority.maxsize AXON.PRIORITY.MAXSIZE] [--axon.port AXON.PORT] [--axon.ip AXON.IP]
                        [--axon.external_port AXON.EXTERNAL_PORT] [--axon.external_ip AXON.EXTERNAL_IP]
                        [--axon.max_workers AXON.MAX_WORKERS] [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS]
                        [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT]
                        [--subtensor._mock] [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES]
                        [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL]
                        [--subtensor.register.no_output_in_place] [--subtensor.register.verbose]
                        [--subtensor.register.cuda.use_cuda] [--subtensor.register.cuda.no_cuda]
                        [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]]
                        [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB] [--logging.debug] [--logging.trace]
                        [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--config CONFIG] [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --guanaco.model_name GUANACO.MODEL_NAME
                        Name or path of model to load
  --guanaco.api_key GUANACO.API_KEY
                        huggingface api key
  --guanaco.device GUANACO.DEVICE
                        Device to load model
  --guanaco.max_new_tokens GUANACO.MAX_NEW_TOKENS
                        Max tokens for model output.
  --guanaco.temperature GUANACO.TEMPERATURE
                        Sampling temperature of model
  --guanaco.do_sample   Whether to use multinomial sampling.
  --guanaco.repetition_penalty GUANACO.REPETITION_PENALTY
                        Repetition penalty for model
  --guanaco.do_prompt_injection
                        Whether to use a custom "system" prompt instead of the one sent by bittensor.
  --guanaco.system_prompt GUANACO.SYSTEM_PROMPT
                        What prompt to replace the system prompt with
  --guanaco.repetition-penalty GUANACO.REPETITION_PENALTY
                        Repetition penalty for greedy decoding. Between 1.0 and infinity. 1.0 means no penalty. Default: 1.0
  --guanaco.top_p GUANACO.TOP_P
                        Top-p (nucleus) sampling. Defaults to 1.0 (top-k sampling). Must be between 0.0 and 1.0.
  --guanaco.top_k GUANACO.TOP_K
                        Top-k sampling. Defaults to 0 (no top-k sampling). Must be between 0 and 1000.
  --guanaco.load_in_8bit GUANACO.LOAD_IN_8BIT
                        Load model in 8 bit precision
  --guanaco.device_map GUANACO.DEVICE_MAP
                        Device map for model parallelism.
  --guanaco.pad_tokens GUANACO.PAD_TOKENS [GUANACO.PAD_TOKENS ...]
                        A list of integers separated by spaces for the pad_tokens.
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
                        If True, the miner will allow non-registered hotkeys to mine.
  --neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE
                        Set default stake for miners.
  --neuron.default_priority NEURON.DEFAULT_PRIORITY
                        Set default priority for miners.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this
                        wallet)
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
                        The maximum number connection handler threads working simultaneously on this endpoint. The grpc
                        server distributes new worker threads to service requests up to this number.
  --axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS
                        Maximum number of allowed active connections
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney (main network) -- local (local running
                        network) -- mock (creates a mock connection (for testing)) If this option is set it overloads
                        subtensor.chain_endpoint with an entry point node from that network.
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