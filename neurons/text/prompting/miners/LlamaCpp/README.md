# LlamaCpp Miner
LlamaCpp (via Langchain) completion miner for bittensor's prompting network. 


## Installation

### Install llama.cpp (after commit: b9fd7ee [May 12, 2023])
Go to the [llama.cpp](https://github.com/ggerganov/llama.cpp) github and follow the instructions there for your platform.


### Get model weights
We can do this from huggingface (or wherever you please).

#### Install git-lfs
Follow the instructions here for your platform.
https://git-lfs.com/


#### Grab the GGML weights
```bash
git lfs install
git clone https://huggingface.co/TheBloke/dromedary-65B-lora-GGML
```


Or, to use any model you like:
```bash
git clone https://huggingface.co/<repo_author>/<repo_name>
```

Install dependencies
```bash
python -m pip install langchain>=0.0.172 llama-cpp-python
```


# Example Usage
```
python3 neurons/text/prompting/miners/LlamaCpp/neuron.py --llama.model_path
```

# Full Usage
```
usage: neuron.py [-h] --llama.model_path LLAMA.MODEL_PATH
                 [--llama.lora_base LLAMA.LORA_BASE]
                 [--llama.lora_path LLAMA.LORA_PATH] [--llama.n_ctx LLAMA.N_CTX]
                 [--llama.n_parts LLAMA.N_PARTS] [--llama.seed LLAMA.SEED]
                 [--llama.f16_kv] [--llama.logits_all] [--llama.vocab_only]
                 [--llama.use_mlock] [--llama.n_threads LLAMA.N_THREADS]
                 [--llama.n_batch LLAMA.N_BATCH] [--llama.suffix LLAMA.SUFFIX]
                 [--llama.max_tokens LLAMA.MAX_TOKENS]
                 [--llama.temperature LLAMA.TEMPERATURE] [--llama.top_p LLAMA.TOP_P]
                 [--llama.logprobs LLAMA.LOGPROBS] [--llama.echo]
                 [--llama.stop LLAMA.STOP [LLAMA.STOP ...]]
                 [--llama.repeat_penalty LLAMA.REPEAT_PENALTY]
                 [--llama.top_k LLAMA.TOP_K]
                 [--llama.last_n_tokens_size LLAMA.LAST_N_TOKENS_SIZE]
                 [--llama.use_mmap] [--llama.streaming] [--llama.verbose]
                 [--llama.do_prompt_injection]
                 [--llama.system_prompt LLAMA.SYSTEM_PROMPT] [--netuid NETUID]
                 [--neuron.name NEURON.NAME]
                 [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH]
                 [--neuron.no_set_weights]
                 [--neuron.max_batch_size NEURON.MAX_BATCH_SIZE]
                 [--neuron.max_sequence_len NEURON.MAX_SEQUENCE_LEN]
                 [--neuron.blacklist.hotkeys [NEURON.BLACKLIST.HOTKEYS [NEURON.BLACKLIST.HOTKEYS ...]]]
                 [--neuron.blacklist.allow_non_registered]
                 [--neuron.blacklist.default_stake NEURON.BLACKLIST.DEFAULT_STAKE]
                 [--neuron.default_priority NEURON.DEFAULT_PRIORITY]
                 [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY]
                 [--wallet.path WALLET.PATH] [--wallet._mock]
                 [--wallet.reregister WALLET.REREGISTER]
                 [--axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS]
                 [--axon.priority.maxsize AXON.PRIORITY.MAXSIZE]
                 [--axon.port AXON.PORT] [--axon.ip AXON.IP]
                 [--axon.external_port AXON.EXTERNAL_PORT]
                 [--axon.external_ip AXON.EXTERNAL_IP]
                 [--axon.max_workers AXON.MAX_WORKERS]
                 [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS]
                 [--subtensor.network SUBTENSOR.NETWORK]
                 [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT]
                 [--subtensor._mock]
                 [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES]
                 [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL]
                 [--subtensor.register.no_output_in_place]
                 [--subtensor.register.verbose] [--subtensor.register.cuda.use_cuda]
                 [--subtensor.register.cuda.no_cuda]
                 [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]]
                 [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB]
                 [--logging.debug] [--logging.trace] [--logging.record_log]
                 [--logging.logging_dir LOGGING.LOGGING_DIR] [--config CONFIG]
                 [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --llama.model_path LLAMA.MODEL_PATH
                        Path of LlamaCpp model to load
  --llama.lora_base LLAMA.LORA_BASE
                        Path to the Llama LoRA base model.
  --llama.lora_path LLAMA.LORA_PATH
                        Path to the Llama LoRA.
  --llama.n_ctx LLAMA.N_CTX
                        Token context window.
  --llama.n_parts LLAMA.N_PARTS
                        Number of parts to split the model into.
  --llama.seed LLAMA.SEED
                        Seed for model.
  --llama.f16_kv        Use half-precision for key/value cache.
  --llama.logits_all    Return logits for all tokens.
  --llama.vocab_only    Only load the vocabulary, no weights.
  --llama.use_mlock     Force system to keep model in RAM.
  --llama.n_threads LLAMA.N_THREADS
                        Number of threads to use.
  --llama.n_batch LLAMA.N_BATCH
                        Number of tokens to process in parallel.
  --llama.suffix LLAMA.SUFFIX
                        A suffix to append to the generated text.
  --llama.max_tokens LLAMA.MAX_TOKENS
                        The maximum number of tokens to generate.
  --llama.temperature LLAMA.TEMPERATURE
                        The temperature to use for sampling.
  --llama.top_p LLAMA.TOP_P
                        The top-p value to use for sampling.
  --llama.logprobs LLAMA.LOGPROBS
                        The number of logprobs to return.
  --llama.echo          Whether to echo the prompt.
  --llama.stop LLAMA.STOP [LLAMA.STOP ...]
                        A list of strings to stop generation when encountered.
  --llama.repeat_penalty LLAMA.REPEAT_PENALTY
                        The penalty to apply to repeated tokens.
  --llama.top_k LLAMA.TOP_K
                        The top-k value to use for sampling.
  --llama.last_n_tokens_size LLAMA.LAST_N_TOKENS_SIZE
                        The number of tokens to look back when applying the
                        repeat_penalty.
  --llama.use_mmap      Whether to keep the model loaded in RAM.
  --llama.streaming     Whether to stream the results, token by token.
  --llama.verbose       Verbose output for LlamaCpp model.
  --llama.do_prompt_injection
                        Whether to use a custom "system" prompt instead of the one sent
                        by bittensor.
  --llama.system_prompt LLAMA.SYSTEM_PROMPT
                        What prompt to replace the system prompt with
  --netuid NETUID       Subnet netuid
  --neuron.name NEURON.NAME
                        Trials for this miner go in miner.root / (wallet_cold -
                        wallet_hot) / miner.name
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
                        The name of the wallet to unlock for running bittensor (name
                        mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --wallet._mock        To turn on wallet mocking for testing purposes.
  --wallet.reregister WALLET.REREGISTER
                        Whether to reregister the wallet if it is not already
                        registered.
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
                        The external ip this axon broadcasts to the network to. ie.
                        [::]
  --axon.max_workers AXON.MAX_WORKERS
                        The maximum number connection handler threads working
                        simultaneously on this endpoint. The grpc server distributes
                        new worker threads to service requests up to this number.
  --axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS
                        Maximum number of allowed active connections
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney
                        (main network) -- local (local running network) -- mock
                        (creates a mock connection (for testing)) If this option is set
                        it overloads subtensor.chain_endpoint with an entry point node
                        from that network.
  --subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT
                        The subtensor endpoint flag. If set, overrides the --network
                        flag.
  --subtensor._mock     To turn on subtensor mocking for testing purposes.
  --subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES, -n SUBTENSOR.REGISTER.NUM_PROCESSES
                        Number of processors to use for registration
  --subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, --subtensor.register.cuda.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, --cuda.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, -u SUBTENSOR.REGISTER.UPDATE_INTERVAL
                        The number of nonces to process before checking for next block
                        during registration
  --subtensor.register.no_output_in_place, --no_output_in_place
                        Whether to not ouput the registration statistics in-place. Set
                        flag to disable output in-place.
  --subtensor.register.verbose
                        Whether to ouput the registration statistics verbosely.
  --subtensor.register.cuda.use_cuda, --cuda, --cuda.use_cuda
                        Set flag to use CUDA to register.
  --subtensor.register.cuda.no_cuda, --no_cuda, --cuda.no_cuda
                        Set flag to not use CUDA for registration
  --subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...], --cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]
                        Set the CUDA device id(s). Goes by the order of speed. (i.e. 0
                        is the fastest).
  --subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB, --cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB
                        Set the number of Threads Per Block for CUDA.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguemnts have
                        been set.
```