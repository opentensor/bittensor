## Install Tortoise
Follow the instructions [here](https://github.com/neonbjb/tortoise-tts)

Basically
```bash
git clone https://github.com/neonbjb/tortoise-tts.git
cd tortoise-tts
python -m pip install -r ./requirements.txt
python setup.py install
```

Make it available anywhere
```bash
export PATH="/path/to/tortoise-tts:$PATH"
# export PATH="/home/jason/dev/tortoise-tts:$PATH"
```


## Adding custom voices
See: [tortoise](https://github.com/neonbjb/tortoise-tts/tree/main#adding-a-new-voice) Repo
TL;DR must be:
- .wav files
- 22050 sample rate


## Usage
## Tortoise Text to Speech Miner
Serves a Tortoise model for text to speech generation with conditioning latents (optional).

# Example Usage
```
python3 -m pip install -r neurons/text_to_speech/miners/tortoise/requirements.txt 
python3 neurons/text_to_speech/miners/tortoise/neuron.py --clips_path <path/to/conditioning/audio.wav>
```

# Full Usage
```
usage: neuron.py [-h] [--device DEVICE] [--default_clips_path DEFAULT_CLIPS_PATH]
                 [--netuid NETUID] [--neuron.name NEURON.NAME]
                 [--neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH]
                 [--neuron.no_set_weights] [--wallet.name WALLET.NAME]
                 [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH]
                 [--wallet._mock] [--wallet.reregister WALLET.REREGISTER]
                 [--axon.port AXON.PORT] [--axon.ip AXON.IP]
                 [--axon.external_port AXON.EXTERNAL_PORT]
                 [--axon.fast_api_port AXON.FAST_API_PORT]
                 [--axon.external_fast_api_port AXON.EXTERNAL_FAST_API_PORT]
                 [--axon.external_ip AXON.EXTERNAL_IP]
                 [--axon.max_workers AXON.MAX_WORKERS]
                 [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS]
                 [--axon.disable_fast_api] [--subtensor.network SUBTENSOR.NETWORK]
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
                 [--logging.logging_dir LOGGING.LOGGING_DIR]
                 [--neuron.blacklist.blacklisted_keys [NEURON.BLACKLIST.BLACKLISTED_KEYS [NEURON.BLACKLIST.BLACKLISTED_KEYS ...]]]
                 [--neuron.blacklist.whitelisted_keys [NEURON.BLACKLIST.WHITELISTED_KEYS [NEURON.BLACKLIST.WHITELISTED_KEYS ...]]]
                 [--neuron.blacklist.allow_non_registered]
                 [--neuron.blacklist.min_allowed_stake NEURON.BLACKLIST.MIN_ALLOWED_STAKE]
                 [--neuron.blacklist.vpermit_required]
                 [--neuron.priority.default_priority NEURON.PRIORITY.DEFAULT_PRIORITY]
                 [--neuron.priority.blacklisted_keys [NEURON.PRIORITY.BLACKLISTED_KEYS [NEURON.PRIORITY.BLACKLISTED_KEYS ...]]]
                 [--neuron.priority.whitelisted_keys [NEURON.PRIORITY.WHITELISTED_KEYS [NEURON.PRIORITY.WHITELISTED_KEYS ...]]]
                 [--config CONFIG] [--strict]

Tortoise Miner

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       Device to load model
  --default_clips_path DEFAULT_CLIPS_PATH
                        Path to WAV file(s) for latent conditioning.
  --netuid NETUID       Subnet netuid
  --neuron.name NEURON.NAME
                        Trials for this miner go in miner.root / (wallet_cold -
                        wallet_hot) / miner.name
  --neuron.blocks_per_epoch NEURON.BLOCKS_PER_EPOCH
                        Blocks until the miner sets weights on chain
  --neuron.no_set_weights
                        If True, the model does not set weights.
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
  --axon.port AXON.PORT
                        The local port this axon endpoint is bound to. i.e. 8091
  --axon.ip AXON.IP     The local ip this axon binds to. ie. [::]
  --axon.external_port AXON.EXTERNAL_PORT
                        The public port this axon broadcasts to the network. i.e. 8091
  --axon.fast_api_port AXON.FAST_API_PORT
                        The local port this axon fast api endpoint is bound to. i.e.
                        8092
  --axon.external_fast_api_port AXON.EXTERNAL_FAST_API_PORT
                        The public fast api port this axon broadcasts to the network.
                        i.e. 8092
  --axon.external_ip AXON.EXTERNAL_IP
                        The external ip this axon broadcasts to the network to. ie.
                        [::]
  --axon.max_workers AXON.MAX_WORKERS
                        The maximum number connection handler threads working
                        simultaneously on this endpoint. The grpc server distributes
                        new worker threads to service requests up to this number.
  --axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS
                        Maximum number of allowed active connections
  --axon.disable_fast_api
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
  --neuron.blacklist.blacklisted_keys [NEURON.BLACKLIST.BLACKLISTED_KEYS [NEURON.BLACKLIST.BLACKLISTED_KEYS ...]]
                        List of ss58 addresses which are always disallowed pass
                        through.
  --neuron.blacklist.whitelisted_keys [NEURON.BLACKLIST.WHITELISTED_KEYS [NEURON.BLACKLIST.WHITELISTED_KEYS ...]]
                        List of ss58 addresses which are always allowed pass through.
  --neuron.blacklist.allow_non_registered
                        If True, the miner will allow non-registered hotkeys to mine.
  --neuron.blacklist.min_allowed_stake NEURON.BLACKLIST.MIN_ALLOWED_STAKE
                        Minimum stake required to pass blacklist.
  --neuron.blacklist.vpermit_required
                        If True, the miner will require a vpermit to pass blacklist.
  --neuron.priority.default_priority NEURON.PRIORITY.DEFAULT_PRIORITY
                        Default call priority in queue.
  --neuron.priority.blacklisted_keys [NEURON.PRIORITY.BLACKLISTED_KEYS [NEURON.PRIORITY.BLACKLISTED_KEYS ...]]
                        List of ss58 addresses which are always given -math.inf
                        priority
  --neuron.priority.whitelisted_keys [NEURON.PRIORITY.WHITELISTED_KEYS [NEURON.PRIORITY.WHITELISTED_KEYS ...]]
                        List of ss58 addresses which are always given math.inf priority
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguemnts have
                        been set.
```