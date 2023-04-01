# OpenAI Bittensor Miner

This repository contains a Bittensor Miner that uses OpenAI's GPT-3.5-turbo model as its synapse. The miner connects to the Bittensor network, registers its wallet, and serves the GPT-3.5-turbo model to the network.

## Prerequisites

- Python 3.8+
- OpenAI Python API (https://github.com/openai/openai)

## Installation

1. Clone the repository
2. Install the required packages with `pip install -r requirements.txt`
3. Set your OpenAI API key in the `api_key` argument when running the script


For more configuration options related to the wallet, axon, subtensor, logging, and metagraph, please refer to the Bittensor documentation.

## Example Usage

To run the OpenAI Bittensor Miner with default settings, use the following command:

```
python3 -m pip install -r neurons/text/prompting/miners/openai/requirements.txt 
python3 neurons/text/prompting/miners/openai/miner.py --neuron.api_key <your OpenAI api_key>
```



# Full Usage
```
usage: miner.py [-h] [--api_key API_KEY] [--config CONFIG] [--neuron.model_name NEURON.MODEL_NAME] [--neuron.suffix NEURON.SUFFIX] [--neuron.max_tokens NEURON.MAX_TOKENS] [--neuron.temperature NEURON.TEMPERATURE] [--neuron.top_p NEURON.TOP_P] [--neuron.n NEURON.N] [--neuron.stream NEURON.STREAM] [--neuron.logprobs NEURON.LOGPROBS] [--neuron.echo NEURON.ECHO] [--neuron.stop NEURON.STOP] [--neuron.presence_penalty NEURON.PRESENCE_PENALTY] [--neuron.frequency_penalty NEURON.FREQUENCY_PENALTY] [--neuron.best_of NEURON.BEST_OF] [--neuron.logit_bias NEURON.LOGIT_BIAS] [--neuron.user NEURON.USER] [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--wallet._mock] [--wallet.reregister WALLET.REREGISTER] [--axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS] [--axon.priority.maxsize AXON.PRIORITY.MAXSIZE] [--axon.port AXON.PORT] [--axon.ip AXON.IP] [--axon.external_port AXON.EXTERNAL_PORT] [--axon.external_ip AXON.EXTERNAL_IP] [--axon.max_workers AXON.MAX_WORKERS] [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS] [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock] [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES] [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL] [--subtensor.register.no_output_in_place] [--subtensor.register.verbose] [--subtensor.register.cuda.use_cuda] [--subtensor.register.cuda.no_cuda] [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID ...] [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB] [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--metagraph._mock] [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --api_key API_KEY     openai api key
  --config CONFIG       If set, defaults are overridden by passed file.
  --neuron.model_name NEURON.MODEL_NAME
                        ID of the model to use.
  --neuron.suffix NEURON.SUFFIX
                        The suffix that comes after a completion of inserted text.
  --neuron.max_tokens NEURON.MAX_TOKENS
                        The maximum number of tokens to generate in the completion.
  --neuron.temperature NEURON.TEMPERATURE
                        Sampling temperature to use, between 0 and 2.
  --neuron.top_p NEURON.TOP_P
                        Nucleus sampling parameter, top_p probability mass.
  --neuron.n NEURON.N   How many completions to generate for each prompt.
  --neuron.stream NEURON.STREAM
                        Whether to stream back partial progress.
  --neuron.logprobs NEURON.LOGPROBS
                        Include the log probabilities on the logprobs most likely tokens.
  --neuron.echo NEURON.ECHO
                        Echo back the prompt in addition to the completion.
  --neuron.stop NEURON.STOP
                        Up to 4 sequences where the API will stop generating further tokens.
  --neuron.presence_penalty NEURON.PRESENCE_PENALTY
                        Penalty for tokens based on their presence in the text so far.
  --neuron.frequency_penalty NEURON.FREQUENCY_PENALTY
                        Penalty for tokens based on their frequency in the text so far.
  --neuron.best_of NEURON.BEST_OF
                        Generates best_of completions server-side and returns the 'best' one.
  --neuron.logit_bias NEURON.LOGIT_BIAS
                        Modify the likelihood of specified tokens appearing in the completion.
  --neuron.user NEURON.USER
                        A unique identifier representing your end-user.
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
                        The subtensor network flag. The likely choices are: -- finney (main network) -- local (local running network) -- mock (creates a mock connection (for testing)) If this option is set it overloads
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
  --metagraph._mock     To turn on metagraph mocking for testing purposes.
  --strict              If flagged, config will check that only exact arguemnts have been set.
```