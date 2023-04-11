# Constitution Validator
This repository the constitution validator which can be programmed with three prompts which define:
  1. The way in which questions are asked about the netowork --default_question_prompt i.e. ask questions about chemistry
  2. The way in which questions are completed in the network --default_completion_prompt i.e. answer questions with extreme detail.
  3. The way in which questions are evaluated in the network --default_evaluation_prompt i.e. questions should have lots of detail.

## Prerequisites
- Python 3.8+
- Bittensor

## Installation
1. Clone the repository
2. Install the required packages with `pip install -r neurons/text/prompting/validators/constitution/requirements.txt`

## Example Usage
To run the Constitution Validator with default settings, use the following command:

```
python3 -m pip install -r neurons/text/prompting/validators/constitution/requirements.txt 
python3 neurons/text/prompting/validators/constitution/neuron.py
```

# Full Usage
```
usage: neuron.py [-h] [--netuid NETUID] [--neuron.name NEURON.NAME] [--neuron.reward_model_name NEURON.REWARD_MODEL_NAME] [--neuron.inference_topk NEURON.INFERENCE_TOPK] [--neuron.training_topk NEURON.TRAINING_TOPK]
                    [--prompting.model_name PROMPTING.MODEL_NAME] [--prompting.min_tokens PROMPTING.MIN_TOKENS] [--prompting.max_tokens PROMPTING.MAX_TOKENS] [--prompting.temperature PROMPTING.TEMPERATURE]
                    [--prompting.top_p PROMPTING.TOP_P] [--prompting.logprobs PROMPTING.LOGPROBS] [--prompting.repetition_penalty PROMPTING.REPETITION_PENALTY] [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY]
                    [--wallet.path WALLET.PATH] [--wallet._mock] [--wallet.reregister WALLET.REREGISTER] [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock]
                    [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES] [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL] [--subtensor.register.no_output_in_place] [--subtensor.register.verbose]
                    [--subtensor.register.cuda.use_cuda] [--subtensor.register.cuda.no_cuda] [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]]
                    [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB] [--metagraph._mock] [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--config CONFIG] [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --netuid NETUID       Prompting network netuid
  --neuron.name NEURON.NAME
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --neuron.reward_model_name NEURON.REWARD_MODEL_NAME
                        GPTRewardModel name
  --neuron.inference_topk NEURON.INFERENCE_TOPK
                        At inference time, how many miners to we query and return the top rewarded.
  --neuron.training_topk NEURON.TRAINING_TOPK
                        During training time, how many miners to we query for each batch based on scores from gating network.
  --prompting.model_name PROMPTING.MODEL_NAME
                        Name of the model to use
  --prompting.min_tokens PROMPTING.MIN_TOKENS
                        Minimum number of tokens to generate
  --prompting.max_tokens PROMPTING.MAX_TOKENS
                        Maximum number of tokens to generate
  --prompting.temperature PROMPTING.TEMPERATURE
                        Temperature for sampling
  --prompting.top_p PROMPTING.TOP_P
                        Top p for sampling
  --prompting.logprobs PROMPTING.LOGPROBS
                        Number of logprobs to return
  --prompting.repetition_penalty PROMPTING.REPETITION_PENALTY
                        Repetition penalty for sampling
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --wallet._mock        To turn on wallet mocking for testing purposes.
  --wallet.reregister WALLET.REREGISTER
                        Whether to reregister the wallet if it is not already registered.
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
  --metagraph._mock     To turn on metagraph mocking for testing purposes.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguemnts have been set.
```