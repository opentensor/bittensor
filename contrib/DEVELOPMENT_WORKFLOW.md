# Bittensor Development Workflow

## Table of Contents
- [1. Architecture](#1-architecture)
    - [Overview of Bittensor's architecture](#overview-of-bittensors-architecture)
    - [How the different components interact?](#how-the-different-components-interact)
- [2. Main Components](#2-main-components)
    - [Bittensor Protocol](#bittensor-protocol)
    - [Neurons](#neurons)
    - [Subtensor](#subtensor)
    - [Metagraph](#metagraph)
- [3. Developer Notes](#3-developer-notes)
    - [Notes for Python Script Style](#notes-for-python-script-style)
        - [1. General Coding Style](#1-general-coding-style)
        - [2. Naming Conventions](#2-naming-conventions)
        - [3. Python Code](#3-python-code)
        - [4. Documentation](#4-documentation)
    - [Notes for Source Coding](#notes-for-source-coding)
        - [1. Neurons](#1-neurons)
        - [2. Metagraph](#2-metagraph)
        - [3. Subtensor](#3-subtensor)
    - [Notes for Pull Request](#notes-for-pull-request)
    - [Notes for Releasing](#notes-for-releasing)
        - [Versioning Script](#versioning-script)
        - [Release Script](#release-script)
        - [Security](#security)
        - [Release Verification](#release-verification)
        - [Post-release Actions](#post-release-actions)
    - [Notes for Logging](#notes-for-logging)
        - [Logging Setup](#logging-setup)
        - [Logging Messages](#logging-messages)
        - [Logging Variables](#logging-variables)
        - [Logging Exceptions](#logging-exceptions)


# 1. Architecture
### Overview of Bittensor's architecture

Bittensor's architecture consists of several key components, including the Bittensor protocol, the wallet, neurons, subtensor, and the Metagraph. These components work together to create a decentralized network for AI model training.

### How the different components interact?

Neurons in the Bittensor network interact with the Metagraph, a decentralized ledger that keeps track of the state of the network. The wallet is used to manage the tokens that are used for incentives in the network. The subtensor is a lower-level protocol that handles communication between neurons.

# 2. Main Components
### Bittensor Protocol

The Bittensor protocol is the backbone of the network. It defines how neurons communicate with each other and with the Metagraph.

### Neurons

Neurons are the nodes in the Bittensor network. They represent AI models that are learning from the network.

### Subtensor

Subtensor is a lower-level protocol that handles communication between neurons. It ensures that data is transfered securely and efficiently across the network.

### Metagraph

The Metagraph is a decentralized ledger that keeps track of the state of the Bittensor network. It records which neurons are part of the network and how they are connected.


# 3. Developer Notes


Notes for Python Script Style
------------------

### 1. General Coding Style
Python's official style guide is PEP 8, which provides conventions for writing code for the main Python distribution. Here are some key points:

- `Indentation:` Use 4 spaces per indentation level.

- `Line Length:` Limit all lines to a maximum of 79 characters.

- `Blank Lines:` Surround top-level function and class definitions with two blank lines. Method definitions inside a class are surrounded by a single blank line.

- `Imports:` Imports should usually be on separate lines and should be grouped in the following order:

    - Standard library imports.
    - Related third party imports.
    - Local application/library specific imports.
- `Whitespace:` Avoid extraneous whitespace in the following situations:

    - Immediately inside parentheses, brackets or braces.
    - Immediately before a comma, semicolon, or colon.
    - Immediately before the open parenthesis that starts the argument list of a function call.
- `Comments:` Comments should be complete sentences and should be used to clarify code and are not a substitute for poorly written code.

You can see the detailed coding style at [here](STYLE.md)


### 2. Naming Conventions

- `Classes:` Class names should normally use the CapWords Convention.
- `Functions and Variables:` Function names should be lowercase, with words separated by underscores as necessary to improve readability. Variable names follow the same convention as function names.

- `Constants:` Constants are usually defined on a module level and written in all capital letters with underscores separating words.

- `Non-public Methods and Instance Variables:` Use a single leading underscore (_). This is a weak "internal use" indicator.

- `Strongly "private" methods and variables:` Use a double leading underscore (__). This triggers name mangling in Python.

### 3. Python Code

- `List Comprehensions:` Use list comprehensions for concise and readable creation of lists.

- `Generators:` Use generators when dealing with large amounts of data to save memory.

- `Context Managers:` Use context managers (with statement) for resource management.

- `String Formatting:` Use f-strings for formatting strings in Python 3.6 and above.

- `Error Handling:` Use exceptions for error handling whenever possible.


Notes for Source Coding
-----------------------------
### 1. Neurons

#### Class Methods
-----------------
The class contains several methods:

`check_config`: This method is a placeholder and currently does nothing.

`add_args`: This method adds arguments to the command-line parser. These arguments are used to configure the GPT4All model and the GPT4ALLMiner.

`__init__`: This is the constructor of the class. It initializes the GPT4All model with the configuration provided in the command-line arguments.

`backward`: This method is a placeholder and currently does nothing.

`_process_history`: This is a helper method that processes a list of messages into a string. Each message is a dictionary with a 'role' and 'content'. The 'role' can be 'system', 'assistant', or 'user'.

`forward`: This method processes a list of messages, generates a response using the GPT4All model, and returns the response.

#### Command-line Arguments
--------------
This is the note for `gpt4all` script. Other models is similar to this note.
```bash
The script accepts several command-line arguments to configure the GPT4All model and the GPT4ALLMiner. These include the path to the pretrained GPT4All model, the number of context tokens, the number of parts to split the model into, the seed, whether to use half-precision for the key/value cache, whether to return logits for all tokens, whether to only load the vocabulary, whether to force the system to keep the model in RAM, whether to use embedding mode only, the number of threads to use, the maximum number of tokens to generate, the temperature for sampling, the top-p value for sampling, the top-k value for sampling, whether to echo the prompt, the last n tokens to penalize, the penalty for repeated tokens, the batch size for prompt processing, and whether to stream the results.
```
#### Main Execution
-----------

```bash
python3 bittensor/neurons/text/prompting/miners/gpt4all/neuron.py
    --netuid SUBNETWORK_TARGET_UID
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_HOTKEY_NAME
    --logging.debug
```

#### Dependencies
-------------------

The script depends on the `argparse`, `bittensor`, and `langchain.llms` modules. The `argparse` module is used to handle command-line arguments. The `bittensor` module provides the `BasePromptingMiner` class and various utility functions. The `langchain.llms` module provides the class.

#### License
---------------

The script is licensed under the MIT License. The copyright is held by Yuma Rao. The license allows anyone to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to certain conditions.

### 2. Metagraph

#### Class Methods
---------------------------

The class contains several methods:

`get_save_dir`: This static method returns the directory path for saving the metagraph based on the network and netuid.

`latest_block_path`: This static method returns the path of the latest block in a given directory.

`__init__`: This is the constructor of the class. It initializes the metagraph with the given netuid and network. It also has options to use a lite version and to sync the metagraph.

`sync`: This method syncs the metagraph with the chain. It can use either a lite version or a full version of the neurons in the chain.

`save`: This method saves the state of the metagraph to a file.

`load`: This method loads the state of the metagraph from a file.

`load_from_path`: This method loads the state of the metagraph from a file at a given path.


#### Dependencies
------------------------

The script depends on the `os`, `torch`, and `bittensor` modules. The `os` module is used to handle file and directory paths. The `torch` module provides the `torch.nn.Module` class and various tensor operations. The `bittensor` module provides various utility functions and classes.

#### License
---------------

The script is licensed under the MIT License. The copyright is held by Yuma Rao. The license allows anyone to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to certain conditions.

#### Usage
-------------

The `metagraph` class can be used to maintain and manipulate the state of a chain in a PyTorch model. It can be saved to a file and loaded from a file for persistence. It can also be synced with the chain to update its state.

### 3. Subtensor

This Python script is part of a blockchain system, specifically designed to handle various data structures such as `NeuronInfo`, `NeuronInfoLite`, `PrometheusInfo`, `DelegateInfo`, and `SubnetInfo`. These data classes represent different types of information in the system, including neuron metadata, delegate info, and subnet info.

The script also includes methods for encoding and decoding these data structures to and from a serialized format using SCALE encoding, a common choice for blockchain systems.

Key components of the script include:

`Custom RPC Type Registry`: This dictionary defines the data structures for various types of information in the system, such as `SubnetInfo`, `DelegateInfo`, `NeuronInfo`, etc. It is used in the decoding process to interpret the serialized data.

`ChainDataType Enum`: This enumeration is used to specify the type of data being handled, which can be one of `NeuronInfo`, `SubnetInfo`, `DelegateInfo`, `NeuronInfoLite`, or `DelegatedInfo`.

`from_scale_encoding Function`: This function is used to decode data from SCALE format. It uses the `scalecodec` library to handle the decoding process. The function takes a list of integers (the serialized data), the type of data, and flags indicating whether the data is a vector or an option. It returns a dictionary representing the decoded data.

`Data Classes`: The script defines several data classes (`NeuronInfo`, `NeuronInfoLite`, `PrometheusInfo`, `DelegateInfo`, `SubnetInfo`) to represent different types of information in the system. Each class includes a `from_vec_u8` method to create an instance of the class from serialized data, and a `fix_decoded_values` method to correct the format of decoded data.

`ProposalVoteData and ProposalCallData`: These are data structures used for handling proposal data in the system. `ProposalVoteData` is a dictionary that includes information about a proposal's votes, while `ProposalCallData` is a generic call object from the `scalecodec` library.

Please note that this script requires the bittensor and scalecodec libraries, which should be installed and properly configured in your environment.


Notes for Pull Request
-----------------

You can see how to contribute with PR at [here](CONTRIBUTING.md/#contribution-workflow)


Notes for Releasing
-------------------------


- Release Process
Branch Creation: Create a new branch named `release/VERSION`, where `VERSION` is the new version number.

- Version Update: Within the release branch, update the version by running the versioning script: `./scripts/release/versioning.sh --update UPDATE_TYPE`. The UPDATE_TYPE can be major, minor, or patch.

- Changelog Update: Add release notes to the CHANGELOG by running the script: `./scripts/release/add_notes_changelog.sh -A -V NEW_VERSION -P PREVIOUS_TAG -T GH_ACCESS_TOKEN`. Replace `NEW_VERSION` with the new version number, `PREVIOUS_TAG` with the previous version tag, and `GH_ACCESS_TOKEN` with your GitHub personal access token.

- Testing: Test the release branch thoroughly to ensure it meets all requirements.

- Release: After merging the release branch, run the release script to finalize the release.

### Versioning Script

The versioning script has two options:

- -`U, --update`: Specifies the type of update (major, minor, patch, or rc - release candidate).
- `-A, --apply:` Applies the release. Without this, the versioning script will only show a dry run without making any changes.

### Release Script
The release script also has two options:

- -A, --apply: Applies the release. Without this, the release script will only show a dry run without making any changes.
- -T,--github-token: Your GitHub personal access token, used to interact with the GitHub API.

### Security
To securely handle your GitHub personal access token, consider using a tool like pass or a similar tool that allows you to store the secret safely and not expose it in the history of the machine you use.

### Release Verification
After the execution of the release script, verify the release by checking for:

- A new git tag in [github.com](#)
- A new GitHub release in [github.com](#)
- A new pip package in [pypi.org](#)
- A new Docker image in [hub.docker.com](#)

### Post-release Actions
After a Bittensor release, update [cubit](#) by updating the Dockerfile, building a new Docker image, and pushing it to Docker Hub. The generated name will be the same but with `-cubit` in its name.

For more details, please see [here](RELEASE_GUIDELINES.md)


Notes for Logging
---------------------------

Logging is a crucial part of any application. It helps developers understand the flow of the program, debug issues, and keep track of events. In Bittensor, we use Python's built-in logging module to handle logging throughout the application.

### Logging Setup
At the start of your application, you should set up the root logger. This can be done in the main function or entry point of your application. Here's an example of how to set up a basic logger:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

```

In this example, `logging.basicConfig(level=logging.INFO)` sets up the root logger with a level of INFO. This means the logger will handle all messages of level INFO and above (i.e., WARNING, ERROR, and CRITICAL). You can adjust the level as needed.

`logger = logging.getLogger(__name__)` gets a logger instance that you can use to log messages. The `__name__` variable is used to set the name of the logger to the name of the module, which is a common practice.

### Logging Messages

Once you have a logger instance, you can log messages using the following methods:

- `logger.debug('Debug message')`
- `logger.info('Info message')`
- `logger.warning('Warning message')`
- `logger.error('Error message')`
- `logger.critical('Critical message')`

Each method corresponds to a level, and the message will be processed by the logger and its handlers based on their levels.

### Logging Variables

You can also log variables by passing them as arguments to the logging method:

```python
name = 'Bittensor'
logger.info('Hello, %s', name)

```
In this example, `%s` is a placeholder for a string, and `name` is the string that will replace the placeholder. This is similar to using the `%` operator for string formatting.

### Logging Exceptions

In addition to standard log messages, you can also log exception information. This is typically done in an exception handler:

```python
try:
    1 / 0
except Exception:
    logger.exception('An error occurred')
```
In this example, `logger.exception('An error occurred')` logs the message 'An error occurred' with level ERROR and adds exception information to the message.


