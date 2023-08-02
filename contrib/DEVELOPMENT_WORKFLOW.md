# Bittensor Development Workflow

## Table of Contents

- [1. Main Components](#1-main-components)
    - [Bittensor Protocol](#bittensor-protocol)
    - [Neurons](#neurons)
    - [Subtensor](#subtensor)
    - [Metagraph](#metagraph)
- [2. Developer Notes](#2-developer-notes)
    - [Notes for Pull Request](#notes-for-pull-request)
    - [Notes for Releasing](#notes-for-releasing)
    - [Notes for Logging](#notes-for-logging)
        - [Logging Setup](#logging-setup)
        - [Logging Messages](#logging-messages)
        - [Logging Variables](#logging-variables)
        - [Logging Exceptions](#logging-exceptions)



# 1. Main Components
### Bittensor Protocol

The Bittensor protocol is the backbone of the network. It defines how neurons communicate with each other and with the Metagraph.

### Neurons

Neurons are the nodes in the Bittensor network. They represent AI models that are learning from the network.

### Subtensor

Subtensor is a lower-level protocol that handles communication between neurons. It ensures that data is transfered securely and efficiently across the network.

### Metagraph

The Metagraph is a decentralized ledger that keeps track of the state of the Bittensor network. It records which neurons are part of the network and how they are connected.


# 2. Developer Notes



Notes for Pull Request
-----------------

You can see how to contribute with PR at [here](CONTRIBUTING.md/#contribution-workflow)


Notes for Releasing
-------------------------


You can see the details for Releasing [here](RELEASE_GUIDELINES.md)


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


