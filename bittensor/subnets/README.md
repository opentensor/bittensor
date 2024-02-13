# Bittensor Subnets API Guide

This guide provides comprehensive instructions on how to use and extend the Bittensor Subnets API, a powerful interface for interacting with the Bittensor network. The Bittensor Subnets API facilitates storing and retrieving data across the decentralized network, leveraging the unique capabilities of Bittensor's blockchain-based infrastructure.

## Overview

The Bittensor Subnets API consists of abstract classes and a registry system to dynamically handle API interactions. It allows developers to implement custom logic for storing and retrieving data, while also providing a straightforward way for end users to interact with these functionalities.

### Core Components

- **APIRegistry**: A central registry that manages API handlers. It allows for dynamic retrieval of handlers based on keys.
- **SubnetsAPI (Abstract Base Class)**: Defines the structure for API implementations, including methods for querying the network and processing responses.
- **StoreUserAPI & RetrieveUserAPI**: Concrete implementations of the `SubnetsAPI` for storing and retrieving user data.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.6 or later
- Bittensor library
- Additional dependencies as required by your project

### Installation

1. Install Bittensor if you haven't already:

```bash
pip install bittensor
```

2. Clone or integrate the API code into your project.

## Usage

### Storing Data

To store data on the Bittensor network:

1. Initialize your wallet and metagraph:

```python
import bittensor as bt

wallet = bt.wallet()
metagraph = bt.subtensor("test").metagraph(netuid=22)
```

2. Use the `StoreUserAPI` to store data:

```python
store_handler = StoreUserAPI(wallet)
cid = await store_handler(
    metagraph=metagraph,
    data=b"some data",
    encrypt=True,
    ttl=60 * 60 * 24 * 30,
    encoding="utf-8",
    uid=None,
)
```

### Retrieving Data

To retrieve data from the Bittensor network:

1. Use the `RetrieveUserAPI` with the CID obtained from storing data:

```python
retrieve_handler = RetrieveUserAPI(wallet)
retrieve_response = await retrieve_handler(metagraph=metagraph, cid=cid)
```

### Using the API Registry

To dynamically retrieve API handlers:

```python
# Retrieve a StoreUserAPI handler
store_handler = APIRegistry.get_api_handler("store_user", wallet)

# Retrieve a RetrieveUserAPI handler
retrieve_handler = APIRegistry.get_api_handler("retrieve_user", wallet)
```

## Implementing Custom API Handlers

To implement your own versions of `StoreUserAPI` or `RetrieveUserAPI`:

1. **Inherit from `SubnetsAPI`**: Your class should inherit from the `SubnetsAPI` abstract base class.

2. **Implement Required Methods**: Implement the `prepare_synapse` and `process_responses` abstract methods with your custom logic.

3. **Register Your Handler**: Use the `register_handler` decorator to register your API handler.

### Example

```python
from bittensor import wallet as bt_wallet

@register_handler("custom_store_user")
class CustomStoreUserAPI(SubnetsAPI):
    def __init__(self, wallet: "bt_wallet"):
        super().__init__(wallet)
        # Custom initialization here

    def prepare_synapse(self, *args, **kwargs):
        # Custom synapse preparation logic
        pass

    def process_responses(self, responses):
        # Custom response processing logic
        pass
```

## Conclusion

The Bittensor Subnets API offers a flexible and powerful way to interact with the decentralized Bittensor network. By following this guide, developers can easily extend the API with custom logic, while end users can utilize the provided functionalities to store and retrieve data on the network.