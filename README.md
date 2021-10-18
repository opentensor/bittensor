<div align="center">

# **Bittensor** <!-- omit in toc -->
[![Pushing Image to Docker](https://github.com/opentensor/bittensor/actions/workflows/docker_image_push.yml/badge.svg?branch=master)](https://github.com/opentensor/bittensor/actions/workflows/docker_image_push.yml)
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/3rUr6EcvbB)
[![PyPI version](https://badge.fury.io/py/bittensor.svg)](https://badge.fury.io/py/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### Internet-scale Neural Networks <!-- omit in toc -->

[Discord](https://discord.gg/3rUr6EcvbB) • [Docs](https://app.gitbook.com/@opentensor/s/bittensor/) • [Network](https://www.bittensor.com/metagraph) • [Research](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/5fa940aea6a95b870067cf09_bittensor.pdf) • [Code](https://github.com/opentensor/BitTensor)

</div>

At Bittensor, we are creating an open, decentralized, peer-to-peer network that functions as a market system for the development of artificial intelligence. Our purpose is not only to accelerate the development of AI by creating an environment optimally condusive to its evolution, but to democratize the global production and use of this valuable commodity. Our aim is to disrupt the status quo: a system that is centrally controlled, inefficient and unsustainable. In developing the Bittensor API, we are allowing standalone engineers to monetize their work, gain access to sophisticated machine intelligence models and join our community of creative, forward-thinking individuals. For more info, read our [paper](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/6021920718efe27873351f68_bittensor.pdf).

- [1. Documentation](#1-documentation)
- [2. Install](#2-install)
- [3. Using Bittensor](#3-using-bittensor)
  - [3.1. Client](#31-client)
  - [3.2. Server](#32-server)
  - [3.3. Validator](#33-validator)
- [4. Features](#4-features)
  - [4.1. Creating a bittensor wallet](#41-creating-a-bittensor-wallet)
  - [4.2. Selecting the network to join](#42-selecting-the-network-to-join)
  - [4.3. Running a template miner](#43-running-a-template-miner)
  - [4.4. Running a template server](#44-running-a-template-server)
  - [4.5. Subscription to the network](#45-subscription-to-the-network)
  - [4.6. Syncing with the chain/ Finding the ranks/stake/uids of other nodes](#46-syncing-with-the-chain-finding-the-ranksstakeuids-of-other-nodes)
  - [4.7. Finding and creating the endpoints for other nodes in the network](#47-finding-and-creating-the-endpoints-for-other-nodes-in-the-network)
  - [4.8. Querying others in the network](#48-querying-others-in-the-network)
  - [4.9. Creating a Priority Thread Pool for the axon](#49-creating-a-priority-thread-pool-for-the-axon)
- [5. License](#5-license)
- [6. Acknowledgments](#6-acknowledgments)

## 1. Documentation

https://app.gitbook.com/@opentensor/s/bittensor/

## 2. Install
Two ways to install Bittensor. 

1. Through installer (recommended):
```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```

2. Through pip (Advanced):
```bash
$ pip3 install bittensor
```

## 3. Using Bittensor

The following examples showcase how to use the Bittensor API for 3 seperate purposes.

### 3.1. Client 

For users that want to explore what is possible using on the Bittensor network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m6c4_D1FHHcZxnDJCW4F0qORWhXV_hc_?usp=sharing)
```python
import bittensor
import torch
wallet = bittensor.wallet().create()
graph = bittensor.metagraph().sync()
representations, _ = bittensor.dendrite( wallet = wallet ).forward_text (
    endpoints = graph.endpoints,
    inputs = "The quick brown fox jumped over the lazy dog"
)
representations = // N tensors with shape (1, 9, 1024)
...
// Distill model. 
...
loss.backward() // Accumulate gradients on endpoints.
```

### 3.2. Server

For users that want to serve up a custom model onto the Bittensor network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12nGV6cmoZNvywb_z6E8CDzdHCQ3F7tpQ?usp=sharing)
```python
import bittensor
import torch
from transformers import BertModel, BertConfig

model = BertModel( BertConfig(vocab_size = bittensor.__vocab_size__, hidden_size = bittensor.__network_dim__) )
optimizer = torch.optim.SGD( [ {"params": model.parameters()} ], lr = 0.01 )

def forward_text( pubkey, inputs_x ):
    return model( inputs_x )
  
def backward_text( pubkey, inputs_x, grads_dy ):
    with torch.enable_grad():
        outputs_y = model( inputs_x.to(device) ).last_hidden_state
        torch.autograd.backward (
            tensors = [ outputs_y.to(device) ],
            grad_tensors = [ grads_dy.to(device) ]
        )
        optimizer.step()
        optimizer.zero_grad() 

wallet = bittensor.wallet().create()
axon = bittensor.axon (
    wallet = wallet,
    forward_text = forward_text,
    backward_text = backward_text
).start().subscribe()
```

### 3.3. Validator 

For users that want to validate the models that currently on the Bittensor network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m6c4_D1FHHcZxnDJCW4F0qORWhXV_hc_?usp=sharing)
```python
import bittensor
import torch

graph = bittensor.metagraph().sync()
dataset = bittensor.dataset()
chain_weights = torch.ones( [graph.n.item()], dtype = torch.float32 )

for batch in dataset.dataloader( 10 ):
    ...
    // Train chain_weights.
    ...
bittensor.subtensor().set_weights (
    weights = chain_weights,
    uids = graph.uids,
    wait_for_inclusion = True,
    wallet = bittensor.wallet(),
)
```
## 4. Features

### 4.1. Creating a bittensor wallet 


```bash
$ bittensor-cli new_coldkey --wallet.name <WALLET NAME>
$ bittensor-cli new_hotkey --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

### 4.2. Selecting the network to join 
There are two open Bittensor networks: Kusanagi and Akatsuki.

- Kusanagi is the test network. Use Kusanagi to get familiar with Bittensor without worrying about losing valuable tokens. 
- Akatsuki is the main network. The main network will reopen on Bittensor-akatsuki: November 2021.

```bash
$ export NETWORK=akatsuki 
$ python (..) --subtensor.network $NETWORK
```

### 4.3. Running a template miner

The following command will run Bittensor's template miner

```bash
$ python ~/.bittensor/bittensor/miners/text/template_miner.py
```

OR with customized settings

```bash
$ python ~/.bittensor/bittensor/miners/text/template_miner.py --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

For the full list of settings, please run

```bash
$ python ~/.bittensor/bittensor/miners/text/template_miner.py --help
```

### 4.4. Running a template server

The template server follows a similar structure as the template miner. 

```bash
$ python ~/.bittensor/bittensor/miners/text/template_server.py --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

For the full list of settings, please run

```bash
$ python ~/.bittensor/bittensor/miners/text/template_server.py --help
```

###  4.5. Subscription to the network

The subscription to the bittensor network is done using the axon. We must first create a bittensor wallet and a bittensor axon to subscribe.

```python
import bittensor

wallet = bittensor.wallet().create()
axon = bittensor.axon (
    wallet = wallet,
    forward_text = forward_text,
    backward_text = backward_text
).start().subscribe()
```

### 4.6. Syncing with the chain/ Finding the ranks/stake/uids of other nodes

Information from the chain are collected by the metagraph.

```python
import bittensor

meta = bittensor.metagraph()
meta.sync()

# --- uid ---
print(meta.uids)

# --- hotkeys ---
print(meta.hotkeys)

# --- ranks ---
print(meta.R)

# --- stake ---
print(meta.S)

```

### 4.7. Finding and creating the endpoints for other nodes in the network

```python
import bittensor

meta = bittensor.metagraph()
meta.sync()

### Address for the node uid 0
address = meta.endpoints[0]
endpoint = bittensor.endpoint.from_tensor(address)
```

### 4.8. Querying others in the network

```python
import bittensor

meta = bittensor.metagraph()
meta.sync()

### Address for the node uid 0
address = meta.endpoints[0]

### Creating the endpoint, wallet, and dendrite
endpoint = bittensor.endpoint.from_tensor(address)
wallet = bittensor.wallet().create()
den = bittensor.dendrite(wallet = wallet)

representations, _ = den.forward_text (
    endpoints = endpoint,
    inputs = "Hello World"
)

```

### 4.9. Creating a Priority Thread Pool for the axon

```python
import bittensor
import torch
from nuclei.server import server

model = server(config=config,model_name='bert-base-uncased',pretrained=True)
optimizer = torch.optim.SGD( [ {"params": model.parameters()} ], lr = 0.01 )
threadpool = bittensor.prioritythreadpool(config=config)
metagraph = bittensor.metagraph().sync()

def forward_text( pubkey, inputs_x ):
    def call(inputs):
        return model.encode_forward( inputs )

    uid = metagraph.hotkeys.index(pubkey)
    priority = metagraph.S[uid].item()
    future = threadpool.submit(call,inputs=inputs_x,priority=priority)
    try:
        return future.result(timeout= model.config.server.forward_timeout)
    except concurrent.futures.TimeoutError :
        raise TimeoutError('TimeOutError')
  

wallet = bittensor.wallet().create()
axon = bittensor.axon (
    wallet = wallet,
    forward_text = forward_text,
).start().subscribe()
```

---

## 5. License
The MIT License (MIT)
Copyright © 2021 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## 6. Acknowledgments
**learning-at-home/hivemind**
