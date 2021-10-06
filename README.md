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

Bittensor is a market which monetizes intelligence production accross the internet. Validators price this information by learning to speculate on its value against unsupervised objectives. Producers who perform well are rewarded with token inflation. Consumers purchase this currency to gain access to network. Bittensor is collectively-run, open-source, and open-access. For more info, read our [paper](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/6021920718efe27873351f68_bittensor.pdf).

- [1. Documentation](#1-documentation)
- [2. Install](#2-install)
- [3. Using Bittensor](#3-using-bittensor)
  - [3.1. Consumer](#31-consumer)
  - [3.2. Producer](#32-producer)
  - [3.3. Validator](#33-validator)
- [4. Features](#4-features)
  - [4.1. Creating a bittensor wallet](#41-creating-a-bittensor-wallet)
  - [4.2. Running a template miner](#42-running-a-template-miner)
  - [4.3. Running a template server](#43-running-a-template-server)
- [5. License](#5-license)
- [6. Acknowledgments](#6-acknowledgments)

## 1. Documentation

https://app.gitbook.com/@opentensor/s/bittensor/

## 2. Install

```bash
$ pip3 install bittensor
```

## 3. Using Bittensor

The following examples showcase how to use the Bittensor API for 3 seperate purposes.

### 3.1. Consumer 

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
representations = // N tensors with shape (1, 9, 512)
...
// Distill model. 
...
loss.backward() // Accumulate gradients on endpoints.
```

### 3.2. Producer

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
dataset = bittensor.dataloader()
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
$ pip3 install bittensor
$ bittensor-cli new_coldkey --wallet.name <WALLET NAME>
$ bittensor-cli new_hotkey --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

### 4.2. Running a template miner

The following command will run Bittensor's template miner

```bash
$ pip3 install bittensor
$ python ~/.bittensor/bittensor/miners/text/template_miner.py
```

OR with customized settings

```bash
$ pip3 install bittensor
$ python ~/.bittensor/bittensor/miners/text/template_miner.py --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

For the full list of settings, please run

```bash
$ python ~/.bittensor/bittensor/miners/text/template_miner.py --help
```

### 4.3. Running a template server

The template server follows a similar structure as the template miner. 

```bash
$ pip3 install bittensor
$ python ~/.bittensor/bittensor/miners/text/template_server.py --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

For the full list of settings, please run

```bash
$ python ~/.bittensor/bittensor/miners/text/template_server.py --help
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
