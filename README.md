<div align="center">

# **Bittensor**
[![Pushing Image to Docker](https://github.com/opentensor/bittensor/actions/workflows/docker_image_push.yml/badge.svg?branch=master)](https://github.com/opentensor/bittensor/actions/workflows/docker_image_push.yml)
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/3rUr6EcvbB)
[![PyPI version](https://badge.fury.io/py/bittensor.svg)](https://badge.fury.io/py/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

### Internet-scale Neural Networks

[Discord](https://discord.gg/3rUr6EcvbB) • [Docs](https://app.gitbook.com/@opentensor/s/bittensor/) • [Network](https://www.bittensor.com/metagraph) • [Research](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/5fa940aea6a95b870067cf09_bittensor.pdf) • [Code](https://github.com/opentensor/BitTensor)

</div>

Bittensor is a p2p-market that rewards the production of machine intelligence with a digital token called Tao. Peers in the system train models by mining knowledge from unsupervised datasets to share with others. Consumers access the network and distill what they learn into production models. The network is collectively-run, open-source, open-access, decentralized, and incentivized to produce state-of-the-art intelligence. For more info, read our [paper](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/6021920718efe27873351f68_bittensor.pdf).

## Install

```bash
$ pip3 install bittensor
```

## Client

```python
import bittensor
import torch
graph = bittensor.metagraph().load().sync().save()
text = torch.tensor([bittensor.tokenizer().encode( "The quick brown fox jumped over the lazy dog" )], dtype=torch.int64)
representations, _ = bittensor.dendrite().forward_text(
    endpoints = graph.endpoints,
    inputs = [text for _ in graph.endpoints]
)
representations = # List[ (1, 9, 512) ... x N ]
```

## Server

```python
import bittensor
import torch
from transformers import BertModel, BertConfig

model = BertModel(BertConfig())

def forward ( pubkey, inputs_x, modality)
  return torch.model( inputs ).narrow(2, 0, bittensor.__network_dim__)

axon = bittensor.axon(
    forward_callback = forward,
).start().subscribe()
```

---

### License
The MIT License (MIT)
Copyright © 2021 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### Acknowledgments
**learning-at-home/hivemind**
