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

Bittensor is a market which monetizes intelligence production accross the internet. Validators price this information by learning to speculate on its value against unsupervised objectives. Producers who perform well are rewarded with token inflation. Consumers purchase this currency to gain access to network. Bittensor is collectively-run, open-source, and open-access. For more info, read our [paper](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/6021920718efe27873351f68_bittensor.pdf).

## Install

```bash
$ pip3 install bittensor
```

## Consumer 
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

## Producer
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

## Validator 
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

---

### License
The MIT License (MIT)
Copyright © 2021 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### Acknowledgments
**learning-at-home/hivemind**
