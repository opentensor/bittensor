<div align="center">

# **Bittensor**
[![Build status](https://circleci.com/gh/opentensor/bittensor.svg?style=shield)](https://circleci.com/gh/opentensor/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/3rUr6EcvbB)

---

### Incentivized Peer to Peer Neural Networks

[Discord](https://discord.gg/3rUr6EcvbB) • [Docs](https://opentensor.github.io/index.html) • [Network](https://www.bittensor.com/metagraph) • [Research](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/5fa940aea6a95b870067cf09_bittensor.pdf) • [Code](https://github.com/opentensor/BitTensor)

</div>

Bittensor is a peer-to-peer machine intelligence market which rewards miners in a digital token called Tao. Computers who compete in this market train machine learning models and share their learned representations with each other. Consumers can access this knowledge by making network queries and distilling what they learn into production models. To begin, network 1 focuses on unsupervised language understanding however, at a later date, the network will be expanded into the multi-model landscape. 

# Setting up

You will need to have rust and gcc installed on your box before these installation instructions work. Preferably you run your own local chain (see: running Subtensor) but you can connect to the main network through our entry peers. Each example in /examples/ demonstrates how to run a unique miner (dataset, training mechanism, and model) on the network. If you wish to have other nodes in the network access your model be sure to open a hole in your router or use the --use_upnpc flag (works for some routers). Note, tokens will be mined into the hotkey account associated with your miner but you will need to unstake them into your coldkey account before they can be moved.

# Installation and Wallet

If you wish to run bittensor through a docker container, simply create a new wallet, and run docker-compose:
```
$ bin/bittensor-cli new_wallet                                                      # Generate default keys
$ docker-compose up
```

If you wish to install it natively on your machine:
```
$ git clone https://github.com/opentensor/bittensor.git                             # Clone the repository
$ cd bittensor && pip3 install -r requirements.txt && pip3 install -e .               # Install bittensor
$ bin/bittensor-cli new_wallet                                                      # Generate default keys
```

# Running the chain locally
```
$ git clone https://github.com/opentensor/subtensor                                 # Clone subtensor
$ ./bin/release/node-subtensor                                                      # Run a main net validator
```

# Running a miner
```
$ python3 examples/TEXT/gpt2_wiki.py   --session.trial_uid=test                      # Train gpt2 on wiki-text
$ tensorboard --logdir=~/.bittensor/sessions/gpt2-wiki/test                         # Serve tensorboard
```

---

### License
The MIT License (MIT)
Copyright © 2021 opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### Acknowledgments
**learning-at-home/hivemind**
