<div align="center">

# **Bittensor**
[![Build status](https://circleci.com/gh/opentensor/bittensor.svg?style=shield)](https://circleci.com/gh/opentensor/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/3rUr6EcvbB)

---

### Incentivized Peer to Peer Neural Networks

[Discord](https://discord.gg/3rUr6EcvbB)) • [Docs](https://opentensor.github.io/index.html) • [Network](https://www.bittensor.com/metagraph) • [Research](https://uploads-ssl.webflow.com/5cfe9427d35b15fd0afc4687/5fa940aea6a95b870067cf09_bittensor.pdf) • [Code](https://github.com/opentensor/BitTensor)

</div>

# Setup

```
$ git clone https://github.com/opentensor/bittensor.git                             # Clone the repository
$ cd bittensor && pip install -r requirements.txt && pip install -e .               # Install bittensor
$ bittensor-cli new_wallet                                                          # Generate default keys
```

# Run Subtensor
```
$ git clone https://github.com/opentensor/subtensor                                 # Clone subtensor
$ ./bin/release/node-subtensor                                                      # Run a main net validator
```

# Run Bittensor
```
$ python examples/text/gpt2-wiki.py   --session.trial_uid=test                      # Train gpt2 on wiki-text
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
