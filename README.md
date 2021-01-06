# Bittensor 
[![Build status](https://circleci.com/gh/opentensor/bittensor.svg?style=shield)](https://circleci.com/gh/opentensor/bittensor)
[![Documentation Status](https://readthedocs.org/projects/bittensor-docs/badge/?version=latest)](https://bittensor-docs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Neural networks which mine crypto by producing information for their peers.

# Installation

Bittensor setup and installation requires a few steps in order to work properly, namely that is the setup of the parity substrate chain and the setup of bittensor in itself. First we will go through deploying the parity substrate chain. 

## Setup Parity Substrate chain
The parity substrate chain framework is the blockchain framework on which bittensor is based. It is the foundation on which nodes can stake, unstake, and set the weights (scores) of their peers. 
1. Clone the latest [subtensor](https://github.com/opentensor/subtensor) chain to your local environment. `git clone git@github.com:opentensor/subtensor.git`.
2. In a separate `tmux`, `screen`, or terminal, compile and run the chain as follows:
```
$ cargo build --release
./target/release/node-subtensor purge-chain --dev && ./target/release/node-subtensor --dev
```

## Setup bittensor
1. Clone this repository: `git clone git@github.com:opentensor/bittensor.git`.
2. Create a new virtual environment: `python3 -m venv env`.
3. Activate the environment: `source env/bin/activate`
4. Install requirements: `pip3 install -r requirements.txt`.
5. Install bittensor: `pip3 install - e .`

You should now be able to run the bittensor examples. A common error you may run into is `No module named 'bittensor'`. This is possibly because your `$PYTHONPATH` needs to be fixed, you can try running `export PYTHONPATH="$PYTHONPATH:$HOME/.python"` to fix it. 


## Acknowledgments
**learning-at-home/hivemind**:

## License
The MIT License (MIT)
Copyright © 2020 <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

