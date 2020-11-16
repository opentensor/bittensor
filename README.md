# Bittensor 
[![Build status](https://circleci.com/gh/opentensor/bittensor.svg?style=shield)](https://circleci.com/gh/opentensor/bittensor)
[![Documentation Status](https://readthedocs.org/projects/bittensor-docs/badge/?version=latest)](https://bittensor-docs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bittensor is a peer-to-peer neural network which rewards the information produced by the computers which compose it.

## Links
- [Documentation](https://bittensor-docs.readthedocs.io/en/latest/index.html)
- [Installation](https://bittensor-docs.readthedocs.io/en/latest/getting-started/installation.html)
- [Getting Started](https://bittensor-docs.readthedocs.io/en/latest/getting-started/run-multiple-bittensor-instances.html)
- [Architecture](https://bittensor-docs.readthedocs.io/en/latest/bittensor-deep-dive/bittensor-architecture.html)

## Acknowledgments
**learning-at-home/hivemind**:


## License
The MIT License (MIT)
Copyright © 2020 <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Installation

You'll need to install our own custom substrate interface in order for bittensor to work

This is for a system wide installation
```bash
git clone git@github.com:opentensor/py-substrate-interface.git
cd py-substrate-interface
sudo export GITHUB_REF=refs/tags/v3.0 && python3 ./setup.py install
```