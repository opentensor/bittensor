<div align="center">

# **Bittensor SDK** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![CodeQL](https://github.com/opentensor/bittensor/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/opentensor/bittensor/actions)
[![PyPI version](https://badge.fury.io/py/bittensor.svg)](https://badge.fury.io/py/bittensor)
[![Codecov](https://codecov.io/gh/opentensor/bittensor/graph/badge.svg)](https://app.codecov.io/gh/opentensor/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Internet-scale Neural Networks <!-- omit in toc -->

[Discord](https://discord.gg/qasY3HA9F9) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper) • [Documentation](https://docs.bittensor.com)

</div>

- [Overview of Bittensor](#overview-of-bittensor)
- [The Bittensor SDK](#the-bittensor-sdk)
- [Is Bittensor a blockchain or an AI platform?](#is-bittensor-a-blockchain-or-an-ai-platform)
- [Subnets](#subnets)
- [Subnet validators and subnet miners](#subnet-validators-and-subnet-miners)
- [Yuma Consensus](#yuma-consensus)
- [Release Notes](#release-notes)
- [Install Bittensor SDK](#install-bittensor-sdk)
- [Upgrade](#upgrade)
- [Install on macOS and Linux](#install-on-macos-and-linux)
  - [Install using a Bash command](#install-using-a-bash-command)
  - [Install using `pip3 install`](#install-using-pip3-install)
  - [Install from source](#install-from-source)
  - [Verify using Python interpreter](#verify-using-python-interpreter)
  - [Verify by listing axon information](#verify-by-listing-axon-information)
- [Release Guidelines](#release-guidelines)
- [Contributions](#contributions)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview of Bittensor

Welcome! Bittensor is an open source platform on which you can produce competitive digital commodities. These digital commodities can be machine intelligence, storage space, compute power, protein folding, financial markets prediction, and many more. You are rewarded in **TAO** when you produce best digital commodities.

## The Bittensor SDK

The Opentensor Foundation (OTF) provides all the open source tools, including this Bittensor SDK, the codebase and the documentation, with step-by-step tutorials and guides, to enable you to participate in the Bittensor ecosystem. 

- **Developer documentation**: https://docs.bittensor.com.
- **A Beginner's Q and A on Bittensor**: https://docs.bittensor.com/questions-and-answers.
- **Bittensor whitepaper**: https://bittensor.com/whitepaper.

This Bittensor SDK contains ready-to-use Python packages for interacting with the Bittensor ecosystem, writing subnet incentive mechanisms, subnet miners, subnet validators and querying the subtensor (the blockchain part of the Bittensor network). 

---

## Is Bittensor a blockchain or an AI platform?

In Bittensor there is one blockchain, and many platforms that are connected to this one blockchain. We call these platforms as **subnets**, and this one blockchain  **subtensor**. So, a subnet can be AI-related or it can be something else. The Bittensor network has a number of distinct subnets. All these subnets interact with subtensor blockchain. If you are thinking, "So, subnets are not part of the blockchain but only interact with it?" then the answer is "yes, exactly."

## Subnets

Each category of the digital commodity is produced in a distinct subnet. Applications are built on these specific subnets. End-users of these applications would be served by these applications.

## Subnet validators and subnet miners

Subnets, which exist outside the blockchain and are connected to it, are off-chain competitions where only the best producers are rewarded. A subnet consists of off-chain **subnet validators** who initiate the competition for a specific digital commodity, and off-chain **subnet miners** who compete and respond by producing the best quality digital commodity.

## Yuma Consensus

Scores are assigned to the top-performing subnet miners and subnet validators. The on-chain Yuma Consensus determines the TAO rewards for these top performers. The Bittensor blockchain, the subtensor, runs on decentralized validation nodes, just like any blockchain.

**This SDK repo is for Bittensor platform only**
This Bittensor SDK codebase is for the Bittensor platform only, designed to help developers create subnets and build tools on Bittensor. For subnets and applications, refer to subnet-specific websites, which are maintained by subnet owners.

## Release Notes

See [Bittensor SDK Release Notes](https://docs.bittensor.com/bittensor-rel-notes).

---

## Install Bittensor SDK

Before you can start developing, you must install Bittensor SDK and then create Bittensor wallet.

## Upgrade

If you already installed Bittensor SDK, make sure you upgrade to the latest version. Run the below command:

```bash
python3 -m pip install --upgrade bittensor
```

---

## Install on macOS and Linux

### Note for macOS users
The macOS preinstalled CPython installation is compiled with LibreSSL instead of OpenSSL. There are a number
of issues with LibreSSL, and as such is not fully supported by the libraries used by bittensor. Thus we highly recommend, if 
you are using a Mac, to first install Python from [Homebrew](https://brew.sh/). Additionally, the Rust FFI bindings 
[if installing from precompiled wheels (default)] require the Homebrew-installed OpenSSL pacakge. If you choose to use
the preinstalled Python version from macOS, things may not work completely.

### Installation
You can install Bittensor SDK on your local machine in either of the following ways. **Make sure you verify your installation after you install**:
- [Install using a Bash command](#install-using-a-bash-command).
- [Install using `pip3 install`](#install-using-pip3-install)
- [Install from source](#install-from-source)

### Install using a Bash command

This is the most straightforward method. It is recommended for a beginner as it will pre-install requirements like Python, if they are not already present on your machine. Copy and paste the following `bash` command into your terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```

**For Ubuntu-Linux users**
If you are using Ubuntu-Linux, the script will prompt for `sudo` access to install all required apt-get packages.

### Install using `pip3 install`

```bash
python3 -m venv bt_venv
source bt_venv/bin/activate
pip install bittensor
```

### Install from source

1. Create and activate a virtual environment

    - Create Python virtual environment. Follow [this guide on python.org](https://docs.python.org/3/library/venv.html#creating-virtual-environments).

    - Activate the new environment. Follow [this guide on python.org](https://docs.python.org/3/library/venv.html#how-venvs-work)

2. Clone the Bittensor SDK repo

```bash
git clone https://github.com/opentensor/bittensor.git
```

3.  Install

You can install using any of the below options:

- **Install SDK**: Run the below command to install Bittensor SDK in the above virtual environment. This will also install `btcli`.

    ```python
    pip install bittensor
    ```

- **Install SDK with `torch`**: Install Bittensor SDK with [`torch`](https://pytorch.org/docs/stable/torch.html).

   ```python
    pip install bittensor[torch]
    ```
  In some environments the above command may fail, in which case run the command with added quotes as shown below:

  ```python
    pip install "bittensor[torch]"
    ```

- **Install SDK with `cubit`**: Install Bittensor SDK with [`cubit`](https://github.com/opentensor/cubit).

  1. Install `cubit` first. See the [Install](https://github.com/opentensor/cubit?tab=readme-ov-file#install) section. **Only Python 3.9 and 3.10 versions are supported**. 
  2. Then install SDK with `pip install bittensor`.
  

### Troubleshooting
#### SSL: CERTIFICATE_VERIFY_FAILED

If you are encountering a `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate` 
error, use the command `python -m bittensor certifi` which will update your local SSL certificates.

---

## Install on Windows

To install and run Bittensor SDK on Windows you must install [**WSL 2** (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/about) on Windows and select [Ubuntu Linux distribution](https://github.com/ubuntu/WSL/blob/main/docs/guides/install-ubuntu-wsl2.md). 

After you installed the above, follow the same installation steps described above in [Install on macOS and Linux](#install-on-macos-and-linux).

**ALERT**: **Limited support on Windows**
While wallet transactions like delegating, transfer, registering, staking can be performed on a Windows machine using WSL 2, the mining and validating operations are not recommended and are not supported on Windows machines.

---

## Verify the installation

You can verify your installation in either of the below ways:

### Verify using `btsdk` version

```bash
python3 -m bittensor
```

The above command will show you the version of the `btsdk` you just installed.

### Verify using Python interpreter

1. Launch the Python interpreter on your terminal.   

    ```bash
    python3
    ```
2. Enter the following two lines in the Python interpreter.
   
    ```python
    import bittensor as bt
    print( bt.__version__ )
    ```
    The Python interpreter output will look like below:

    ```python
    Python 3.11.6 (main, Oct  2 2023, 13:45:54) [Clang 15.0.0 (clang-1500.0.40.1)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import bittensor as bt
    >>> print( bt.__version__ )
    <version number>
    ```
You will see the version number you installed in place of `<version number>`. 

### Verify by listing axon information

You can also verify the Bittensor SDK installation by listing the axon information for the neurons. Enter the following lines in the Python interpreter.

```python
import bittensor
metagraph = bittensor.Metagraph(1)
metagraph.axons[:10]
```
The Python interpreter output will look like below.

```bash
[AxonInfo( /ipv4/3.139.80.241:11055, 5GqDsK6SAPyQtG243hbaKTsoeumjQQLhUu8GyrXikPTmxjn7, 5D7u5BTqF3j1XHnizp9oR67GFRr8fBEFhbdnuVQEx91vpfB5, 600 ), AxonInfo( /ipv4/8.222.132.190:5108, 5CwqDkDt1uk2Bngvf8avrapUshGmiUvYZjYa7bfA9Gv9kn1i, 5HQ9eTDorvovKTxBc9RUD22FZHZzpy1KRfaxCnRsT9QhuvR6, 600 ), AxonInfo( /ipv4/34.90.71.181:8091, 5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2, 5ChuGqW2cxc5AZJ29z6vyTkTncg75L9ovfp8QN8eB8niSD75, 601 ), AxonInfo( /ipv4/64.247.206.79:8091, 5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN, 5E7W9QXNoW7se7B11vWRMKRCSWkkAu9EYotG5Ci2f9cqV8jn, 601 ), AxonInfo( /ipv4/51.91.30.166:40203, 5EXYcaCdnvnMZbozeknFWbj6aKXojfBi9jUpJYHea68j4q1a, 5CsxoeDvWsQFZJnDCyzxaNKgA8pBJGUJyE1DThH8xU25qUMg, 601 ), AxonInfo( /ipv4/149.137.225.62:8091, 5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3, 5Ccmf1dJKzGtXX7h17eN72MVMRsFwvYjPVmkXPUaapczECf6, 600 ), AxonInfo( /ipv4/38.147.83.11:8091, 5Hddm3iBFD2GLT5ik7LZnT3XJUnRnN8PoeCFgGQgawUVKNm8, 5DCQw11aUW7bozAKkB8tB5bHqAjiu4F6mVLZBdgJnk8dzUoV, 610 ), AxonInfo( /ipv4/38.147.83.30:41422, 5HNQURvmjjYhTSksi8Wfsw676b4owGwfLR2BFAQzG7H3HhYf, 5EZUTdAbXyLmrs3oiPvfCM19nG6oRs4X7zpgxG5oL1iK4MAh, 610 ), AxonInfo( /ipv4/54.227.25.215:10022, 5DxrZuW8kmkZPKGKp1RBVovaP5zHtPLDHYc5Yu82Z1fWqK5u, 5FhXUSmSZ2ec7ozRSA8Bg3ywmGwrjoLLzsXjNcwmZme2GcSC, 601 ), AxonInfo( /ipv4/52.8.243.76:40033, 5EnZN591jjsKKbt3yBtfGKWHxhxRH9cJonqTKRT5yTRUyNon, 5ChzhHyGmWwEdHjuvAxoUifHEZ6xpUjR67fDd4a42UrPysyB, 601 )]
>>>
```

### Testing
You can run integration and unit tests in interactive mode of IDE or in terminal mode using the command:
```bash
pytest tests/integration_tests
pytest tests/unit_tests
```

#### E2E tests have 2 options for launching (legacy runner):
- using a compiler based on the substrait code
- using an already built docker image (docker runner)

#### Local environment variables:
- `LOCALNET_SH_PATH` - path to `localnet.sh` script in cloned subtensor repository (for legacy runner);
- `BUILD_BINARY` - (`=0` or `=1`) - used with `LOCALNET_SH_PATH` for build or not before start localnet node (for legacy runner);
- `USE_DOCKER` - (`=0` or `=1`) - used if you want to use specific runner to run e2e tests (for docker runner);
- `FAST_BLOCKS` - (`=0` or `=1`) - allows you to run a localnet node in fast or non-fast blocks mode (for both types of runers).
- `SKIP_PULL` - used if you are using a Docker image, but for some reason you want to temporarily limit the logic of updating the image from the repository.

#### Using `docker runner` (default for now):
- E2E tests with docker image do not require preliminary compilation
- are executed very quickly
- require docker installed in OS

How to use:
```bash
pytest tests/e2e_tests
```

#### Using `legacy runner`:
- Will start compilation of the collected code in your subtensor repository
- you must provide the `LOCALNET_SH_PATH` variable in the local environment with the path to the file `/scripts/localnet.sh` in the cloned repository within your OS
- you can use the `BUILD_BINARY=0` variable, this will skip the copy step for each test.
- you can use the `USE_DOCKER=0` variable, this will run tests using the "legacy runner", even if docker is installed in your OS

#### How to use:
Regular e2e tests run
```bash
LOCALNET_SH_PATH=/path/to/your/localnet.sh pytest tests/e2e_tests
```

If you want to skip re-build process for each e2e test
```bash
BUILD_BINARY=0 LOCALNET_SH_PATH=/path/to/your/localnet.sh pytest tests/e2e_tests
```

If you want to use legacy runner even with installed Docker in your OS
```bash
USE_DOCKER=0 BUILD_BINARY=0 LOCALNET_SH_PATH=/path/to/your/localnet.sh pytest tests/e2e_tests
```

---

## Release Guidelines
Instructions for the release manager: [RELEASE_GUIDELINES.md](./contrib/RELEASE_GUIDELINES.md) document.

## Contributions
Ready to contribute? Read the [contributing guide](./contrib/CONTRIBUTING.md) before making a pull request.

## License
The MIT License (MIT)
Copyright © 2025 The Opentensor Foundation
Copyright © 2025 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Acknowledgments
**learning-at-home/hivemind**
