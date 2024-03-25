## Installation

First, make sure you have Bittensor installed correctly. There are three ways to install Bittensor:

1. Through the installer:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```

2. With pip:

```bash
pip install bittensor
```

3. From source:

```bash
git clone https://github.com/opentensor/bittensor.git
python3 -m pip install -e bittensor/
```

You can test your installation by running:

```bash
python3 -c "import bittensor; print(bittensor.__version__)"
```
## Logging
Make good use of the `bittensor.logging` module. It can be your friend and will help you find things that are otherwise difficult to get visibility on.

You can enable debug or trace modes by running:
```
import bittensor
bittensor.trace() # lowest level of granularity, best for figuring out what went wrong.
bittensor.debug() # for most everything else that you don't want to see normally at runtime
```
at the top of your script or source file to enable more verbose output logs.

You can also write your own in the code simply:
```python
# Bittensor's wallet maintenance class.
wallet = bittensor.wallet()

bittensor.logging.debug( f"wallet keypair: {wallet.hotkey}" )

...

# Bittensor's chain state object.
metagraph = bittensor.metagraph(netuid=1)

bittensor.logging.trace( f"metagraph created! netuid {metagraph.netuid}" )
```


## Querying the Network

Ensure you can query the Bittensor network using the Python API. If something is broken with your installation or the chain, this won't work out of the box. Here's an example of how to do this:

```python
import bittensor
bittensor.trace()

# Attempt to query through the foundation endpoint.
print(bittensor.prompt("Heraclitus was a "))
```

## Debugging Miners


First, try registering and running on a testnet:
```bash
btcli register --netuid <testnet uid> --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443
```

If that works, then try to register a miner on mainnet:

```bash
btcli register --netuid <subnetwork uid>
```

See if you can observe your slot specified by UID:

```bash
btcli overview --netuid <subnetwork uid>
```

Here's an example of how to run a pre-configured miner:

```bash
python3 bittensor/neurons/text_prompting/miners/GPT4ALL/neuron.py --netuid <subnetwork uid>
```

## Debugging with the Bittensor Package

The Bittensor package contains data structures for interacting with the Bittensor ecosystem, writing miners, validators, and querying the network. 

Try to use the Bittensor package to create a wallet, connect to the axon running on slot 10, and send a prompt to this endpoint and see where things are breaking along this typical codepath:

```python
import bittensor

# Bittensor's wallet maintenance class.
wallet = bittensor.wallet()

# Bittensor's chain interface.
subtensor = bittensor.subtensor()

# Bittensor's chain state object.
metagraph = bittensor.metagraph(netuid=1)

# Instantiate a Bittensor endpoint.
axon = bittensor.axon(wallet=wallet, metagraph=metagraph)

# Start servicing messages on the wire.
axon.start()

# Register this axon on a subnetwork
subtensor.serve_axon(netuid=1, axon=axon)

# Connect to the axon running on slot 10, use the wallet to sign messages.
dendrite = bittensor.text_prompting(keypair=wallet.hotkey, axon=metagraph.axons[10])

# Send a prompt to this endpoint
dendrite.forward(roles=['user'], messages=['Who is Rick James?'])
```

> NOTE: It may be helpful to throw in breakpoints such as with `pdb`.
```python
# some code ...
import pdb; pdb.set_trace() # breakpoint!
# more code ...

```
This will stop execution at the breakpoint you set and can operate on the stack directly in the terminal.

## Searching for strings
Use `ag`.  It's fast, convenient, and widely available on unix systems. Ag will highlight all occurnaces of a given pattern.

```bash
apt-get install silversearcher-ag
```

Usage:
```bash
$ ag "query_subtensor"

>>> bittensor/_subtensor/subtensor_mock.py
>>> 165:    e.g. We mock `Subtensor.query_subtensor` instead of all query methods.
>>> 536:    def query_subtensor(
>>> 1149:        curr_total_hotkey_stake = self.query_subtensor(
>>> 1154:        curr_total_coldkey_stake = self.query_subtensor(
>>> 1345:            return self.query_subtensor(name=name, block=block, params=[netuid]).value
>>> 
>>> bittensor/_subtensor/subtensor_impl.py
>>> 902:    def query_subtensor(
>>> 1017:        return self.query_subtensor("Rho", block, [netuid]).value
...
```

Remember, debugging involves a lot of trial and error. Don't be discouraged if things don't work right away. Keep trying different things, and don't hesitate to ask for help if you need it.
