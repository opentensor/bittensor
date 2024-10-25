# The MIT License (MIT)
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from munch import Munch, munchify

defaults: Munch = munchify(
    {
        "netuid": 1,
        "subtensor": {
            "network": "rao",
            "chain_endpoint": "wss://rao.chain.opentensor.ai:9944",
            "_mock": False,
        },
        "pow_register": {
            "num_processes": None,
            "update_interval": 50000,
            "output_in_place": True,
            "verbose": False,
            "cuda": {"dev_id": [0], "use_cuda": False, "tpb": 256},
        },
        "axon": {
            "port": 8091,
            "ip": "[::]",
            "external_port": None,
            "external_ip": None,
            "max_workers": 10,
            "maximum_concurrent_rpcs": 400,
        },
        "priority": {"max_workers": 5, "maxsize": 10},
        "prometheus": {"port": 7091, "level": "INFO"},
        "wallet": {
            "name": "default",
            "hotkey": "default",
            "path": "~/.bittensor/wallets/",
        },
        "dataset": {
            "batch_size": 10,
            "block_size": 20,
            "num_workers": 0,
            "dataset_names": "default",
            "data_dir": "~/.bittensor/data/",
            "save_dataset": False,
            "max_datasets": 3,
            "num_batches": 100,
        },
        "logging": {
            "debug": False,
            "trace": False,
            "record_log": False,
            "logging_dir": "~/.bittensor/miners",
        },
    }
)

# Subnets
from .subnets.list import ListSubnetsCommand

# from .subnets.metagraph import ShowMetagraph
from .subnets.create import RegisterSubnetworkCommand
from .subnets.pow_register import PowRegisterCommand
from .subnets.register import RegisterCommand
from .subnets.show import ShowSubnet

# Wallet
from .wallet.list import ListCommand
from .wallet.overview import OverviewCommand
from .wallet.transfer import TransferCommand
from .wallet.inspect import InspectCommand
from .wallet.balance import WalletBalanceCommand
from .wallet.create import WalletCreateCommand
from .wallet.new_hotkey import NewHotkeyCommand
from .wallet.new_coldkey import NewColdkeyCommand
from .wallet.regen_coldkey import RegenColdkeyCommand
from .wallet.regen_coldkeypub import RegenColdkeypubCommand
from .wallet.regen_hotkey import RegenHotkeyCommand
from .wallet.faucet import RunFaucetCommand
from .wallet.update import UpdateWalletCommand
from .wallet.swap_hotkey import SwapHotkeyCommand
from .wallet.set_identity import SetIdentityCommand
from .wallet.get_identity import GetIdentityCommand
from .wallet.history import GetWalletHistoryCommand

# Staking
from .stake.add import AddStakeCommand
from .stake.remove import RemoveStakeCommand
from .stake.list import StakeList
from .stake.move import MoveStakeCommand

# Sudo
from .sudo.vote import VoteCommand
from .sudo.senate import SenateCommand
from .sudo.set_take import SetTakeCommand
from .sudo.set_hparam import SubnetSudoCommand
from .sudo.hyperparameters import SubnetHyperparamsCommand

# Misc
from .misc.misc import UpdateCommand, AutocompleteCommand

# Weights
from .weights import CommitWeightCommand, RevealWeightCommand

# Children
from .stake.children import (
    GetChildrenCommand,
    SetChildrenCommand,
    RevokeChildrenCommand,
)

# TODO: Unused command, either remove or use
# from .senate import (
#     ShowVotesCommand,
#     SenateRegisterCommand,
#     SenateLeaveCommand,
# )
# from .root import (
#     RootSetWeightsCommand,
#     RootGetWeightsCommand,
#     RootSetBoostCommand,
#     RootSetSlashCommand,
# )
