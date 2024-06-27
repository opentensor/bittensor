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
from .stake import StakeCommand, StakeShow
from .unstake import UnStakeCommand
from .overview import OverviewCommand
from .register import (
    PowRegisterCommand,
    RegisterCommand,
    RunFaucetCommand,
    SwapHotkeyCommand,
)
from .delegates import (
    NominateCommand,
    ListDelegatesCommand,
    DelegateStakeCommand,
    DelegateUnstakeCommand,
    MyDelegatesCommand,
)
from .wallets import (
    NewColdkeyCommand,
    NewHotkeyCommand,
    RegenColdkeyCommand,
    RegenColdkeypubCommand,
    RegenHotkeyCommand,
    UpdateWalletCommand,
    WalletCreateCommand,
    WalletBalanceCommand,
    GetWalletHistoryCommand,
)
from .transfer import TransferCommand
from .inspect import InspectCommand
from .metagraph import MetagraphCommand
from .list import ListCommand
from .misc import UpdateCommand, AutocompleteCommand
from .profile import (
    ProfileCommand,
    ProfileListCommand,
    ProfileShowCommand,
    ProfileDeleteCommand,
    ProfileSetValueCommand,
    ProfileDeleteValueCommand,
    ProfileSetCommand,
)

from .senate import (
    SenateCommand,
    ProposalsCommand,
    ShowVotesCommand,
    SenateRegisterCommand,
    SenateLeaveCommand,
    VoteCommand,
)
from .network import (
    RegisterSubnetworkCommand,
    SubnetLockCostCommand,
    SubnetListCommand,
    SubnetSudoCommand,
    SubnetHyperparamsCommand,
    SubnetGetHyperparamsCommand,
)
from .root import (
    RootRegisterCommand,
    RootList,
    RootSetWeightsCommand,
    RootGetWeightsCommand,
    RootSetBoostCommand,
    RootSetSlashCommand,
)
from .identity import GetIdentityCommand, SetIdentityCommand
