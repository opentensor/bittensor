from .run import RunCommand
from .stake import StakeCommand
from .unstake import UnStakeCommand
from .overview import OverviewCommand
from .register import RegisterCommand
from .delegates import NominateCommand, ListDelegatesCommand, DelegateStakeCommand, DelegateUnstakeCommand
from .wallets import NewColdkeyCommand, NewHotkeyCommand, RegenColdkeyCommand, RegenColdkeypubCommand, RegenHotkeyCommand
from .transfer import TransferCommand
from .inspect import InspectCommand
from .metagraph import MetagraphCommand
from .list import ListCommand
from .weights import SetWeightsCommand, WeightsCommand
from .query import QueryCommand
from .misc import HelpCommand, UpdateCommand, ListSubnetsCommand