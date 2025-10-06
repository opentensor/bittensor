from dataclasses import dataclass
from typing import Optional, Union

from bittensor_wallet import Wallet


@dataclass
class RegisterSubnet:
    wallet: Wallet


@dataclass
class ActivateSubnet:
    wallet: Wallet
    netuid: Optional[int] = None


@dataclass
class RegisterNeuron:
    wallet: Wallet
    netuid: Optional[int] = None


REGISTER_SUBNET = RegisterSubnet
ACTIVATE_SUBNET = ActivateSubnet
REGISTER_NEURON = RegisterNeuron

STEPS = Union[REGISTER_SUBNET, ACTIVATE_SUBNET, REGISTER_NEURON]


def split_command(command):
    """Parse command and return four objects (wallet, pallet, sudo, kwargs)."""
    d = command._asdict()
    wallet = d.pop("wallet")
    pallet = d.pop("pallet")
    sudo = d.pop("sudo")
    func_name = type(command).__name__.lower()
    kwargs = d
    return wallet, pallet, sudo, func_name, kwargs


def is_instance_namedtuple(obj):
    """Check if the object is an instance of a namedtuple."""
    return (
        isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_asdict")
    )
