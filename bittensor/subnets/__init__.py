from . import subnet1 as sn1  # Sn1
from . import subnet5 as image  # Sn5
from . import subnet15 as chain  # Sn15
from . import subnet18 as cortex  # Sn18
from . import subnet21 as storage  # Sn21


def setup_synapse(subnet: int, protocol: str, *args, **kwargs):
    """
    Setup a synapse for the given subnet and protocol.
    Args:
        subnet (int): The subnet to setup the synapse for.
        protocol (str): The protocol to setup the synapse for. E.g. `Store` for storage sn21, or ImageRequest for image sn5.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    Returns:
        synapse (bittensor.synapse): The synapse object for the given subnet and protocol.
    """
    if subnet == 1:
        return sn1.setup_synapse(protocol, *args, **kwargs)
    elif subnet == 5:
        return image.setup_synapse(protocol, *args, **kwargs)
    elif subnet == 15:
        return chain.setup_synapse(protocol, *args, **kwargs)
    elif subnet == 18:
        return cortex.setup_synapse(protocol, *args, **kwargs)
    elif subnet == 21:
        return storage.setup_synapse(protocol, *args, **kwargs)
    else:
        raise ValueError(
            f"Subnet {subnet} not currently supported. Please open a PR if you are the subnet owner."
        )
