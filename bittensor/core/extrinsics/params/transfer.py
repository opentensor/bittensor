from dataclasses import dataclass
from typing import Optional, Union

from bittensor.utils.balance import Balance


def get_transfer_fn_params(
    amount: Optional["Balance"], destination: str, keep_alive: bool
) -> tuple[str, dict[str, Union[str, int, bool]]]:
    """
    Helper function to get the transfer call function and call params, depending on the value and keep_alive flag
    provided.

    Parameters:
        amount: the amount of Tao to transfer. `None` if transferring all.
        destination: the destination SS58 of the transfer
        keep_alive: whether to enforce a retention of the existential deposit in the account after transfer.

    Returns:
        tuple[call function, call params]
    """
    call_params: dict[str, Union[str, int, bool]] = {"dest": destination}
    if amount is None:
        call_function = "transfer_all"
        if keep_alive:
            call_params["keep_alive"] = True
        else:
            call_params["keep_alive"] = False
    else:
        call_params["value"] = amount.rao
        if keep_alive:
            call_function = "transfer_keep_alive"
        else:
            call_function = "transfer_allow_death"
    return call_function, call_params


@dataclass
class TransferParams:
    @classmethod
    def transfer_all(
        cls,
        destination: str,
        amount: Optional[Balance] = None,
        keep_alive: bool = True,
    ) -> dict:
        """Returns the parameters for the `transfer_all`."""
        _, call_params = get_transfer_fn_params(amount, destination, keep_alive)
        return call_params

    @classmethod
    def transfer_keep_alive(
        cls,
        destination: str,
        amount: Optional[Balance] = None,
        keep_alive: bool = True,
    ) -> dict:
        """Returns the parameters for the `transfer_keep_alive`."""
        _, call_params = get_transfer_fn_params(amount, destination, keep_alive)
        return call_params

    @classmethod
    def transfer_allow_death(
        cls,
        destination: str,
        amount: Optional[Balance] = None,
        keep_alive: bool = True,
    ) -> dict:
        """Returns the parameters for the `transfer_allow_death`."""
        _, call_params = get_transfer_fn_params(amount, destination, keep_alive)
        return call_params
