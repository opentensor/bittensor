from dataclasses import dataclass

from bittensor.utils.balance import Balance


@dataclass
class SimSwapResult:
    """
    Represents the result of a simulated swap operation.

    This class is used to encapsulate the amounts and fees for the  simulated swap process, including both tao and alpha token values.
    It provides a convenient way to manage and interpret the swap results.

    Attributes:
        tao_amount: The amount of tao tokens obtained as the result of the swap.
        alpha_amount: The amount of alpha tokens obtained as the result of the swap.
        tao_fee: The fee associated with the tao token portion of the swap.
        alpha_fee: The fee associated with the alpha token portion of the swap.
    """

    tao_amount: Balance
    alpha_amount: Balance
    tao_fee: Balance
    alpha_fee: Balance

    @classmethod
    def from_dict(cls, data: dict, netuid: int) -> "SimSwapResult":
        """
        Converts a dictionary to a SimSwapResult instance.

        This method acts as a factory to create a SimSwapResult object using the data
        from a dictionary. It parses the specified dictionary, converts values into
        Balance objects, and sets associated units based on parameters and context.

        Parameters:
            data: A dictionary containing the swap result data. It must include  the keys "tao_amount",  "alpha_amount",
                "tao_fee", and "alpha_fee" with their respective values.
            netuid: A network-specific unit identifier used to set the unit for alpha-related amounts.

        Returns:
            SimSwapResult: An instance of SimSwapResult initialized with the parsed  and converted data.
        """
        return cls(
            tao_amount=Balance.from_rao(data["tao_amount"]).set_unit(0),
            alpha_amount=Balance.from_rao(data["alpha_amount"]).set_unit(netuid),
            tao_fee=Balance.from_rao(data["tao_fee"]).set_unit(0),
            alpha_fee=Balance.from_rao(data["alpha_fee"]).set_unit(netuid),
        )
