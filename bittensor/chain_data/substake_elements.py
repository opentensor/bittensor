from typing import Dict, List

from substrateinterface.utils.ss58 import ss58_encode

from bittensor.chain_data.utils import ChainDataType, from_scale_encoding, SS58_FORMAT
from bittensor.utils.balance import Balance


class SubstakeElements:
    @staticmethod
    def decode(result: List[int]) -> List[Dict]:
        descaled = from_scale_encoding(
            input_=result, type_name=ChainDataType.SubstakeElements, is_vec=True
        )
        result = []
        for item in descaled:
            result.append(
                {
                    "hotkey": ss58_encode(item["hotkey"], SS58_FORMAT),
                    "coldkey": ss58_encode(item["coldkey"], SS58_FORMAT),
                    "netuid": item["netuid"],
                    "stake": Balance.from_rao(item["stake"]),
                }
            )
        return result
