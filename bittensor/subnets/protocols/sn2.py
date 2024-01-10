import bittensor as bt
import pydantic
from typing import List

class Translate(bt.Synapse):
    source_texts: List[str] = pydantic.Field(..., allow_mutation=False)
    translated_texts: List[str] = []
    source_lang: str = pydantic.Field(..., allow_mutation=False)
    target_lang: str = pydantic.Field(..., allow_mutation=False)
    required_hash_fields: list[str] = pydantic.Field(  ["source_texts", "source_lang", "target_lang"], allow_mutation = False)
