# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import bittensor
from transformers import GPT2Tokenizer

import bittensor 

class tokenizer:

    def __new__( cls, version: str = None ):
        if version == None:
            version = bittensor.__version__
        return tokenizer.get_tokenizer_for_version( version )
        
    # Tokenizer
    # NOTE (const): tokenizers are guaranteed to improve and expand as time progresses. We version the tokenizer here.
    # neurons must be aware that versions will increase and be ready to convert between tokenizers.
    # TODO (const): Add functionality to allow tokenizer conversion. i.e. for input token conversion.
    @staticmethod
    def get_tokenizer_for_version( version = bittensor.__version__ ):

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
        tokenizer.padding_side = "left"
        tokenizer.add_prefix_space = False
        tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
        tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
        tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
        tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
        tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
        tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
        tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
        additional_special_tokens = [
            "<s>NOTUSED",  # Used by BARThez
            "</s>NOTUSED", # Used by BARThez
            "<eop>", # Used by MarianMT
            "<eod>", # Used by MarianMT
            "<formula>", # Used by Transformer XL
            "<mask_1>" # Used by Pegasus
            "<special0>", # Used by XLM
            "<special1>", # Used by XLM
            "<special2>", # Used by XLM
            "<special3>", # Used by XLM
            "<special4>", # Used by XLM
            "<special5>", # Used by XLM
            "<special6>", # Used by XLM
            "<special7>", # Used by XLM
            "<special8>", # Used by XLM
            "<special9>", # Used by XLM
        ]
        tokenizer.additional_special_tokens = additional_special_tokens
        return tokenizer