from loguru import logger
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bittensor code and protocol version.
__version__ = '0.0.0'

# Version compatiability, users running these code versions should be able to speak with each other
__compatability__ = { __version__ : [ __version__ ] }

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time.
__blocktime__ = 6 

# Tokenizer
# NOTE (const): tokenizers are guaranteed to improve and expand as time progresses. We version the tokenizer here.
# neurons must be aware that versions will increase and be ready to convert between tokenizers. 
# TODO (const): Add functionality to allow tokenizer conversion. i.e. for input token conversion. 
__vocab_size__ = (50278 + 100) if __version__ == "0.0.0" else ValueError ('No __vocab_size__ for version {}'.format( __version__ )) # Must match the __tokenizer__() vocab size.
def __tokenizer__( version = __version__ ):
    from transformers import GPT2Tokenizer
    if version in [ "0.0.0" ]:
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
            "<special9>" # Used by XLM
        ]
        tokenizer.additional_special_tokens = additional_special_tokens
        global __vocab_size__
        __vocab_size__ = len(tokenizer) + len(additional_special_tokens) + 100 # Plus 100 for eventual toke size increase.
        return tokenizer

    else:
        raise ValueError ('No tokenizer for version {}'.format(version))


# Default logger
logger_config = {
    "handlers": [{
        "sink":
            sys.stdout,
        "format":
            "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    }]
}
logger.configure(**logger_config)
