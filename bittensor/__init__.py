from loguru import logger
import os
import sys
from bittensor.subtensor.interface import Keypair
from bittensor.config import Config
from bittensor.session import Session
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__version__ = '0.0.0'
__blocktime__ = 6 # seconds
__network_dim__ = 512

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

# Initialize the global bittensor session object.
session = None
def init(config: Config):
    global session
    session = Session(config)
    return session

# Tokenizer
__vocab_size__ = 204483
def __tokenizer__():
    from transformers import GPT2Tokenizer
    if __version__ == "0.0.0":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
        tokenizer.pad_token = '[PAD]'
        tokenizer.mask_token = -100
        return tokenizer
    else:
        raise ValueError



