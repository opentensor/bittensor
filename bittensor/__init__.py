from loguru import logger
import os
import sys
from bittensor.subtensor.interface import Keypair
from transformers import GPT2Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bittensor.config import Config
from bittensor.session import Session

# Bittensor code verion.
__version__ = '0.0.0'

# Bittensor proto version.
__proto_version__ = 0
__proto_compatability__ = [ 0 ] # Version compatability

__blocktime__ = 6 # seconds
__network_dim__ = 512 # Tensor.shape = [batch_size, sequence_lenght, 512]

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


