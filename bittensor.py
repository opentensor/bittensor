import argparse
import bittensor
import os
from loguru import logger
from importlib.machinery import SourceFileLoader

from bittensor.subtensor import Keypair
from bittensor.config import Config

def main():

    # 1. Load passed --neuron_path for custom config.
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_path', default='/bittensor/neurons/mnist/', type=str, help='directory path to a neuron.py class.')
    parser.add_argument('--neuron_name', type=str, help='Name of the neuron to be run, or parent directory of the neuron.py class')
    params = parser.parse_known_args()[0]

    # 2. Load Config.
    logger.info('Load Config ...')
    config = Config.load(params.neuron_path)

    # If neuron name not set, force set it in the config.
    neuron_name = params.neuron_name
    if not neuron_name:
        neuron_name = params.neuron_path.rsplit("/",1)[1]
        neuron_name_dict = {"neuron_name": neuron_name}
        config.neuron.update(neuron_name_dict)

    logger.info(Config.toString(config))

    # 3. Load Keypair.
    logger.info('Load Keyfile ...')
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 4. Load Neuron.
    logger.info('Load Neuron ... ')
    neuron_module = SourceFileLoader("Neuron", os.getcwd() + '/' + params.neuron_path + '/neuron.py').load_module()
    neuron = neuron_module.Neuron( config )
    

    # 5. Load Session.
    logger.info('Build Session ... ')
    session = bittensor.init(config, keypair)

    # 6. Start Neuron.
    logger.info('Start ... ')
    with session:
        neuron.start(session)

if __name__ == "__main__":
    main()
