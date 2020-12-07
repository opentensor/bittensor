#!/bin/python3
import argparse
import bittensor
import os
from loguru import logger
from importlib.machinery import SourceFileLoader

from bittensor.subtensor import Keypair
from bittensor.config import Config
import asyncio
import rollbar
from argparse import ArgumentParser


def main():
    # 1. Load customr configuration arguments
    params = load_custom_config_args()

    # 2. Load Config.
    config = load_config(params)

    # If neuron name not set, force set it in the config.
    force_neuron_name_into_config(config, params)

    # Display current configuration
    logger.info(Config.toString(config))

    # 3. Load Keypair.
    keypair = load_keypair()

    # 4. Load Neuron.
    neuron = load_neuron(config, params)

    # 5. Load Session.
    session = load_session(config, keypair)

    # 6. Start Neuron.
    logger.info('Start ... ')

    with session:
        neuron.start(session)


def load_session(config, keypair):
    logger.info('Build Session ... ')
    session = bittensor.init(config, keypair)
    return session


def load_neuron(config, params):
    logger.info('Load Neuron ... ')
    neuron_module = SourceFileLoader("Neuron", os.getcwd() + '/' + params.neuron_path + '/neuron.py').load_module()
    neuron = neuron_module.Neuron(config)
    return neuron


def load_keypair():
    logger.info('Load Keyfile ...')
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    return keypair


def force_neuron_name_into_config(config, params):
    neuron_name = params.neuron_name
    if not neuron_name:
        neuron_name = params.neuron_path.rsplit("/", 1)[1]
        neuron_name_dict = {"neuron_name": neuron_name}
        config.neuron.update(neuron_name_dict)


def load_custom_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_path', default='/bittensor/neurons/mnist/', type=str,
                        help='directory path to a neuron.py class.')
    parser.add_argument('--neuron_name', type=str,
                        help='Name of the neuron to be run, or parent directory of the neuron.py class')
    params = parser.parse_known_args()[0]
    return params


def load_config(params):
    logger.info('Load Config ...')
    config = Config.load(params.neuron_path)
    return config


if __name__ == "__main__":
    token = os.environ.get("ROLLBAR_TOKEN", None)
    if not token:
        main()
    else:
        env = os.environ.get("BT_ENV", "production")

        logger.info("Error reporting enabled using {}:{}",token,env)
        rollbar.init(token, env)

        try:
            main()
        except Exception as e:
            rollbar.report_exc_info()
            raise e
