import argparse
import subprocess
import bittensor
import time

from loguru import logger
from importlib.machinery import SourceFileLoader
from substrateinterface import SubstrateInterface, Keypair

def main():

    # Init the argparser.
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--neuron_path', default = '', type=str, help='path to bittensor neuron to run.')
    hparams = argparser.parse_args()

    # Load bittensor config.
    logger.info('Load config.')
    config_service = bittensor.ConfigService(hparams.neuron_path)
    config = config_service.create(hparams.neuron_path + "/config.ini", argparser)
    if not config:
        logger.error("Invalid configuration. Aborting")
        quit(-1)
    else:
        config.log()

    # Load bittensor keypair.
    # TODO (const): load from file.
    logger.info('Load keyfile')
    try:
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
    except:
        logger.error('Unable to load key file with error {}', e)
        quit(-1)

    # Main try catch.
    try:
        # Start substrate chain sending logs to subprocess.PIPE and subprocess.STDOUT.
        logger.info('Start subtensor background process.')
        try:
            subtensor_process = subprocess.Popen(['./subtensor/target/release/node-subtensor', '--dev'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            time.sleep(2)
        except Exception as e:
            logger.error('Unable to start subtensor process with error {}', e)
            assert False

        # Load Neuron module.
        logger.info('Load neuron module.')
        try:
            neuron_module = SourceFileLoader("Neuron", 'bittensor/' + config.neuron_path + "/neuron.py").load_module()
        except Exception as e:
            logger.error('Unable to load Neuron class at {} with error {}', config.neuron_path, e)
            assert False


        bittensor.init(config, keypair)
        
        bittensor.session.start()
        bittensor.session.subscribe()
        
        neuron = neuron_module.Neuron( config, bittensor.session )
        neuron.start()
         
    finally:

        # Stop neuron.
        logger.info('Stop neuron.')
        if neuron != None:
            try:
                neuron.stop()
            except:
                logger.error('Unable to stop neuron.')

        # Unsubscribe from subtensor.
        logger.info('Unsubscribe Neuron.')
        try:
            bittensor.session.unsubscribe()
        except:
            logger.error('Unable to unsubscribe from subtensor.')

        # Stop bittensor background threads.
        logger.info('Stop bittensor.')
        try:
            bittensor.session.stop()
        except:
            logger.error('Unable to stop bittensor')

        # Kill the subtensor process.
        logger.info('Terminate subtensor.')
        if subtensor_process != None:
            try:
                subtensor_process.terminate()
            except:
                logger.info('Unable to terminate subtensor process.')


if __name__ == "__main__":
    main()
