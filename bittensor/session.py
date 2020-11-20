from bittensor.synapse import Synapse
from bittensor.dendrite import Dendrite
from bittensor.axon import Axon
from bittensor.metagraph import Metagraph
from bittensor.tb_logger import TBLogger
from substrateinterface import SubstrateInterface, Keypair
from loguru import logger

class FailedConnectToChain(Exception):
    pass

class FailedSubscribeToChain(Exception):
    pass

class FailedToEnterSession(Exception):
    pass

class FailedToPollChain(Exception):
    pass

class BTSession:
    def __init__(self, config, keypair: Keypair):
        self.config = config 
        self.__keypair = keypair
        self.metagraph = Metagraph(self.config, self.__keypair)
        self.axon = Axon(self.config, self.__keypair)
        self.dendrite = Dendrite(self.config, self.__keypair)
        self.tbwriter = TBLogger(self.config.session_settings.logdir)

    def __del__(self):
        self.stop()

    def serve(self, synapse: Synapse):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                synapse (:obj:`bittensor.Synapse`, `required`): 
                    synpase object to serve on the axon server.
        """
        self.axon.serve(synapse)

    def __enter__(self):
        logger.info('session enter')
        self.start()
        return self

    def __exit__(self, *args):
        logger.info('session exit')
        self.stop()

    def start(self):
        logger.info('Connect to chain ...')
        try:
            if not self.metagraph.connect(5):
                logger.error('SESSION: Timeout while subscribing to the chain endpoint')
                raise FailedConnectToChain
        except Exception as e:
            logger.error('SESSION: Error while connecting to the chain endpoint: {}', e)
            raise FailedToEnterSession

        logger.info('Subscribe to chain ...')
        try:
            if not self.metagraph.subscribe(10):
                logger.error('SESSION: Timeout while subscribing to the chain endpoint')
                raise FailedSubscribeToChain
        except Exception as e:
            logger.error('SESSION: Error while subscribing to the chain endpoint: {}', e)
            raise FailedToEnterSession

        logger.info('Poll chain ...')
        try:
            self.metagraph.pollchain()
        except Exception as e:
            logger.error('SESSION: Failed during poll to chain with error: {}', e)
            raise FailedToPollChain

        logger.info('Start polling thread ...')
        try:
            self.metagraph.start()
        except Exception as e:
            logger.error('SESSION: Failed during poll to chain with error: {}', e)
            raise FailedToPollChain

        logger.info('Start axon server...')
        try:
            self.axon.start()
        except Exception as e:
            logger.error('SESSION: Failed to start axon server with error: {}', e)
            raise FailedToEnterSession


    def stop(self):
        logger.info('Stopping polling thread ...')
        try:
            self.metagraph.stop()
        except Exception as e:
            logger.error('SESSION: Failed during poll to chain with error: {}', e)
            raise FailedToPollChain

        logger.info('Stopping axon server..')
        try:
            self.axon.stop()
        except Exception as e:
            logger.error('SESSION: Error while stopping axon server: {} ', e)

        # Stop background grpc threads for serving synapse objects.
        logger.info('Unsubscribe from chain ...')
        try:
            if not self.metagraph.unsubscribe(10):
                logger.error('SESSION: Timeout while unsubscribing to the chain endpoint')
        except Exception as e:
            logger.error('SESSION: Error while unsubscribing to the chain endpoint: {}', e)

    def neurons (self):
       return self.metagraph.neurons()

    def subscribe (self):
       self.metagraph.subscribe()

    def unsubscribe (self):
        self.metagraph.unsubscribe()
