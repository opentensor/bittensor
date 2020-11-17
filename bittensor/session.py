import struct
import socket
import bittensor
import subprocess
import time

from bittensor.synapse import Synapse
from bittensor.dendrite import Dendrite
from bittensor.axon import Axon
from bittensor.metagraph import Metagraph

from loguru import logger
from substrateinterface import SubstrateInterface, Keypair
from torch.utils.tensorboard import SummaryWriter

class FailedConnectToChain(Exception):
    pass

class FailedSubscribeToChain(Exception):
    pass

class FailedToEnterSession(Exception):
    pass

class SubtensorProcess:
    def __init__(self, config):
        self._config = config
        self._process = None
    
    def start(self):
        args = ['./subtensor/target/release/node-subtensor', '--dev']
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def stop(self):
        if self._process != None:
            self._process.terminate()
            self._process.kill()
            
class BTSession:
    def __init__(self, config, keypair: Keypair):
        self.config = config 
        self.__keypair = keypair
        self.metagraph = Metagraph(self.config, self.__keypair)
        self.axon = Axon(self.config, self.__keypair)
        self.dendrite = Dendrite(self.config, self.__keypair)
        self.tbwriter = SummaryWriter(log_dir=self.config.session_settings.logdir)
        self.subtensor_process = SubtensorProcess(self.config)

    def __del__(self):
        self.stop()

    def serve(self, synapse: Synapse):
        # Serve the synapse object on the grpc endpoint.
        self.axon.serve(synapse)

    def __enter__(self):
        logger.info('session enter')
        self.start()
        return self

    def __exit__(self, *args):
        logger.info('session exit')
        self.stop()

    def start(self):
        # Stop background grpc threads for serving the synapse object.
        logger.info('Start axon server...')
        try:
            self.axon.start()
        except Exception as e:
            logger.error('SESSION: Failed to start axon server with error: {}', e)
            raise FailedToEnterSession

        logger.info('Start chain ...')
        try:
            self.subtensor_process.start()
        except Exception as e:
            logger.error('SESSION: Failed to create subtensor subprocess with error: {}', e)
            raise FailedToEnterSession

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


    def stop(self):
        # Stop background grpc threads for serving synapse objects.
        logger.info('Unsubscribe from chain ...')
        try:
            if not self.metagraph.unsubscribe(10):
                logger.error('SESSION: Timeout while unsubscribing to the chain endpoint')
        except Exception as e:
            logger.error('SESSION: Error while unsubscribing to the chain endpoint: {}', e)

        logger.info('Stopping axon server..')
        try:
            self.axon.stop()
        except Exception as e:
            logger.error('SESSION: Error while stopping axon server: {} ', e)

        logger.info('Stopping subtensor process ...')
        try:
            self.subtensor_process.stop()
        except Exception as e:
            logger.error('SESSION: Error while stopping subtensor process: {} ', e)

    def neurons (self):
       return self.metagraph.neurons()

    def subscribe (self):
       self.metagraph.subscribe()

    def unsubscribe (self):
        self.metagraph.unsubscribe()
