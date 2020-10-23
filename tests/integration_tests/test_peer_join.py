import unittest
from loguru import logger
import bittensor
import time

class TestPeerJoinsMetagraph(unittest.TestCase):
    meta = None
    axon = None

    def setUp(self):
        logger.info('Loading Metagraph...')
        metagraph_port = "8123"
        axon_port = "8120"
        bootstrap = "3.tcp.ngrok.io:24519"
        config = bittensor.Config(  axon_port = axon_port,
                                    metagraph_port = metagraph_port,
                                    bootstrap = bootstrap)
        logger.info('config: {}', config)
        self.meta = bittensor.Metagraph(config)
        self.axon = bittensor.Axon(config)
        
        self.meta.start()
        self.axon.start()

    def test_wait_for_peers(self):
        limit = 60
        attempt = 0
        peers_found = False
        logger.info("Waiting for peer to join network, {} second timeout...".format(limit))
        while( attempt < limit ):
            logger.info("Checking for peer, attempt: {}, peers_found: {}".format(attempt, len(self.meta.peers())))
            #self.meta.do_gossip()
            if len(self.meta.peers()) == 2:
                peers_found = True
                break
            else:
                time.sleep(1)
            attempt += 1
        
        assert peers_found
    
    def test_wait_node_to_boot_up(self):
        limit = 60
        attempt = 0
        peers_found = False
        logger.info("Waiting for mnist node to join network, {} second timeout...".format(limit))
        while( attempt < limit ):
            logger.info("Checking for peer, attempt: {}, mnist_nodes_found: {}".format(attempt, len(self.meta.peers())))
            #self.meta.do_gossip()
            if len(self.meta.peers()) == 1:
                peers_found = True
                break
            else:
                time.sleep(1)
            attempt += 1
        
        assert peers_found