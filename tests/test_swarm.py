from multiprocessing import Process
from loguru import logger 
import os
import random
import torch

import bittensor
from bittensor.synapses.null.model import NullSynapse

def test_null_synapse():
    config = bittensor.Config()
    meta = bittensor.Metagraph(config)
    axon = bittensor.Axon(config)
    dendrite = bittensor.Dendrite(config)
    synapse = NullSynapse(meta, dendrite)
    axon.serve(synapse)
    meta.subscribe(synapse)
    try:
        meta.start()
        axon.start()
        batch_size = 3
        sequence_len = 2
        synapse.forward(torch.zeros(batch_size, sequence_len, bittensor.__network_dim__), query=True)

    except Exception as e:
        logger.info(e)

    finally:
        meta.stop()
        axon.stop()

def test_metagraph_swarm():
    n = 10
    ports = [x for x in range(8000, 8000 + n)]
    metagraphs = []
    for i in range(n):
        metagraph_port = str(ports[i])
        if i == 0:
            bootstrap = 'localhost:' + str(ports[-1])
        else:
            bootstrap = 'localhost:' + str(ports[i-1])
        config = bittensor.Config(  metagraph_port = metagraph_port,
                                    bootstrap = bootstrap)
        meta = bittensor.Metagraph(config)
        metagraphs.append(meta)
        logger.info('address: {}, bootstrap: {}', metagraph_port, bootstrap)
        
    try:
        for i, meta in enumerate(metagraphs):
            meta.start()
            logger.info('start {}', i)

        for j in range(n*n):
            for i, meta in enumerate(metagraphs):
                meta.do_gossip()
            logger.info('gossip {}', j)

        for i, meta in enumerate(metagraphs):
            logger.info('meta {} - {}', i, meta.peers())

        for i, meta in enumerate(metagraphs):
            if len(meta.peers()) != n:
                logger.error("peers not fully connected")
                assert False
            else:
                logger.info("peers fully connected")

        for i, meta in enumerate(metagraphs):
            logger.info('stopping {}', i)
            meta.stop()

    except Exception as e:
        logger.error(e)

    finally:
        for i, meta in enumerate(metagraphs):
            meta.stop()
            logger.info('stop {}', i)



def test_metagraph_swarm():
    n = 10
    ports = [x for x in range(8000, 8000 + n)]
    metagraphs = []
    for i in range(n):
        metagraph_port = str(ports[i])
        if i == 0:
            bootstrap = 'localhost:' + str(ports[-1])
        else:
            bootstrap = 'localhost:' + str(ports[i-1])
        config = bittensor.Config(  metagraph_port = metagraph_port,
                                    bootstrap = bootstrap)
        meta = bittensor.Metagraph(config)
        metagraphs.append(meta)
        logger.info('address: {}, bootstrap: {}', metagraph_port, bootstrap)
        
    try:
        for i, meta in enumerate(metagraphs):
            meta.start()
            logger.info('start {}', i)

        for j in range(n*n):
            for i, meta in enumerate(metagraphs):
                meta.do_gossip()
            logger.info('gossip {}', j)

        for i, meta in enumerate(metagraphs):
            logger.info('meta {} - {}', i, meta.peers())

        for i, meta in enumerate(metagraphs):
            if len(meta.peers()) != n:
                logger.error("peers not fully connected")
                assert False
            else:
                logger.info("peers fully connected")

        for i, meta in enumerate(metagraphs):
            logger.info('stopping {}', i)
            meta.stop()

    except Exception as e:
        logger.error(e)

    finally:
        for i, meta in enumerate(metagraphs):
            meta.stop()
            logger.info('stop {}', i)

if __name__ == "__main__": 
    test_null_synapse()