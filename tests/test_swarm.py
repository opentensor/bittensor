from multiprocessing import Process
from loguru import logger 
import os
import random

import bittensor

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
    test_metagraph_swarm()