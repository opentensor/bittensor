from multiprocessing import Process
from loguru import logger 
import os
import random
import sys
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional

import bittensor
from bittensor.synapses.mnist.model import MnistSynapse

def test_mnist_synapse_swarm():
    n = 5
    batches = 100
    batch_size_train = 64
    meta_ports = [x for x in range(8000, 8000 + n)]
    axon_ports = [x for x in range(9000, 9000 + n)]
    configs = []
    metagraphs = []
    axons = []
    dendrites = []
    synapses = []
    optimizers = []
    logger.info('Build swram...')
    for i in range(n):
        metagraph_port = str(meta_ports[i])
        axon_port = str(axon_ports[i])
        if i == 0:
            bootstrap = 'localhost:' + str(meta_ports[-1])
        else:
            bootstrap = 'localhost:' + str(meta_ports[i-1])
        config = bittensor.Config(  axon_port = axon_port,
                                    metagraph_port = metagraph_port,
                                    bootstrap = bootstrap)
        logger.info('config: {}', config)
                                    
        meta = bittensor.Metagraph(config)
        axon = bittensor.Axon(config)
        dendrite = bittensor.Dendrite(config)
        
        synapse = MnistSynapse(dendrite, meta)
        axon.serve(synapse)
        meta.subscribe(synapse)
        optimizer = optim.SGD(synapse.parameters(), lr=0.1, momentum=0.9)

        configs.append(config)
        axons.append(axon)
        metagraphs.append(meta)
        dendrites.append(dendrite)
        synapses.append(synapse)
        optimizers.append(optimizer)

        logger.info('synapse: {}', synapse)
    logger.info('Finished building graphs')

    # Connect metagraphs.
    logger.info('Connect swram...')
    try:
        for i, meta in enumerate(metagraphs):
            meta.start()
            logger.info('start meta {}', i)

        logger.info('Connecting metagraphs ...')
        for j in range(n*n):
            for i, meta in enumerate(metagraphs):
                meta.do_gossip()
        for i, meta in enumerate(metagraphs):
            if len(meta.peers()) != n:
                logger.error("peers not fully connected")
                assert False
        logger.info("Metagraphs fully connected.")

    except Exception as e:
        logger.error(e)

    finally:
        for i, meta in enumerate(metagraphs):
            logger.info('stopping meta {}', i)
            meta.stop()

    logger.info('Load Mnist dataset.')
    train_data = torchvision.datasets.MNIST(root = configs[0].datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train, shuffle=True, num_workers=2)

    # Run swarm.
    logger.info('Running mnist swarm..')
    try:
        for i, axon in enumerate(axons):
            axon.start()
            logger.info('start axon {}', i)

        for batch_idx, (images, labels) in enumerate(trainloader):
            losses = []
            for i, model in enumerate(synapses): 
                optimizer.zero_grad()
                labels = torch.LongTensor(labels)
                output = model(images, labels, query = True)
                loss = output['loss']
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                loss.backward()
                optimizers[i].step()
                losses.append(loss.item())
            logger.info('Batch {} {}', batch_idx, losses)
            if batch_idx > batches:
                break

    except Exception as e:
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(exc_type, fname, exc_tb.tb_lineno)

    finally:
        for i, axon in enumerate(axons):
            logger.info('stopping axon {}', i)
            axon.stop()

class NullSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on PIL images to detect handwritten characters.
    """
    def __init__(self, metagraph, dendrite):
        super(NullSynapse, self).__init__()
        self.router = bittensor.Router(x_dim = bittensor.__network_dim__, key_dim = 100, topk = 10)
        self.metagraph = metagraph
        self.dendrite = dendrite

    def forward_tensor(self, tensor: torch.LongTensor):
        logger.info("accept forward tensor {}", tensor)
        return self.forward(inputs = tensor, query = False)

    def forward (   self, 
                    inputs: torch.Tensor,
                    query: bool = False):

        logger.info('Inputs: {} {}', inputs.shape, inputs)
        batch_size = inputs.shape[0]
        sequence_dim = inputs.shape[1]
        network_dim = bittensor.__network_dim__
        if query:
            logger.info('do query')
            context = torch.ones((batch_size, network_dim)) 
            synapses = self.metagraph.synapses() 
            logger.info('synapses: {} {}', len(synapses), synapses)
            requests, _ = self.router.route( synapses, context, inputs )
            responses = self.dendrite.forward_tensor( synapses, requests )
            assert len(responses) == len(synapses)
            network = self.router.join( responses )

        output = inputs + torch.ones((batch_size, sequence_dim, network_dim))
        return output

def test_null_synapse_swarm():
    n = 5
    meta_ports = [x for x in range(8000, 8000 + n)]
    axon_ports = [x for x in range(9000, 9000 + n)]
    metagraphs = []
    axons = []
    dendrites = []
    synapses = []
    logger.info('Build graphs...')
    for i in range(n):
        metagraph_port = str(meta_ports[i])
        axon_port = str(axon_ports[i])
        if i == 0:
            bootstrap = 'localhost:' + str(meta_ports[-1])
        else:
            bootstrap = 'localhost:' + str(meta_ports[i-1])
        config = bittensor.Config(  axon_port = axon_port,
                                    metagraph_port = metagraph_port,
                                    bootstrap = bootstrap)
        logger.info('config: {}', config)
                                    
        meta = bittensor.Metagraph(config)
        axon = bittensor.Axon(config)
        dendrite = bittensor.Dendrite(config)
        
        synapse = NullSynapse(meta, dendrite)
        axon.serve(synapse)
        meta.subscribe(synapse)

        axons.append(axon)
        metagraphs.append(meta)
        dendrites.append(dendrite)
        synapses.append(synapse)

        logger.info('synapse: {}', synapse)
    logger.info('Finished building graphs')

    # Connect metagraphs.
    try:
        for i, meta in enumerate(metagraphs):
            meta.start()
            logger.info('start meta {}', i)

        for i, axon in enumerate(axons):
            axon.start()
            logger.info('start axon {}', i)

        logger.info('Connecting metagraphs ...')
        for j in range(n*n):
            for i, meta in enumerate(metagraphs):
                meta.do_gossip()
        for i, meta in enumerate(metagraphs):
            if len(meta.peers()) != n:
                logger.error("peers not fully connected")
                assert False
        logger.info("Metagraphs fully connected.")

        logger.info('Forward messages...')
        for i in range(1):
            for j, synapse in enumerate(synapses):
                batch_size = 3
                sequence_len = 2
                inputs = torch.ones(batch_size, sequence_len, bittensor.__network_dim__) * (i + 1)  * (j + 1)
                logger.info(inputs)
                synapse.forward(inputs, query=True)
        logger.info('Done forwarding synapses.')

    except Exception as e:
        logger.error(e)

    finally:
        for i, meta in enumerate(metagraphs):
            logger.info('stopping meta {}', i)
            meta.stop()

        for i, axon in enumerate(axons):
            logger.info('stopping axon {}', i)
            axon.stop()

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

    except Exception as e:
        logger.error(e)

    finally:
        for i, meta in enumerate(metagraphs):
            meta.stop()
            logger.info('stop {}', i)

if __name__ == "__main__": 
    test_mnist_synapse_swarm()
