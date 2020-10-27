import unittest
import torch
import pytest
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import bittensor
import os
import random
import sys
import time

from loguru import logger
from bittensor.synapses.ffnn import FFNNSynapse, FFNNConfig 
    
class TestMnistSwarm(unittest.TestCase):
    num_nodes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_ports = [x for x in range(8000, 8000 + num_nodes)]
    axon_ports = [x for x in range(9000, 9000 + num_nodes)]
    configs = []
    metagraphs = []
    axons = []
    dendrites = []
    synapses = []
    optimizers = []

    def setUp(self):
        assert self.device == torch.device("cuda")
        logger.info('Build swarm...')
        for i in range(self.num_nodes):
            metagraph_port = str(self.meta_ports[i])
            axon_port = str(self.axon_ports[i])
            if i == 0:
                bootstrap = 'localhost:' + str(self.meta_ports[-1])
            else:
                bootstrap = 'localhost:' + str(self.meta_ports[i-1])
            config = bittensor.Config(  axon_port = axon_port,
                                        metagraph_port = metagraph_port,
                                        bootstrap = bootstrap)
            logger.info('config: {}', config)
                                        
            meta = bittensor.Metagraph(config)
            axon = bittensor.Axon(config)
            dendrite = bittensor.Dendrite(config)
            
            model_config = FFNNConfig()
            synapse = FFNNSynapse(model_config, dendrite, meta)
            synapse.to( self.device )

            assert synapse.device == torch.device("cuda")

            axon.serve( synapse )
            meta.subscribe( synapse )
            optimizer = optim.SGD(synapse.parameters(), lr=0.01, momentum=0.9)

            self.configs.append(config)
            self.axons.append(axon)
            self.metagraphs.append(meta)
            self.dendrites.append(dendrite)
            self.synapses.append(synapse)
            self.optimizers.append(optimizer)

            logger.info('synapse: {}', synapse)
        
        logger.info('Finished building graphs')

        # Connect metagraphs.
        logger.info('Connect swarm...')
        try:
            for i, meta in enumerate(self.metagraphs):
                meta.start()
                logger.info('start meta {}', i)

            logger.info('Connecting metagraphs ...')
            for j in range(self.num_nodes*self.num_nodes):
                for i, meta in enumerate(self.metagraphs):
                    meta.do_gossip()
            for i, meta in enumerate(self.metagraphs):
                if len(meta.peers()) != self.num_nodes:
                    logger.error("peers not fully connected")
                    assert False
            logger.info("Metagraphs fully connected.")

        except Exception as e:
            logger.error(e)

        finally:
            for i, meta in enumerate(self.metagraphs):
                logger.info('stopping meta {}', i)
                meta.stop()

    def test_mnist_swarm_loss(self):
        logger.info('Load Mnist dataset.')
        batch_size_train = 64
        train_data = torchvision.datasets.MNIST(root = self.configs[0].datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train, shuffle=True, num_workers=2)

        # Run swarm.
        logger.info('Running mnist swarm..')
        try:
            for i, axon in enumerate(self.axons):
                axon.start()
                logger.info('start axon {}', i)

            epochs = 2
            log_interval = 10
            accuracies = [0 for _ in self.synapses]
            logger.info('Train ...')
            time.sleep(2)
            for epoch in range(epochs):
                for i, model in enumerate(self.synapses):
                    correct = 0.0
                    for batch_idx, (images, labels) in enumerate(trainloader):
                        # Clear gradients on model parameters.
                        self.optimizers[i].zero_grad()

                        # Targets and images to correct device.
                        labels = torch.LongTensor(labels).to(self.device)
                        images = images.to(self.device)
                        
                        # Computes model outputs and loss.
                        output = model(images, labels, remote = True)

                        # Loss and step.
                        max_logit = output.remote_target.data.max(1, keepdim=True)[1]
                        correct += max_logit.eq( labels.data.view_as(max_logit) ).sum()

                        loss = output.remote_target_loss
                        loss.backward()
                        self.optimizers[i].step()

                        if batch_idx % log_interval == 0:
                            n = len(train_data)
                            accuracy = (100. * correct.item()) / ((batch_idx + 1) * batch_size_train)
                            logger.info('Synapse {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {}'.format(i,
                                epoch, (batch_idx * batch_size_train), n, (100. * batch_idx * batch_size_train)/n, output.remote_target_loss.item(), accuracy)) 
                            accuracies[i] = accuracy

                        if batch_idx > 100:
                            break

            # Assert loss convergence.
            logger.info(accuracies)
            for acc in accuracies:
                if (acc < 0.60):
                    assert False 

        except Exception as e:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #logger.error(exc_type, fname, exc_tb.tb_lineno)
            assert False

        finally:
            for i, axon in enumerate(self.axons):
                logger.info('stopping axon {}', i)
                axon.stop()