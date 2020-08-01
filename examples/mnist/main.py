from loguru import logger

import argparse
import pickle
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import bittensor
from bittensor import bittensor_pb2
import bittensor

class Net(bittensor.Synapse):
    """ An bittensor endpoint trained on 28, 28 pixel images to detect handwritten characters.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    # TODO(const): hide protos
    def indef(self):
        x_def = bittensor.bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = [-1, 784],
                    dtype = bittensor_pb2.FLOAT32,
                    requires_grad = True,
                )
        return [x_def]
    
    def outdef(self):
        y_def = bittensor.bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = [-1, 10],
                    dtype = bittensor_pb2.FLOAT32,
                    requires_grad = True,
                )
        return [y_def]
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)
        x = x.view(-1, 10)
        return x


def main(hparams):

    # Training params.
    batch_size_train = 50
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Dataset.
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root="~/tmp/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])),
                                               batch_size=batch_size_train,
                                               shuffle=True)

    # bittensor:
    # Load bittensor config from hparams.
    config = bittensor.Config(hparams)
    
    # Build the neuron from configs.
    neuron = bittensor.Neuron(config)
    
    # Init a trainable request router.
    router = bittensor.Router(x_dim = 784, key_dim = 100, topk = 10)
    
    # Build local network.
    net = Net()
    
    # Subscribe the local network to the network
    neuron.subscribe(net)
    
    # Start the neuron backend.
    neuron.start()
    
    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir='./runs/' + config.neuron_key)
    
    # Build the optimizer.
    optimizer = optim.SGD(net.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    def train(epoch, global_step):
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Flatten mnist inputs
            inputs = torch.flatten(data, start_dim=1)
            
            # Query the remote network.
            synapses = neuron.synapses() # Returns a list of synapses on the network.
            requests, scores = router.route(inputs, synapses) # routes inputs to network.
            responses = neuron(requests, synapses) # Makes network calls.
            remote = router.join(responses) # Joins responses based on scores.
            
            # Query the local network.
            local = net(inputs)

            # Train.
            output = local + remote
                        
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = neuron.getweights(synapses)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            neuron.setweights(synapses, weights)

            if batch_idx % log_interval == 0:
                writer.add_scalar('n_peers', len(neuron.metagraph.peers),
                                  global_step)
                writer.add_scalar('n_synapses', len(neuron.metagraph.synapses),
                                  global_step)
                writer.add_scalar('Loss/train', float(loss.item()),
                                  global_step)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tnP|nS: {}|{}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), len(neuron.metagraph.peers), len(neuron.metagraph.synapses)))

    epoch = 0
    global_step = 0
    try:
        while True:
            train(epoch, global_step)
            epoch += 1
    except Exception as e:
        logger.error(e)
        neuron.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
