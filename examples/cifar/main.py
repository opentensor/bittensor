from loguru import logger

import argparse
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import bittensor
from bittensor import bittensor_pb2

class Net(bittensor.Synapse):
        
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):   
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def indef(self):
        x_def = bittensor.bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = [-1, 3072],
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
    
def main(hparams):

    batch_size = 4
    input_dimension = 3072
    output_dimension = 10

    # Load CIFAR
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # bittensor:
    # Load bittensor config from hparams.
    config = bittensor.Config(hparams)
    
    # Build the neuron from configs.
    neuron = bittensor.Neuron(config)
    
    # Init a trainable request router.
    router = bittensor.Router(x_dim = input_dimension, key_dim = 100, topk = 10)
    
    # Build local network.
    net = Net()
    
    # Subscribe the local network to the network
    neuron.subscribe(net)
    
    # Start the neuron backend.
    neuron.start()
    
    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir='./runs/' + config.neuron_key)
    
    # Build the optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def train(epoch, global_step):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Flatten cifar inputs to [batch_size, input_dimension]
            inputs_flatten = torch.flatten(inputs, start_dim=1)
            
            # Query the remote network.
            synapses = neuron.synapses() # Returns a list of synapses on the network. [...]
            requests, scores = router.route(inputs_flatten, synapses) # routes inputs to network.
            responses = neuron(requests, synapses) # Makes network calls.
            outputs = router.join(responses) # Joins responses based on scores.
            
            # Loss computation. 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logstr = '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000)
                logger.info(logstr)
                running_loss = 0.0

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
