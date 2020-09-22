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

class CIFAR(bittensor.Synapse):
        
    def __init__(self):
        super(CIFAR, self).__init__()
        
        # Main Network
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # Distill Network
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def distill(self, x):   
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward(self, x, y = None):
        y = self.distill(x) if y == None else y   
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

    batch_size_train = 50
    batch_size_test = 50
    input_dimension = 3072
    output_dimension = 10
    log_interval = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="~/tmp/bittensor/data/", train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_train,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="~/tmp/bittensor/data/", train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size_test,
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
    model = CIFAR()
    model.to(device)
    
    # Subscribe the local network to the network
    neuron.subscribe(model)
    
    # Start the neuron backend.
    neuron.start()
    
    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir='./tmp/bittensor/runs/' + config.neuron_key)
    
    # Build the optimizer.
    criterion = nn.CrossEntropyLoss()
    params = list(router.parameters()) + list(model.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    def train(model, epoch, global_step):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs.to(device)
            targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Flatten cifar inputs to [batch_size, input_dimension]
            inputs_flatten = torch.flatten(inputs, start_dim=1)
            
            # Query the remote network.
            synapses = neuron.synapses() # Returns a list of synapses on the network. [...]
            requests, scores = router.route(inputs_flatten, synapses) # routes inputs to network.
            responses = neuron(requests, synapses) # Makes network calls.
            network_outputs = router.join(responses) # Joins responses based on scores.
            
             # Run distilled model.
            dist_output = model.distill(inputs)
            dist_loss = F.kl_div(dist_output, network_outputs.detach())
            
            # Query the local network.
            local_output = model.forward(inputs, network_outputs)
            target_loss = F.nll_loss(local_output, targets)
            
            loss = (target_loss + dist_loss)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = neuron.getweights(synapses).to(device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            neuron.setweights(synapses, weights)

            if i % log_interval == 0:

                writer.add_scalar('n_peers', len(neuron.metagraph.peers),
                                  global_step)
                writer.add_scalar('n_synapses', len(neuron.metagraph.synapses),
                                  global_step)
                writer.add_scalar('Loss/train', float(loss.item()),
                                  global_step)
            
                n = len(trainloader.dataset)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDistill Loss: {:.6f}'.format(
                    epoch, (i * batch_size_train), n, (100. * i * batch_size_train)/n, loss.item(), dist_loss.item()))

    def test(model):
        
        # Turns off Dropoutlayers, BatchNorm etc.
        model.eval()
        
        # Turns off gradient computation for inference speed up.
        with torch.no_grad():
            
            loss = 0.0
            correct = 0.0
            for i, data in enumerate(testloader, 0):
                # Set data to correct device.
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Measure loss.
                logits = model( inputs )
                loss += F.nll_loss(logits, targets, size_average=False).item()
                
                # Count accurate predictions.
                max_logit = logits.data.max(1, keepdim=True)[1]
                correct += max_logit.eq( targets.data.view_as(max_logit) ).sum()

        # Log results.
        n = len(testloader.dataset)
        loss /= n
        accuracy = (100. * correct) / n
        logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))        
        return loss, accuracy

    epoch = 0
    global_step = 0
    try:
        while True:
            # Train model
            train( model, epoch, global_step )
            
            # Test model.
            test_loss, test_accuracy = test( model )
            
            epoch += 1
            
    except Exception as e:
        logger.error(e)
        neuron.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
