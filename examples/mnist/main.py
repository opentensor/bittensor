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

class Net(bittensor.Synapse):
    """ An bittensor endpoint trained on 28, 28 pixel images to detect handwritten characters.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1).to(self.device)
        self.average1 = nn.AvgPool2d(2, stride=2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1).to(self.device)
        self.average2 = nn.AvgPool2d(2, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1).to(self.device)
        self.fc1 = nn.Linear(120, 82).to(self.device)
        self.fc2 = nn.Linear(82, 10).to(self.device)
        
        
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
        x = x.to(self.device)
        x = x.view(-1, 1, 28, 28).to(self.device)
        x = torch.tanh(self.conv1(x)).to(self.device)
        x = self.average1(x).to(self.device)
        x = torch.tanh(self.conv2(x)).to(self.device)
        x = self.average2(x).to(self.device)
        x = torch.tanh(self.conv3(x)).to(self.device)
        x = x.view(-1, x.shape[1]).to(self.device)
        x = F.dropout(x, training=self.training).to(self.device)
        x = F.relu(self.fc1(x)).to(self.device)
        x = F.relu(self.fc2(x)).to(self.device)
        x = F.log_softmax(x, dim=1).to(self.device)
        return x


def main(hparams):

    # Training params.
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.1
    momentum = 0.9
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

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='~/tmp/',
        train=False, 
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])),
                                            batch_size=batch_size_test, 
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
    #writer = SummaryWriter(log_dir='./runs/' + config.neuron_key)
    
    # Build the optimizer.
    optimizer = optim.SGD(router.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    def train(epoch, global_step):
        net.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Flatten mnist inputs
            inputs = torch.flatten(data, start_dim=1).to(net.device)
            # Query the remote network.
            synapses = neuron.synapses() # Returns a list of synapses on the network.
            requests, scores = router.route(inputs, synapses) # routes inputs to network.
            responses = neuron(requests, synapses) # Makes network calls.
            output = router.join(responses) # Joins responses based on scores.
            
            # Query the local network.
            #local = net(inputs)

            loss = F.nll_loss(output, target.to(net.device))
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = neuron.getweights(synapses).to(net.device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            neuron.setweights(synapses, weights)

            if batch_idx % log_interval == 0:
                #writer.add_scalar('n_peers', len(neuron.metagraph.peers),
                #                  global_step)
                #writer.add_scalar('n_synapses', len(neuron.metagraph.synapses),
                #                  global_step)
                #writer.add_scalar('Loss/train', float(loss.item()),
                #                    global_step)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tnP|nS: {}|{}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), len(neuron.metagraph.peers), len(neuron.metagraph.synapses)))

    def test( model ):
        
        # Turns off BatchNormLayer, DropOutlayers ...
        model.eval()
    
        # Turns off gradient computation.
        with torch.no_grad():
        
            n_correct = 0
            test_loss = 0
        
            for data, target in test_loader:
            
                data = data.to( model.device )
                targets = target.to( model.device )
                
                # Sum of: negative loglikihood (logits, targets)
                logits = model( data )
                loss = F.nll_loss( logits, targets, size_average=False )
                test_loss += loss
    
                # Sum of: argmax ( logits ) == target
                max_logit = logits.data.max( 1, keepdim=True )[1].to(net.device)
                target = targets.data.view_as( max_logit )
                n_correct += max_logit.eq( target ).sum()


            test_size = len(test_loader.dataset)
            test_loss /= test_size
            logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                        .format( test_loss, n_correct, test_size, 100. * n_correct / test_size))        
    
    step = 0
    try:
        
        # Alternating (Train, Test) loop,
        # Produces saved models into: 
        #       ~/tmp/bittensor/models/$CONFIG.UID$
        for epoch in range( 2 ):
            
            model = train(epoch, step)
            test ( model )
            
    except Exception as e:
        logger.error(e)
        neuron.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
