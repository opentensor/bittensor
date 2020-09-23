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
class Mnist(bittensor.Synapse):
    """ Bittensor endpoint trained on 28, 28 pixel images to detect handwritten characters.
    """
    def __init__(self, config: bittensor.Config):
        super(Mnist, self).__init__(config)
        
        # Main Network
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.average1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.average2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(120, 82)
        self.fc2 = nn.Linear(82, 10)
        
        # Distillation Network
        self.dist_conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.dist_average1 = nn.AvgPool2d(2, stride=2)
        self.dist_conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.dist_average2 = nn.AvgPool2d(2, stride=2)
        self.dist_conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        self.dist_fc1 = nn.Linear(120, 82)
        self.dist_fc2 = nn.Linear(82, 10)    
    
    @property
    def input_shape(self):
        return [-1, 784]
    
    @property
    def output_shape(self):
        return [-1, 10]
    
    def distill(self, x):
        x = x.view(-1, 1, 28, 28)
        x = torch.tanh(self.dist_conv1(x))
        x = self.dist_average1(x)
        x = torch.tanh(self.dist_conv2(x))
        x = self.dist_average2(x)
        x = torch.tanh(self.dist_conv3(x))
        x = x.view(-1, x.shape[1])
        x = F.relu(self.dist_fc1(x))
        x = F.relu(self.dist_fc2(x))
        return x
    
    def forward (self, x, y = None):
        y = self.distill(x) if y == None else y
        x = x.view(-1, 1, 28, 28)
        x = torch.tanh(self.conv1(x))
        x = self.average1(x)
        x = torch.tanh(self.conv2(x))
        x = self.average2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, x.shape[1])
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x + y)
        return x

def main(hparams):

    # Training params.
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.1
    momentum = 0.9
    log_interval = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load bittensor config from hparams.
    config = bittensor.Config( hparams )

    # Dataset.
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root="~/tmp/bittensor/data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])),
                                               batch_size=batch_size_train,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='~/tmp/bittensor/data/',
        train=False, 
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])),
                                            batch_size=batch_size_test, 
                                            shuffle=True)
    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir='~/tmp/bittensor/runs/' + config.neuron_key)
    
    # Build local synapse to serve on the network.
    model = Mnist(config) # Synapses take a config object.
    model.to( device ) # Set model to device.
    
    # Build and start the metagraph background object.
    # The metagraph is responsible for connecting to the blockchain
    # and finding the other neurons on the network.
    metagraph = bittensor.Metagraph( config )
    metagraph.subscribe( model ) # Adds the synapse to the metagraph.
    metagraph.start() # Starts the metagraph gossip threads.
    
    # Build and start the Axon server.
    # The axon server serves the synapse objects 
    # allowing other neurons to make queries through a dendrite.
    axon = bittensor.Axon( config )
    axon.serve( model ) # Makes the synapse available on the axon server.
    axon.start() # Starts the server background threads. Must be paired with axon.stop().
    
    # Build the dendrite and router. 
    # The dendrite is a torch object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    dendrite = bittensor.Dendrite( config )
    router = bittensor.Router(x_dim = 784, key_dim = 100, topk = 10)
        
    # Build the optimizer.
    params = list(router.parameters()) + list(model.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)

    def train(model, epoch, global_step):
        model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            # Set data device.
            data = data.to(device)
            target = target.to(device)
            
            # Query the remote network.
            # Flatten mnist inputs for routing.
            inputs = torch.flatten( data, start_dim=1 )
            synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
            requests, scores = router.route( synapses, inputs ) # routes inputs to network.
            responses = dendrite ( synapses, requests ) # Makes network calls.
            network_input = router.join( responses ) # Joins responses based on scores.
            
            # Run distilled model.
            dist_output = model.distill(inputs)
            dist_loss = F.kl_div(dist_output, network_input.detach())
            
            # Query the local network.
            local_output = model.forward(inputs, network_input)
            target_loss = F.nll_loss(local_output, target)
            
            loss = (target_loss + dist_loss)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = metagraph.getweights(synapses).to(device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            metagraph.setweights(synapses, weights)

            if batch_idx % log_interval == 0:
                writer.add_scalar('n_peers', len(metagraph.peers),
                                  global_step)
                writer.add_scalar('n_synapses', len(metagraph.synapses),
                                  global_step)
                writer.add_scalar('Loss/train', float(loss.item()),
                                  global_step)
            
                n = len(train_loader.dataset)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDistill Loss: {:.6f}'.format(
                    epoch, (batch_idx * batch_size_train), n, (100. * batch_idx * batch_size_train)/n, loss.item(), dist_loss.item()))

    def test(model):
        
        # Turns off Dropoutlayers, BatchNorm etc.
        model.eval()
        
        # Turns off gradient computation for inference speed up.
        with torch.no_grad():
            
            loss = 0.0
            correct = 0.0
            for data, target in test_loader:
                # Set data to correct device.
                data = data.to(device)
                target = target.to(device)
                
                # Measure loss.
                logits = model( data )
                loss += F.nll_loss(logits, target, size_average=False).item()
                
                # Count accurate predictions.
                max_logit = logits.data.max(1, keepdim=True)[1]
                correct += max_logit.eq( target.data.view_as(max_logit) ).sum()

        # Log results.
        n = len(test_loader.dataset)
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
            
            # TODO (const): save(model)
            # TODO (const): axon.serve(model)
            epoch += 1
            
    except Exception as e:
        logger.error(e)
        metagraph.stop()
        axon.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
