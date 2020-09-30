from loguru import logger

import argparse
import math
import time
import torch
import torchvision
import copy
import traceback

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import bittensor
from bittensor import bittensor_pb2

class CIFAR(bittensor.Synapse):
    def __init__(self, config: bittensor.Config):
        super(CIFAR, self).__init__(config)
        model_config = self.DPN26()
        in_planes, out_planes = model_config['in_planes'], model_config['out_planes']
        num_blocks, dense_depth = model_config['num_blocks'], model_config['dense_depth']

        # Image encoder
        self._transform = bittensor.utils.batch_transforms.Normalize((0.1307,), (0.3081,))
        self._adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        # Main Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=1)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear((out_planes[3] * 4)+(((num_blocks[3]+1) * 4)*dense_depth[3]), 10)

        # Distill Network
        self.dist_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dist_bn1 = nn.BatchNorm2d(64)
        self.dist_layer1 = copy.deepcopy(self.layer1)
        self.dist_layer2 = copy.deepcopy(self.layer2)
        self.dist_layer3 = copy.deepcopy(self.layer3)
        self.dist_layer4 = copy.deepcopy(self.layer4)

        # Logit Network
        self.logit_layer1 = nn.Linear((out_planes[3] * 4)+(((num_blocks[3]+1) * 4)*dense_depth[3]), 512)
        self.logit_layer2 = nn.Linear(512, 256)
        self.logit_layer3 = nn.Linear(256, 10)


    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(self.Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def DPN92(self):
        cfg = {
            'in_planes': (96,192,384,768),
            'out_planes': (256,512,1024,2048),
            'num_blocks': (3,4,20,3),
            'dense_depth': (16,32,24,128)
        }
        return cfg

    def DPN26(self):
        cfg = {
            'in_planes': (96,192,384,768),
            'out_planes': (256,512,1024,2048),
            'num_blocks': (2,2,2,2),
            'dense_depth': (16,32,24,128)
        }
        return cfg

    def distill(self, x):
        x = x.to(self.device)
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.dist_bn1(self.dist_conv1(x)))
        x = self.dist_layer1(x)
        x = self.dist_layer2(x)
        x = self.dist_layer3(x)
        x = self.dist_layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def logits (self, x):
        x = F.relu(self.logit_layer1 (x))
        x = F.relu(self.logit_layer2 (x))
        x = F.log_softmax(x)
        return x

    def forward(self, x, y = None):        
        y = self.distill(x) if y == None else y
        x = x.to(self.device)
        y = y.to(self.device)
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def encode_image(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._transform(inputs)
        inputs = self._adaptive_pool(inputs)
        return torch.flatten(inputs, start_dim=1)

    @property
    def input_shape(self):
        return [-1, 3072]

    @property
    def output_shape(self):
        return [-1, 10]

    class Bottleneck(nn.Module):
        def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
            super(CIFAR.Bottleneck, self).__init__()
            self.out_planes = out_planes
            self.dense_depth = dense_depth

            self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
            self.bn2 = nn.BatchNorm2d(in_planes)
            self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)

            self.shortcut = nn.Sequential()
            if first_layer:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes + dense_depth)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            x = self.shortcut(x)
            d = self.out_planes
            out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
            out = F.relu(out)
            return out


def main(hparams):
    config = bittensor.Config(hparams)
    batch_size_train = 32
    batch_size_test = 16
    input_dimension = 3072
    output_dimension = 10
    learning_rate = 0.01
    momentum = 0.9
    log_interval = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log/data/model paths.
    trial_id =  'cifar-' + str(time.time()).split('.')[0]
    data_path = "data/datasets/"
    log_dir = 'data/' + trial_id + '/logs/'
    model_path = 'data/' + trial_id + '/model.torch'

    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir=log_dir)

    # Load CIFAR
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size_test, shuffle=False, num_workers=2)

    # Build local synapse to serve on the network.
    model = CIFAR(config) # Synapses take a config object.
    model.to( device ) # Set model to device.

    if device == 'cuda':
        model = torch.nn.DataParallel(model)

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
    router = bittensor.Router(x_dim = input_dimension, key_dim = 100, topk = 10)

    # Build the optimizer.
    criterion = nn.CrossEntropyLoss()
    params = list(router.parameters()) + list(model.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train(model, epoch, global_step):
        running_loss = 0.0
        for i, (image_inputs, targets) in enumerate(trainloader, 0):

            # zero the parameter gradients
            optimizer.zero_grad()

            # Encode PIL images to tensors.
            encoded_inputs = model.encode_image(image_inputs).to(device)

            # Target to tensor
            targets = torch.LongTensor(targets).to(device)

            # Query the remote network.
            synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network. [...]
            requests, scores = router.route( synapses, encoded_inputs, image_inputs ) # routes inputs to network.
            responses = dendrite.forward_image ( synapses, requests ) # Makes network calls.
            network_input = router.join(responses) # Joins responses based on scores.

            # Run distilled model.
            dist_output = model.distill(image_inputs)
            dist_loss = F.mse_loss(dist_output, network_input.detach())

            # Distill loss
            student_output = model.forward(encoded_inputs, dist_output)
            student_logits = model.logits(student_output)
            student_loss = F.nll_loss(student_logits, targets)

            # Query the local network.
            local_output = model.forward(encoded_inputs, network_input)
            local_logits = model.logits(local_output)
            target_loss = criterion(local_logits, targets)
            
            torch.nn.utils.clip_grad_norm_(router.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            loss = (target_loss + dist_loss + student_loss)
            loss.backward()
            optimizer.step()
            global_step += 1

            # Set network weights.
            weights = metagraph.getweights(synapses).to(device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            metagraph.setweights(synapses, weights)

            if i % log_interval == 0:

                writer.add_scalar('n_peers', len(metagraph.peers),
                                  global_step)
                writer.add_scalar('n_synapses', len(metagraph.synapses),
                                  global_step)
                writer.add_scalar('train_loss', float(loss.item()),
                                  global_step)

                n = len(trainloader.dataset)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDistill Loss: {:.6f}\tStudent Loss: {:.6f}\tnP|nS: {}|{}'.format(
                    epoch, (i * batch_size_train), n, (100. * i * batch_size_train)/n, target_loss.item(), dist_loss.item(), student_loss.item(), len(metagraph.peers), 
                            len(metagraph.synapses)))

            # Empty device cache
            if device == 'cuda':
                torch.cuda.empty_cache()

    def test(model):

        # Turns off Dropoutlayers, BatchNorm etc.
        model.eval()
        # Turns off gradient computation for inference speed up.
        with torch.no_grad():
            loss = 0.0
            correct = 0.0
            for batch_idx, (image_inputs, targets) in enumerate(testloader, 0):
                # Set data to correct device.
                # Encode PIL images to tensors.
                encoded_inputs = model.encode_image(image_inputs).to(device)

                # Targets to Tensor
                targets = torch.LongTensor(targets).to(device)

                # Measure loss.
                embedding = model.forward( encoded_inputs, model.distill ( encoded_inputs ) )
                logits = model.logits(embedding)
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

    epoch = 1
    global_step = 0
    best_test_loss = math.inf
    try:
        while True:
            # Train model
            train( model, epoch, global_step )

            # Test model.
            test_loss, test_accuracy = test( model )

            # Save best model.
            if test_loss < best_test_loss:
                # Update best_loss.
                best_test_loss = test_loss

                # Save the best local model.
                logger.info('Saving model: epoch: {}, loss: {}, path: {}', model_path, epoch, test_loss)
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'test_loss': test_loss}, model_path)

            epoch += 1

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        metagraph.stop()
        axon.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
