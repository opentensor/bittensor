"""Training a CIFAR Neuron.

This file demonstrates a training pipeline for an CIFAR Neuron.

Example:
        $ python examples/cifar/main.py
"""

import bittensor
from bittensor.synapses.dpn import DPNSynapse, DPNConfig
from bittensor.utils.model_utils import ModelToolbox

import argparse
from loguru import logger
import math
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import traceback

def main(hparams):
     
    # Additional training params.
    batch_size_train = 32
    batch_size_test = 16
    learning_rate = 0.001
    momentum = 0.9
    log_interval = 10
    epoch = 0
    global_step = 0
    best_test_loss = math.inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Setup Bittensor.
    # Create background objects.
    # Connect the metagraph.
    # Start the axon server.
    config = bittensor.Config.from_hparams( hparams )
    logger.info(config)
    bittensor.init( config )
    bittensor.start()
    
    # Build local synapse to serve on the network.
    model_config = DPNConfig()
    model = DPNSynapse( model_config )
    model.to( device ) # Set model to device.
    bittensor.serve( model.deepcopy() )

    # Build the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10.0, gamma=0.1)

    # Load (Train, Test) datasets into memory.
    train_data = torchvision.datasets.CIFAR10(
        root=config.datapath, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.CIFAR10(root=config.datapath, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, shuffle=False, num_workers=2)

    # Train loop: Single threaded training of MNIST.
    def train(model, epoch, global_step):
        # Turn on Dropoutlayers BatchNorm etc.
        model.train()
        for batch_idx, (images, targets) in enumerate(trainloader):
            # Clear gradients.
            optimizer.zero_grad()

            # Forward pass.
            images = images.to(device)
            targets = torch.LongTensor(targets).to(device)
            output = model(images, targets, remote = True)

            # Backprop.
            loss = output['loss']
            loss.backward()
            optimizer.step()
            global_step += 1
                            
            # Logs:
            if batch_idx % log_interval == 0:            
                n = len(train_data)
                max_logit = output['remote_target'].data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                loss_item  = output['remote_target_loss'].item()
                processed = ((batch_idx + 1) * batch_size_train)
                
                progress = (100. * processed) / n
                accuracy = (100.0 * correct) / batch_size_train
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}', 
                    epoch, processed, n, progress, loss_item, accuracy)

    # Test loop.
    # Evaluates the local model on the hold-out set.
    # Returns the test_accuracy and test_loss.
    def test( model: bittensor.Synapse ):
        
        # Turns off Dropoutlayers, BatchNorm etc.
        model.eval()
        
        # Turns off gradient computation for inference speed up.
        with torch.no_grad():
        
            loss = 0.0
            correct = 0.0
            for _, (images, labels) in enumerate(testloader):                
               
                images = images.to(device)

                # Labels to Tensor
                labels = torch.LongTensor(labels).to(device)
                
                # Compute full pass and get loss.
                outputs = model (images, labels, remote = False)
                loss = loss + outputs['loss']
                
                # Count accurate predictions.
                max_logit = outputs['local_target'].data.max(1, keepdim=True)[1]
                correct += max_logit.eq( labels.data.view_as(max_logit) ).sum()
        
        # # Log results.
        n = len(test_data)
        loss /= n
        accuracy = (100. * correct) / n
        logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))        
        return loss, accuracy
    
    
    while True:
        try:
            # Train model
            train( model, epoch, global_step )
            scheduler.step()
            # Test model.
            test_loss, _ = test( model )
        
            # Save best model. 
            if test_loss < best_test_loss:
                # Update best loss.
                best_test_loss = test_loss
                
                # Save the best local model.
                logger.info('Serving / Saving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, config.logdir + '/model.torch')
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'test_loss': test_loss}, config.logdir + '/model.torch' )
                bittensor.serve( model.deepcopy() )

            epoch += 1

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            bittensor.stop()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
