"""GPT2 Language Modelling 

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python examples/gpt/main.py

"""
import bittensor
from bittensor.synapses.gpt2 import GPT2LMSynapse, GPT2MLMConfig, nextbatch

import argparse
from datasets import load_dataset
from loguru import logger
import torch

def main():
    argparser = argparse.ArgumentParser()

    # Args
    learning_rate = 0.01 
    batch_size = 20
    epoch_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Bittensor.
    # Create background objects.
    # Connect the metagraph.
    # Start the axon server.
    bittensor.init(argparser)
    bittensor.start()

    # Build Synapse
    model_config = GPT2MLMConfig()  
    model = GPT2LMSynapse(model_config)
    model.to(device)
    bittensor.serve( model )

    # Dataset: 74 million sentences pulled from books.
    dataset = load_dataset('bookcorpus')['train']
  
    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    def train(dataset, model, epoch):
        model.train()  # Turn on the train mode.
        optimizer.zero_grad() # Zero out lingering gradients.

        step = 0
        while step < epoch_size:
            # Next batch.
            inputs = nextbatch(dataset, batch_size, bittensor.__tokenizer__)
            
            # Compute full pass and get loss with a network query.
            output = model(inputs.to(device), training = True, remote = True)
            
            output.loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                epoch, step, epoch_size, float(step * 100)/float(epoch_size), output.remote_target_loss.item(), output.local_target_loss.item(), output.distillation_loss.item()))

    epoch = 0
    try:
        while True:
            train(dataset, model, epoch)
            epoch += 1
    except Exception as e:
        logger.exception(e)
        bittensor.stop()
        


if __name__ == "__main__":
    main()