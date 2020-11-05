"""GPT2 Language Modelling 

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python examples/gpt/main.py

"""
import bittensor
from bittensor.synapses.gpt2 import GPT2LMSynapse, GPT2MLMConfig

import argparse
from datasets import load_dataset
from loguru import logger
import math
import random
import time
import torch

def nextbatch(data, batch_size, tokenizer):
    """ Returns a random batch of sentences from text dataset.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            batch_inputs torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
    """
    batch_text = []
    for _ in range(batch_size):
        batch_text.append(data[random.randint(0, len(data))]['text'])
    batch_inputs = tokenizer(batch_text, return_tensors='pt', padding=True)['input_ids']
    return batch_inputs
            
def main():
    argparser = argparse.ArgumentParser()

    # Args
    learning_rate = 0.01 
    mini_batch_size = 10
    full_batch_size = 100
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
    bittensor.serve( model.deepcopy() )

    # Dataset: 74 million sentences pulled from books.
    dataset = load_dataset('bookcorpus')['train']
  
    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    def train(dataset, model):
        model.train()  # Turn on the train mode.

        full_step = 0
        mini_step = 0
        best_loss = math.inf
        while True:

            # Run full training step.
            optimizer.zero_grad()
            full_loss = 0.0
            n_mini_steps = int(full_batch_size/mini_batch_size)
            for mini_step in range(n_mini_steps):
                # Next mini batch.
                inputs = nextbatch(dataset, mini_batch_size, bittensor.__tokenizer__)
            
                # Compute full pass and get loss with a network query.
                output = model(inputs.to(device), training = True, remote = True)
            
                # Aggregate grads.
                loss = output.loss.item()
                loss = loss / n_mini_steps # Fixes the learning rate over multiple steps.
                loss.backward()

                # Mini log.
                full_loss += output.local_target_loss.item() / n_mini_steps
                logger.info('Block: {}, Full Step: {}, Mini Step: [{}/{} ({:.1f}%)]\t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                    bittensor.height(), full_step, mini_step, n_mini_steps, float(mini_step * 100)/float(n_mini_steps), output.remote_target_loss.item(), output.local_target_loss.item(), output.distillation_loss.item()))
             
            # Apply grads on Full step.
            full_step += 1
            optimizer.step()
            scheduler.step()

            # Serve next best model.
            if full_loss < best_loss:
                best_loss = full_loss
                copy = model.deepcopy() # Make model copy for serving.
                copy.eval() # Set to eval.
                bittensor.serve( copy ) # Serve to axon.
                logger.info('Serve model: Block {}, Full Step: {}, Loss: {}', bittensor.height(), full_step, best_loss)

    try:
        train(dataset, model)
    except Exception as e:
        logger.exception(e)
        bittensor.stop()
        

if __name__ == "__main__":
    main()