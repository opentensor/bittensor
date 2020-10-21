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
            
def main(hparams):
    # Args
    learning_rate = 0.01 
    batch_size = 20
    epoch_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Bittensor.
    # Create background objects.
    # Connect the metagraph.
    # Start the axon server.
    config = bittensor.Config.from_hparams( hparams )
    logger.info(config)
    bittensor.init( config )
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
            
            loss = output['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                epoch, step, epoch_size, float(step * 100)/float(epoch_size), output['remote_target_loss'].item(), output['local_target_loss'].item(), output['distillation_loss'].item()))
            bittensor.tbwriter.add_scalar('train remote target loss', output['remote_target_loss'].item(), time.time())
            bittensor.tbwriter.add_scalar('train local target loss', output['local_target_loss'].item(), time.time())
            bittensor.tbwriter.add_scalar('train distilation loss', output['distillation_loss'].item(), time.time())
            bittensor.tbwriter.add_scalar('train loss', output['loss'].item(), time.time())

    epoch = 0
    try:
        while True:
            train(dataset, model, epoch)
            epoch += 1
    except Exception as e:
        logger.exception(e)
        bittensor.stop()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)