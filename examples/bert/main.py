"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python examples/bert/main.py

"""
import bittensor
from bittensor.synapses.bert import BertSynapseConfig, BertMLMSynapse

import argparse
from datasets import load_dataset
from loguru import logger
import random
import torch
import transformers
from transformers import DataCollatorForLanguageModeling

def mlm_batch(data, batch_size, tokenizer, collator):
    """ Returns a random batch from text dataset with 50 percent NSP.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            tensor_batch torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
            labels torch.Tensor (batch_size, sequence_length)
    """
    batch_text = []
    for _ in range(batch_size):
        batch_text.append(data[random.randint(0, len(data))]['text'])

    # Tokenizer returns a dict { 'input_ids': list[], 'attention': list[] }
    # but we need to convert to List [ dict ['input_ids': ..., 'attention': ... ]]
    # annoying hack...
    tokenized = tokenizer(batch_text)
    tokenized = [dict(zip(tokenized,t)) for t in zip(*tokenized.values())]

    # Produces the masked language model inputs aw dictionary dict {'inputs': tensor_batch, 'labels': tensor_batch}
    # which can be used with the Bert Language model. 
    collated_batch =  collator(tokenized)
    return collated_batch['input_ids'], collated_batch['labels']
            
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
    model_config = BertSynapseConfig()
    model = BertMLMSynapse(model_config)
    model.to(device)
    bittensor.serve( model )

    # Dataset: 74 million sentences pulled from books.
    # The collator accepts a list [ dict{'input_ids, ...; } ] where the internal dict 
    # is produced by the tokenizer.
    dataset = load_dataset('bookcorpus')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bittensor.__tokenizer__, mlm=True, mlm_probability=0.15
    )

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    def train(dataset, model, epoch):
        model.train()  # Turn on the train mode.
        optimizer.zero_grad() # Zero out lingering gradients.

        step = 0
        while step < epoch_size:
            # Next batch.
            inputs, labels = mlm_batch(dataset['train'], batch_size, bittensor.__tokenizer__, data_collator)
            
            # Compute full pass and get loss with a network query.
            output = model( inputs.to(device), labels.to(device), remote = True)
            
            output.loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Network Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
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
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)