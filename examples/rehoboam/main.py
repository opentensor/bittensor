from bittensor import bittensor_pb2
import bittensor

import os, sys
import argparse
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer import TransformerModel
from dataset import Dataset
from loguru import logger


class TransformerSynapse(bittensor.Synapse):
    """ An bittensor endpoint trained on wiki corpus.
    """
    def __init__(self, transformer, ntokens):
        super(TransformerSynapse, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = transformer
        self.ntokens = ntokens
                
    def indef(self):
        x_def = bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = [700, 20],
                    dtype = bittensor_pb2.INT64,
                    requires_grad = True,
                )
        return [x_def]
    
    def outdef(self):
        y_def = bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = [700, 20],
                    dtype = bittensor_pb2.INT64,
                    requires_grad = True,
                )
        return [y_def]
    
    def forward(self, x):
        # Move x over to device, if any
        x = x.to(self.device)
        # Encode x
        x = self.transformer.encode(x)
        x = torch.flatten(x, start_dim=1)
        return x

def main(hparams):
    
    # Args
    batch_size = 20
    eval_batch_size = 20
    bptt = 6
    log_interval = 10
    
    dataset = Dataset(bptt)
    train_data = dataset.batchify(dataset.train_txt, batch_size)
    val_data = dataset.batchify(dataset.val_txt, eval_batch_size)
    test_data = dataset.batchify(dataset.test_txt, eval_batch_size)

    test_results_file = "rehoboam_test_results.txt"

    # Transformer model architecture
    ntokens = len(dataset.TEXT.vocab.stoi)  # the size of vocabulary
    emsize = 20  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    transformer = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)

    # bittensor:
    # Load bittensor config from hparams.
    config = bittensor.Config(hparams)
    
    # Build the neuron from configs.
    neuron = bittensor.Neuron(config)
    
    # Init a trainable request router.
    router = bittensor.Router(x_dim = dataset.bptt * emsize, key_dim = 100, topk = 10)
    
    # Build local network.
    net = TransformerSynapse(transformer, ntokens)
    
    # Subscribe the local network to the network
    neuron.subscribe(net)
    
    # Start the neuron backend.
    neuron.start()
    
    # Optimizer.
    criterion = nn.CrossEntropyLoss()  # loss function
    lr = 3.0 # learning rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    
    def train(dataset, transformer, epoch):
        transformer.train()  # Turn on the train mode
        total_loss = 0.
        global_step = 0
        start_time = time.time()
        ntokens = len(dataset.TEXT.vocab.stoi)
        for batch_idx, i in enumerate(range(0,train_data.size(0) - 1, dataset.bptt)):
            data, targets = dataset.get_batch(train_data, i)
            optimizer.zero_grad()

            # Flatten encoder inputs inputs
            inputs = data.view(-1, bptt, emsize)
            inputs = torch.flatten(inputs, start_dim=1)

            # Query the local network.
            #local = net(inputs)
            
            # Query the remote network.
            synapses = neuron.synapses() # Returns a list of synapses on the network.
            
            requests, scores = router.route(inputs.float(), synapses) # routes inputs to network.

            # Convert request indices back to type long()
            request_list = [*requests]
            request_list[0] = requests[0].type(torch.LongTensor)
            requests = *request_list,

            responses = neuron(requests, synapses) # Makes network calls.
            
            output = router.join(responses) # Joins responses based on scores.

            #local = net(inputs)
            #remote = output.view(-1, batch_size, emsize)

            # Train.
            #output = remote + local
            # Decode responses.

            output = net.transformer.decode(output.view(-1, batch_size, emsize))
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = neuron.getweights(synapses).to(net.device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            neuron.setweights(synapses, weights)
            
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tnP|nS: {}|{}'.format(
                            epoch, 
                            batch_idx * len(data), 
                            train_data.size(0) - 1,
                            100. * (batch_idx * len(data)) / train_data.size(0) - 1, 
                            loss.item(), 
                            len(neuron.metagraph.peers), 
                            len(neuron.metagraph.synapses)))
 
    def test(data_source):
        # Turn on evaluation mode
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i in range(0,
                      data_source.size(0) - 1, dataset.bptt):

                # Get batch
                data, targets = dataset.get_batch(data_source, i)
                
                # Query local network
                output = net(data)
                output = net.transformer.decode(output.view(-1, batch_size, emsize))

                output_flat = output.view(-1, ntokens)
                test_loss += len(data) * criterion(output_flat, targets).item()
                

        test_loss /= (len(data_source) - 1)
        test_result = 'Test set: Avg. loss: {:.4f}\n'.format(test_loss)
        #test_result = 'Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    test_loss, correct, 
        #    test_data.size(0),
        #    100. * correct / test_data.size(0))
        
        logger.info(test_result)
        
        if os.path.exists(test_results_file):
            append_write = 'a'
        else:
            append_write = 'w'
        
        outF = open(test_results_file, append_write)
        outF.write(test_result)
        outF.close()

        
    global_step = 0
    epoch = 0
    try:
        while True:
            train(dataset, net, epoch)
            test(test_data)
            scheduler.step()
            epoch += 1
    except Exception as e:
        logger.error(e)
        neuron.stop()
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
