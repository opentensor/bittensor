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
    def __init__(self, config, transformer, ntokens):
        super(TransformerSynapse, self).__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = transformer
        self.ntokens = ntokens
        
    @property
    def input_shape(self):
        return [-1, 20]
    
    @property
    def input_dtype(self):
        return bittensor_pb2.INT64
    
    @property
    def output_shape(self):
        return [-1, 120]
    
    @property
    def output_dtype(self):
        return bittensor_pb2.FLOAT32
   
    def forward(self, x):
        # Move x over to device, if any
        x = x.to(self.device)
        # Encode x
        x = self.transformer.encode(x)
        x = torch.flatten(x, start_dim=1)
        return x

def main(hparams):
    
    # Args
    batch_size = 50
    eval_batch_size = 20
    bptt = 200
    log_interval = 10
    config = bittensor.Config( hparams )
    
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

    # Build local network.
    net = TransformerSynapse(config, transformer, ntokens)
    
    # Build and start the metagraph background object.
    # The metagraph is responsible for connecting to the blockchain
    # and finding the other neurons on the network.
    metagraph = bittensor.Metagraph( config )
    metagraph.subscribe( net ) # Adds the synapse to the metagraph.
    metagraph.start() # Starts the metagraph gossip threads.
    
    # Build and start the Axon server.
    # The axon server serves the synapse objects 
    # allowing other neurons to make queries through a dendrite.
    axon = bittensor.Axon( config )
    axon.serve( net ) # Makes the synapse available on the axon server.
    axon.start() # Starts the server background threads. Must be paired with axon.stop().
    
    # Build the dendrite and router. 
    # The dendrite is a torch object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    dendrite = bittensor.Dendrite( config )
    router = bittensor.Router(x_dim = batch_size, key_dim = 100, topk = 10)
    
    # Optimizer.
    criterion = nn.CrossEntropyLoss()  # loss function
    lr = 3.0 # learning rate
    params = list(router.parameters()) + list(net.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    def train(dataset, transformer, epoch):
        transformer.train()  # Turn on the train mode
        total_loss = 0.
        global_step = 0
        start_time = time.time()
        ntokens = len(dataset.TEXT.vocab.stoi)
        for batch_idx, i in enumerate(range(0,train_data.size(0) - 1, bptt)):
            data, targets = dataset.get_batch(train_data, i)
            optimizer.zero_grad()

            # Query the remote network.
            synapses = metagraph.get_synapses(1000) # Returns a list of synapses on the network.
            requests, scores = router.route(synapses, data.float()) # routes inputs to network.

            # Convert request indices back to type long()
            request_list = [*requests]
            request_list[0] = requests[0].type(torch.LongTensor)
            requests = *request_list,

            responses = dendrite(synapses, requests) # Makes network calls.
            
            output = router.join(responses) # Joins responses based on scores.
            # Since model returns encoded version, we should decode here.
            output = net.transformer.decode(output.view(-1, batch_size, emsize))
            loss = criterion(output.view(-1, ntokens), targets)
            torch.nn.utils.clip_grad_norm_(router.parameters(), 0.5)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = metagraph.getweights(synapses).to(net.device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            metagraph.setweights(synapses, weights)
            
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tnP|nS: {}|{}'.format(
                            epoch, 
                            batch_idx * batch_size, 
                            train_data.size(0) - 1,
                            100. * (batch_idx * batch_size) / train_data.size(0) - 1, 
                            loss.item(), 
                            len(metagraph.peers), 
                            len(metagraph.synapses)))
 
    def test(data_source):
        # Turn on evaluation mode
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i in range(0,
                      data_source.size(0) - 1, bptt):
                
                # Get batch
                data, targets = dataset.get_batch(data_source, i)
                
                # Query local network
                output = net(data)
                output = net.transformer.decode(output.view(-1, eval_batch_size, emsize))

                output_flat = output.view(-1, ntokens)
                test_loss += len(data) + criterion(output_flat, targets).item()
                

        test_loss /= (data_source.size(0) - 1)
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
        logger.exception(e)
        metagraph.stop()
        axon.stop()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
