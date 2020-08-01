from bittensor import bittensor_pb2
import bittensor

import os, sys
import argparse
import math
import time

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer import TransformerModel
from dataset import Dataset

class TransformerSynapse(bittensor.Synapse):
    """ An bittensor endpoint trained on 28, 28 pixel images to detect handwritten characters.
    """
    def __init__(self, transformer):
        super(Net, self).__init__()
        self.transformer = transformer
        
    def indef(self):
        x_def = bittensor_pb2.TensorDef(
                    version = bittensor.PROTOCOL_VERSION,
                    shape = [-1, 784],
                    dtype = bittensor_pb2.FLOAT32,
                    requires_grad = True,
                )
        return x_def
    
    def outdef(self):
        y_def = bittensor_pb2.TensorDef(
                    version = bittensor.PROTOCOL_VERSION,
                    shape = [-1, 10],
                    dtype = bittensor_pb2.FLOAT32,
                    requires_grad = True,
                )
        return y_def
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = transformer.encode(x)
        return x

def main(hparams):
    
    # Args
    batch_size = 20
    eval_batch_size = 10
    bptt = 35
    
    dataset = Dataset()
    train_data = dataset.batchify(dataset.train_txt, batch_size)
    val_data = dataset.batchify(dataset.val_txt, eval_batch_size)
    test_data = dataset.batchify(dataset.test_txt, eval_batch_size)

    # Transformer model architecture
    ntokens = len(dataset.TEXT.vocab.stoi)  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    transformer = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)

    # Optimizer.
    criterion = nn.CrossEntropyLoss()  # loss function
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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
    neuron.subscribe(transformer)
    
    # Start the neuron backend.
    neuron.start()
    
    def train(dataset, transformer):
        model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(dataset.TEXT.vocab.stoi)
        for batch, i in enumerate(
                range(0,
                      train_data.size(0) - 1, dataset.bptt)):
            data, targets = dataset.get_batch(train_data, i)
            optimizer.zero_grad()
            
            # data 
            print (data.shape)
            
            # Flatten encoder inputs inputs
            inputs = data.view(-1, bptt, emsize)
            inputs = torch.flatten(inputs, start_dim=1)
            
            # Query the remote network.
            synapses = neuron.synapses() # Returns a list of synapses on the network.
            requests, scores = router.route(inputs, synapses) # routes inputs to network.
            responses = neuron(requests, synapses) # Makes network calls.
            remote = router.join(responses) # Joins responses based on scores.

            # Encode sequence inputs.
            encodings = transformer.encode(data)  # (seq_len, batch_size, embedding_size)
            
            
            

            # Get nodes from metagraph.
            # and map nodes to torch keys.
            axons = neuron.axons()  # List[bittensor_pb2.Node]))
            keys = keymap.toKeys(axons)  # (-1, key_dim)

            # Learning a map from the gate_inputs to keys
            # gates[i, j] = score for the jth key for input i
            gate_inputs = encodings.view(
                batch_size, x_dim)  # (batch_size, seq_len * embedding_size)
            gates = gate(gate_inputs, keys, topk=min(len(keys), topk))

            # Dispatch data to inputs for each key.
            # when gates[i, j] == 0, the key j does not recieve input i
            dispatch_inputs = data.view(batch_size,
                                        -1)  # (batch_size, sequence_length)
            dispatch = dispatcher.dispatch(dispatch_inputs,
                                           gates)  # List[(-1, seq_len)]

            # Query the network by mapping from keys to node endpoints.
            # results = list[torch.Tensor], len(results) = len(keys)
            axons = keymap.toAxons(keys)  # List[bittensor_pb2.Node]
            query = neuron(dispatch, axons)  # List[(-1, embedding_size)]

            # Join results using gates to combine inputs.
            results = dispatcher.combine(
                query, gates)  # (batch_size, seq_len * embedding_size)

            # Decode responses.
            results = results.view(
                -1, batch_size,
                emsize)  # (seq_len, batch_size, embedding_size)
            to_decode = results + encodings
            output = model.decode(
                to_decode)  # (target_len, batch_size, embedding_size)

            # Loss and optimizer step
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Update bittensor weights
            weights = neuron.getweights(axons)
            weights = (0.95) * weights + (0.05) * torch.mean(gates, dim=0)
            neuron.setweights(axons, weights)

            total_loss += loss.item()
            log_interval = 1
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, batch,
                          len(train_data) // dataset.bptt,
                          scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(dataset, model)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap',
                        default='',
                        type=str,
                        help='ip address of bootstrap metagraph')
    hparams = parser.parse_args()
    main(hparams)
