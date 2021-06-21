import argparse
import random
from numpy import average
import bittensor
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from nuclei.ffnn import FFNNNucleus
from loguru import logger
from torchvision import datasets, transforms
torch.autograd.set_detect_anomaly(False)

mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
mnist_train = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=False)

MAX_DEPTH: int = 2
CURRENT_POS: int = None 

class Neuron:
    def __init__( self, position:int, wallet: bittensor.Wallet, endpoint: bittensor.Endpoint, child: bittensor.Endpoint ):
        
        # chain position
        self.position = position

        # wallet.
        self.wallet = wallet

        # Local info.
        self.endpoint = endpoint
        
        # Child to call forward on.
        self.child = child
                
        # Axon RPC server.
        # We attach the forward and backward passes to this miner class.
        # When this miner recieves a Forward/Backward request it calls these functions
        self.axon = bittensor.axon ( 
            wallet = self.wallet,
            local_port = self.endpoint.port,
            forward_callback = self.forward,
            backward_callback = self.backward
        )
        self.axon.start()
        
        # Dendrite RPC Client.
        # Differentiable RPC function which calls Forward and Backward 
        # on passes endpoints.
        self.dendrite = bittensor.dendrite( 
            wallet = self.wallet 
        )
        
        # Torch NN Module with remote_forward and local_forwadd functions.
        # plus a routing function.
        self.nucleus = FFNNNucleus ( 
            routing_callback = self.route 
        )
        
        # Base Torch optimizer.
        self.optimizer = torch.optim.AdamW(self.nucleus.parameters(), lr = 0.01, betas = (0.9, 0.95) )

    # Tear down the axon serving endpoint.
    def __del__(self):
        self.axon.stop()
                
    # Function is called by the nucleus to query child and get responses.
    def route( self, inputs: torch.FloatTensor, query: torch.FloatTensor ) -> torch.FloatTensor:
        
        # Is this a leaf node.
        if self.child == None:
            responses = torch.zeros( [inputs.shape[0], inputs.shape[1], bittensor.__network_dim__] )
        
        # Otherwise, makes differentiable calls.
        else:
            # Takes a list of endpoints and a list of inputs
            # Sends inputs to endpoints.
            responses, return_codes = self.dendrite.forward_image (
                endpoints = self.child, 
                inputs = inputs,
                timeout = 1,
                requires_grad = False
            )
            
        return responses
    
    # Function which is called when this miner recieves a forward request from a dendrite.
    def forward ( self, pubkey:str, images: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        # Call nucleus (locally, i.e. using the distillation model instead of calling the child)
        # return the last hidden layer.  
        # images: remove sequence dimension from images.
        # images.shape = [batch_size, channels, rows, cols] 
        images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4])

        # hidden: hidden layer using local_contextcontext for local computation only.
        # hidden.shape = [batch_size, __network_dim__] 
        if (CURRENT_POS - self.position) >= MAX_DEPTH:
            hidden = self.nucleus.local_forward ( images = images ).local_hidden
        else:
            hidden = self.nucleus.remote_forward ( images = images ).remote_hidden
        
        # hidden: re-add sequence dimension to outputs.
        # hidden.shape = [batch_size, sequence_dim, __network_dim__] 
        hidden = hidden.view(images.shape[0], images.shape[1], bittensor.__network_dim__)

        return hidden

    # Function which is called when this miner recieves a backward request.
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        inputs_x.requires_grad = True
        with torch.enable_grad():
            outputs_y = self.nucleus.forward_image(images = inputs_x)
            grads_dx = torch.autograd.grad (
                outputs = outputs_y, 
                inputs = inputs_x, 
                grad_outputs = grads_dy, 
                only_inputs = True,
                create_graph = False, 
                retain_graph = False
            )
        return grads_dx[0]

    # Run a single epoch.
    def run_epoch(self, epoch_length: int):
        global CURRENT_POS
        CURRENT_POS = self.position

        # ---- Next Batch ----
        total_accuracy = 0
        total_loss = 0.0
        for iteration, (inputs, targets) in tqdm(enumerate( mnist_train )): 
            if iteration > epoch_length:
                break       

            # ---- Forward pass ----
            output = self.nucleus.remote_forward(
                images = inputs,
                targets = targets,
            )

            # ---- Backward pass ----
            output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            output.loss.backward() # Accumulates gradients on the nucleus.
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation
            total_accuracy += output.remote_accuracy
            total_loss += output.remote_target_loss
        print ('Pos:', self.position, 'key:', self.endpoint.hotkey, 'Acc:', total_accuracy.item()/epoch_length, 'Loss:', total_loss.item()/epoch_length )


def wallet_for_index( i:int ):
    wallet = bittensor.wallet( name = 'n' + str(i), path = '/tmp/pytest' )
    if not wallet.has_hotkey:
        wallet.create_new_hotkey(use_password=False)
    if not wallet.has_coldkey:
        wallet.create_new_coldkey(use_password=False)
    wallet.hotkey
    wallet.coldkey
    return wallet

def endpoint_for_index( i:int, wallet: bittensor.Wallet ):
    endpoint = bittensor.endpoint( 
        uid = i, 
        hotkey = wallet.hotkey.public_key,
        coldkey = wallet.coldkey.public_key,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = 8080 + i, 
        modality = 0, 
    )
    return endpoint

def test_dgl():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', type=int, help='''n neurons''', default=3)
    parser.add_argument('--k', dest='k', type=int, help='''max depth''', default=1)
    parser.add_argument('--epoch_length', dest='epoch_length', type=int, help='''example per epoch''', default=10)
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, help='''number of epochs''', default=1)
    parser.add_argument('--debug', dest='debug', action='store_true', help='''turn on debug.''', default=False)
    hparams = parser.parse_args()

    # Set depth.
    global MAX_DEPTH
    MAX_DEPTH = hparams.k

    # Set debug.
    bittensor.logging.set_debug( hparams.debug )

    # Create neurons.
    neurons = []
    for i in range(hparams.n):
        wallet = wallet_for_index( i )
        endpoint = endpoint_for_index( i, wallet )
        neurons.append( 
            Neuron( 
                position = i,
                wallet = wallet, 
                endpoint = endpoint,
                child = None if i == 0 else neurons[ i - 1 ].endpoint
            )
        )

    for epoch in range(hparams.n_epochs):
        for i, neuron in enumerate( neurons ):
            neuron.run_epoch( hparams.epoch_length )
       

if __name__ == "__main__":
    test_dgl()