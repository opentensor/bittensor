import argparse
import random
from numpy import average
from torch.utils.tensorboard.summary import hparams
import bittensor
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from nuclei.ffnn import FFNNNucleus
from loguru import logger
from torchvision import datasets, transforms

mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
mnist_train = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=False)

QUIT = False

class Neuron:
    def __init__( self, wallet: bittensor.Wallet, endpoint: bittensor.Endpoint ):

        self.update_lock = threading.Lock()
        
        # wallet.
        self.wallet = wallet

        # Local info.
        self.endpoint = endpoint
        
        # Child to call forward on.
        self.children = []
                
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
        if len(self.children) == 0:
            responses = torch.zeros( [inputs.shape[0], inputs.shape[1], bittensor.__network_dim__] )
        
        # Otherwise, makes differentiable calls.
        else:
            # Takes a list of endpoints and a list of inputs
            # Sends inputs to endpoints.
            responses, return_codes = self.dendrite.forward_image (
                endpoints = self.children, 
                inputs = [ inputs for _ in self.children ]
            )
            # Average response across network dimension.
            responses = torch.stack(responses, dim=2).sum(dim=2)
            
        return responses
    
    # Function which is called when this miner recieves a forward request from a dendrite.
    def forward ( self, pubkey:str, images: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        # Call nucleus (locally, i.e. using the distillation model instead of calling the child)
        # return the last hidden layer.  
        with self.update_lock:
            outputs = self.nucleus.forward_image (
                images = images       
            )
        return outputs

    # Function which is called when this miner recieves a backward request.
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        with self.update_lock:
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
    def run_epoch(self):
        print (self.endpoint.hotkey, 'started running epoch')
        # ---- Next Batch ----
        for _, (inputs, targets) in enumerate( mnist_train ): 
            if QUIT:
                break

            # ---- Forward pass ----
            output = self.nucleus.remote_forward(
                images = inputs,
                targets = targets,
            )

            # ---- Backward pass ----
            with self.update_lock:
                output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
                output.loss.backward() # Accumulates gradients on the nucleus.
                self.optimizer.step() # Applies accumulated gradients.
                self.optimizer.zero_grad() # Zeros out gradients for next accummulation


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

def test_swarm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', type=int, help='''n neurons''', default=3)
    parser.add_argument('--k', dest='k', type=int, help='''n children''', default=2)
    parser.add_argument('--debug', dest='debug', action='store_true', help='''turn on debug.''', default=False)
    hparams = parser.parse_args()

    # Set debug.
    bittensor.logging.set_debug( hparams.debug )

    # Create neurons.
    neurons = []
    for i in range(hparams.n):
        wallet = wallet_for_index( i )
        endpoint = endpoint_for_index( i, wallet )
        neurons.append( Neuron ( 
            wallet = wallet, 
            endpoint = endpoint, 
        ))

    # Set children
    for n_a in neurons:
        n_a.children = [ n_b.endpoint for n_b in random.sample( neurons, hparams.k ) if n_b.endpoint.hotkey != n_a.endpoint.hotkey ]

    # Run in threadpool.
    try:
        with ThreadPoolExecutor() as executor:
            for n in neurons:
                executor.submit( n.run_epoch )
    except KeyboardInterrupt:
        QUIT = True


if __name__ == "__main__":
    test_swarm()