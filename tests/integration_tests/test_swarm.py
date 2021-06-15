
import bittensor
import torch
from nuclei.ffnn import FFNNNucleus
from loguru import logger
from torchvision import datasets, transforms
bittensor.__debug_on__ = True

mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
mnist_train = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=False)

class Neuron:
    def __init__( self, wallet: bittensor.Wallet, endpoint: bittensor.Endpoint, child: bittensor.Endpoint ):
        
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
                
    # Function is called by the nucleus to query child and get responses.
    def route( self, inputs: torch.FloatTensor, query: torch.FloatTensor ) -> torch.FloatTensor:
        
        # Is this a leaf node.
        if self.child == None:
            response = [torch.zeros( [inputs.shape[0], inputs.shape[1], bittensor.__network_dim__ ])]
        
        # Otherwise, makes differentiable calls.
        else:
            # Takes a list of endpoints and a list of inputs
            # Sends inputs to endpoints.
            responses, return_codes = self.dendrite.forward_image (
                endpoints = [self.child], 
                inputs = [inputs] 
            )
            
        return responses[0]
    
    # Function which is called when this miner recieves a forward request from a dendrite.
    def forward ( self, pubkey:str, images: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        # Call nucleus (locally, i.e. using the distillation model instead of calling the child)
        # return the last hidden layer.  
        return self.nucleus.forward_image (
            images = images       
        )

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
    
    # Start the axon serving endpoint.
    def start(self):
        self.axon.start()
        
    # Tear down the axon serving endpoint.
    def __del__(self):
        self.axon.stop()

    # Run a single epoch.
    def epoch(self):
        # ---- Next Batch ----
        for _, (inputs, targets) in enumerate( mnist_train ):      

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
            

def test_swarm():
    # Two fake bittensor endpoints for the miners.
    wallet_A = bittensor.wallet( name = 'alice', path = '/tmp/pytest', hotkey='alice_hotkey')
    if not wallet_A.has_hotkey:
        wallet_A.create_new_hotkey(use_password=False)
    if not wallet_A.has_coldkey:
        wallet_A.create_new_coldkey(use_password=False)
    wallet_A.hotkey
    wallet_A.coldkey
    wallet_B = bittensor.wallet( name = 'bob', path = '/tmp/pytest', hotkey='bob_hotkey')
    if not wallet_B.has_hotkey:
        wallet_B.create_new_hotkey(use_password=False)
    if not wallet_B.has_coldkey:
        wallet_B.create_new_coldkey(use_password=False)
    wallet_A.hotkey
    wallet_A.coldkey
    endpoint_A = bittensor.endpoint( 
        uid = 0, 
        hotkey = wallet_A.hotkey.public_key,
        coldkey = wallet_A.coldkey.public_key,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = 8080 , 
        modality = 0, 
    )
    endpoint_B = bittensor.endpoint( 
        uid = 1, 
        hotkey = wallet_B.hotkey.public_key,
        coldkey = wallet_B.coldkey.public_key,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = 8081, 
        modality = 0, 
    )
    neuron_A = Neuron ( 
        wallet = wallet_A,
        endpoint = endpoint_A, 
        child = endpoint_B 
    )
    neuron_B = Neuron ( 
        wallet = wallet_B, 
        endpoint = endpoint_B, 
        child = endpoint_A 
    )
    neuron_A.start()
    neuron_B.start()
    neuron_A.epoch()
    #neuron_B.epoch()


if __name__ == "__main__":
    test_swarm()