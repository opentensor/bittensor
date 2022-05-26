import bittensor
import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

adv_server = bittensor._neuron.text.advanced_server.server()

def test_set_fine_tuning_params():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 50; network_dim = 10; nlayers_1 = 4; nlayers_2 = 3; max_n = 5; nhead = 2
            self.embedding = torch.nn.Embedding( vocab_size,  network_dim )
            self.encoder_layers = TransformerEncoderLayer( network_dim, nhead )
            self.encoder = TransformerEncoder( self.encoder_layers, nlayers_1 )
            self.encoder2 = TransformerEncoder( self.encoder_layers, nlayers_2 )
            self.decoder = torch.nn.Linear( network_dim, vocab_size , bias=False)
          
    # test for the basic default gpt2 case
    assert adv_server.set_fine_tuning_params() == (True, 'h.11')
    
    # test for the case when there are 2 modulelists
    adv_server.pre_model = Model()
    assert adv_server.set_fine_tuning_params() == (True, 'encoder2.layers.2')
    
    # test for user specification of the number of layers
    adv_server.config.neuron.finetune.num_layers = 3
    assert adv_server.set_fine_tuning_params() == (True, 'encoder2.layers.0')
    
    # test for user specification of the number of layers
    adv_server.config.neuron.finetune.num_layers = 4
    assert adv_server.set_fine_tuning_params() == (True, 'encoder.layers.0')
    
    # test for user specification of the number of layers set too large
    adv_server.config.neuron.finetune.num_layers = 5
    assert adv_server.set_fine_tuning_params() == (False, None)
    
    # test for user specification of the layer name
    adv_server.config.neuron.finetune.layer_name = 'encoder2.layers.1'
    assert adv_server.set_fine_tuning_params() == (True, 'encoder2.layers.1')
    
    # test for user specification of a non-existing layer name
    adv_server.config.neuron.finetune.layer_name = 'non_existing_layer'
    assert adv_server.set_fine_tuning_params() == (False, 'non_existing_layer')
    

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 50; network_dim = 10; nlayers = 2; max_n = 5; nhead = 2
            self.decoder = torch.nn.Linear( network_dim, vocab_size , bias=False)
            
    # test for a non-existing modulelist
    adv_server.pre_model = Model()
    adv_server.config.neuron.finetune.layer_name = None
    assert adv_server.set_fine_tuning_params() == (False, None) 


if __name__ == '__main__':
    test_set_fine_tuning_params()
