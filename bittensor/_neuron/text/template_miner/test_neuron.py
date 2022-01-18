import os
from re import L
import pandas
from pandas.core.frame import DataFrame
import bittensor
import math
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import traceback
import sys
import wandb
from termcolor import colored
from qqdm import qqdm, format_str
from loguru import logger

from bittensor._metagraph import metagraph


from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from functools import partial

import torch.nn.functional as F
from torch.multiprocessing import Manager
import copy

from unittest.mock import MagicMock
import neuron_impl
import copy

logger = logger.opt(colors=True)
bittensor.logging(debug = True)

class NeuronDDP(neuron_impl.Neuron):
    def __init__(self):
        config = bittensor.neurons.template_miner.neuron.config()
        config.neuron.device = 'cpu'
        config.wallet.name = 'test4'
        config.neuron.world_size = 3
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        bittensor.logging.success("full path", sufix = f"{config.neuron.full_path}")
        config.dataset.dataset_name = ['Books3']

        nucleus = bittensor.neurons.template_miner.nucleus(config)
        super().__init__( config, nucleus )

        self.wallet.create()

    def __exit__(self):
        del self.dendrite
        self.dataset.close()
        self.cleanup()

    def get_data(self, rank):
        inputs = torch.load(f'./bittensor/_neuron/text/template_miner/dataset_input{rank}.pt')
        return inputs

    def run(self, rank, world_size):

        with self:
            self.init_process(rank)
            self.init_bit(rank)

            try:
                self.reload(rank)
            except:
                if rank == 0:
                    bittensor.logging.success("saving", sufix = f"{rank}")
                    self.save()
                dist.barrier()
                self.reload(rank)

            self.nucleus.device = self.device
            self.nucleus.dendrite = self.dendrite
            # self.nucleus.metagraph = self.metagraph
            self.nucleus = DDP(self.nucleus)

            inputs = self.get_data(rank)
            output = self.nucleus.forward(
                inputs = inputs,
                training = True,
            )

            bittensor.logging.success("finished forward", sufix = "yy")
            output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            output.loss.backward(retain_graph = True)
            bittensor.logging.success("finished backward", sufix = "yy")
            # dist.barrier()
            # for p in self.nucleus.named_parameters():
            #     print(p)

        return self.nucleus.parameters()

class NeuronDDP2:
    def __init__(self):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        self.world_size = 3
        self.nucleus = bittensor.neurons.template_miner.nucleus()
        self.config = bittensor.neurons.template_miner.neuron().config
        self.wallet = bittensor.wallet ( config = self.config, name = 'test4', hotkey = 'd1')
        self.wallet.create()
        
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( self.config.logging.logging_dir, self.config.wallet.name, self.config.wallet.hotkey, self.config.neuron.name ))
        self.config.neuron.full_path = os.path.expanduser(full_path)

    def __exit__(self):
        del self.dendrite
        self.dataset.close()
        self.cleanup()
    
    def reload(self):
        state_dict =  torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        self.nucleus.device = self.device 
        self.nucleus.dendrite = self.dendrite # Set local dendrite.
        self.nucleus.metagraph = self.metagraph # Set local metagraph.
        bittensor.logging.success('reloaded', sufix = '')

    def get_data(self, rank):
        inputs = torch.load(f'./bittensor/_neuron/text/template_miner/dataset_input{rank}.pt')
        return inputs

    def init_process(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8865'
        backend = 'gloo'
        dist.init_process_group(backend, rank=rank, world_size=self.world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def run_parallel( self ):
        mp.spawn(self.run,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True
        )

    def init_bit(self, rank):
        self.device = torch.device( device = 'cpu' ) 
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = lambda : bittensor.metagraph ( config = self.config, subtensor = self.subtensor ).sync()
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dendrite.receptor_pool.forward = MagicMock(return_value = [torch.tensor([]), [2,2,2,2,2], [0]]) 
        if rank == 0 :
            self.subtensor.register( self.wallet )

    def run(self, rank, world_size):
        self.init_process(rank)
        self.init_bit(rank)

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)

        self.reload()
        inputs = self.get_data(rank)
        bittensor.logging.success("finished gettign data", sufix = "yy")

        self.nucleus = DDP(self.nucleus)
        bittensor.logging.success("enabled ddp", sufix = "yy")

        output = self.nucleus.forward(
            inputs = inputs,
            training = True,
        )
        bittensor.logging.success("finished forward", sufix = "yy")
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        
        output.loss.backward(retain_graph = True)

        bittensor.logging.success("finished backward", sufix = "yy")

        if rank == 0:
            grads = {k.replace('module.', '') : (v, v.grad) for k,v in self.nucleus.named_parameters()}
            torch.save(grads, 'DDP_params_grad.pt')
            print('saved pt')

        return 

class NeuronDDPSim:
    def __init__(self):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        self.world_size = 3
        self.config = bittensor.neurons.template_miner.neuron().config
        self.wallet = bittensor.wallet ( config = self.config, name = 'test4', hotkey = 'd1')
        self.wallet.create()
        self.device = torch.device( device = 'cpu' ) 
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = lambda : bittensor.metagraph ( config = self.config, subtensor = self.subtensor ).sync()
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dendrite.receptor_pool.forward = MagicMock(return_value = [torch.tensor([]), [2,2,2,2,2], [0]]) 
        self.subtensor.register( self.wallet )

        self.nucleus = bittensor.neurons.template_miner.nucleus()
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( self.config.logging.logging_dir, self.config.wallet.name, self.config.wallet.hotkey, self.config.neuron.name ))
        self.config.neuron.full_path = os.path.expanduser(full_path)
        self.reload()

    def get_data(self, rank):
        inputs = torch.load(f'./bittensor/_neuron/text/template_miner/dataset_input{rank}.pt')
        return inputs

    def reload(self):
        state_dict =  torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        self.nucleus.device = self.device 
        self.nucleus.dendrite = self.dendrite # Set local dendrite.
        self.nucleus.metagraph = self.metagraph # Set local metagraph.
        bittensor.logging.success('reloaded', sufix = '')
        
    def run(self):
        grads = {}
        for rank in range(self.world_size):
            self.nucleus.zero_grad()
            inputs = self.get_data(rank)
            bittensor.logging.success("finished gettign data", sufix = f"{rank}")
            print(f"finished gettign data {rank}")

            output = self.nucleus.forward(
                inputs = inputs,
                training = True,
            )
            bittensor.logging.success("finished forward", sufix = f"{rank}")
            output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            output.loss.backward(retain_graph = True)

            bittensor.logging.success("finished backward", sufix = f"{rank}")
            print("finished gettign backward")
            g = {k.replace('module.', ''): (v, v.grad) for k,v in self.nucleus.named_parameters()}
            grads[rank] = copy.deepcopy(g)

        avg_grads = {}
        for n, v in grads[0].items():
            sum_grad = v[1] + grads[1][n][1] + grads[2][n][1]
            avg_grads[n] = (v[0], sum_grad/3)
            print(n, v[1], grads[1][n][1], grads[2][n][1] , sum_grad/3)

        return inputs, output, avg_grads

def test_grads_align():
    
    neuron1 = NeuronDDPSim()
    input1, output1, grads1 = neuron1.run()
    
    neuron2 = NeuronDDP2()
    neuron2.run_parallel()
    grads2 = torch.load('DDP_params_grad.pt')
    
    for key in grads1.keys():
        print("======================", key)
        p1 = grads1[key]
        p2 = grads2[key]

        if torch.any(p1[0] != p2[0]).item():
            print("value NOT eq")

    for key in grads1.keys():
        print("====================== grad, ", key)
        p1 = grads1[key]
        p2 = grads2[key]

        if torch.any( abs(p1[1] - p2[1]) > 1e-7 ).item():
            print("NOT eq")
            not_eq_ids = torch.where(abs(p1[1] - p2[1]) > 1e-7)
            print(p1[1][not_eq_ids], '\n', p2[1][not_eq_ids])
            print((p1[1]-p2[1])[not_eq_ids])

        else:
            print("eq")

    
if __name__ == '__main__':
    test_grads_align()