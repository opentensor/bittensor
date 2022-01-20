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
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
from torch._six import inf
from torch.nn.utils import clip_grad_norm_

logger = logger.opt(colors=True)
bittensor.logging(debug = True)
class NeuronDDP:
    def __init__(self):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        self.world_size = 3
        self.nucleus = bittensor.neurons.template_miner.nucleus()
        self.config = bittensor.neurons.template_miner.neuron().config
        self.wallet = bittensor.wallet ( config = self.config, name = 'test4', hotkey = 'd2')
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

    def init_process(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8865'
        backend = 'gloo'
        dist.init_process_group(backend, rank=rank, world_size=self.world_size)

    def init_bit(self, rank):
        self.device = torch.device( device = 'cpu' ) 
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = lambda : bittensor.metagraph ( config = self.config, subtensor = self.subtensor ).sync()
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dendrite.receptor_pool.forward = MagicMock(return_value = [torch.tensor([]), [2,2,2,2,2], [0]]) 
        if rank == 0 :
            self.subtensor.register( self.wallet )

    def cleanup(self):
        dist.destroy_process_group()

    def run_parallel( self ):
        mp.spawn(self.run,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True
        )

    def get_data(self, rank):
        inputs = torch.load(f'./bittensor/_neuron/text/template_miner/dataset_input{rank}.pt')
        return inputs

    def allreduce_hook(
        self, process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ): # -> torch.futures.Future[torch.Tensor]:
        print("in all reduce, index", bucket.index(), len(bucket.parameters()))
        
        if bucket.index() == 0:
            total_norm = clip_grad_norm_(bucket.parameters(), 1)
            self.total_norm = total_norm
            print("total norm", total_norm)
        if bucket.index() == 1:
            clip_coef = 1 / (self.total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for p in bucket.parameters():
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

        flat_grads = [ torch.reshape(p.grad, (-1,) ) for p in bucket.parameters()]
        tensor = torch.cat(flat_grads)
        bucket.set_buffer(tensor)
        group_to_use = process_group if process_group is not None else dist.group.WORLD

        # Apply the division first to avoid overflow, especially for FP16.
        tensor.div_(group_to_use.size())

        return (
            dist.all_reduce(tensor, group=group_to_use, async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    def run(self, rank, world_size):
        self.init_process(rank)
        self.init_bit(rank)

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)

        self.reload()
        inputs = self.get_data(rank)

        self.nucleus = DDP(self.nucleus, bucket_cap_mb = 10000000)
        self.nucleus.register_comm_hook(state=None, hook=self.allreduce_hook)

        bittensor.logging.success("Enabled ddp", sufix = f"rank: {rank}")

        output = self.nucleus.forward(
            inputs = inputs,
            training = True,
        )

        bittensor.logging.success("Forward pass", sufix = f"rank: {rank}")
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        
        output.loss.backward(retain_graph = True)

        bittensor.logging.success("Backward pass", sufix = f"rank: {rank}")

        if rank == 0:
            grads = {k.replace('module.', '') : (v, v.grad) for k,v in self.nucleus.named_parameters()}
            torch.save(grads, 'DDP_params_grad.pt')
            print('saved pt')

        self.cleanup()
        return 

class NeuronDDPSim:
    def __init__(self):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        self.world_size = 3
        self.config = bittensor.neurons.template_miner.neuron().config
        self.wallet = bittensor.wallet ( config = self.config, name = 'test4', hotkey = 'd2')
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

            output = self.nucleus.forward(
                inputs = inputs,
                training = True,
            )
            bittensor.logging.success("Forward pass", sufix = f"rank: {rank}")
            output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            output.loss.backward(retain_graph = True)

            total_norm = clip_grad_norm_(self.nucleus.parameters(), 1)
            bittensor.logging.success("Backward pass", sufix = f"rank: {rank}")
            g = {k.replace('module.', ''): (v, v.grad) for k,v in self.nucleus.named_parameters()}
            grads[rank] = copy.deepcopy(g)

        avg_grads = {}
        for n, v in grads[0].items():
            sum_grad = v[1] + grads[1][n][1] + grads[2][n][1]
            avg_grads[n] = (v[0], sum_grad/3)

        return inputs, output, avg_grads

def test_grads_align():
    
    neuron_ddp = NeuronDDP()
    neuron_ddp.run_parallel()
    grads_ddp = torch.load('DDP_params_grad.pt')

    neuron_ddp_sim = NeuronDDPSim()
    input1, output1, grads_ddp_sim = neuron_ddp_sim.run()
    
    for key in grads_ddp.keys():
        print("======================", key)
        p1 = grads_ddp[key]
        p2 = grads_ddp_sim[key]

        if torch.any(p1[0] != p2[0]).item():
            print("value NOT eq")


        grad_diff = abs(p1[1] - p2[1]) 
        if torch.any(  torch.logical_and (grad_diff > 1e-7, grad_diff > abs(p1[1]/1000) )  ).item():
            print("grad NOT eq")
            not_eq_ids = torch.where(abs(p1[1] - p2[1]) > 1e-7)
            print(p1[1][not_eq_ids], '\n', p2[1][not_eq_ids])
            print((p1[1]-p2[1])[not_eq_ids])

    
if __name__ == '__main__':
    test_grads_align()