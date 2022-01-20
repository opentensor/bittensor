import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn.utils import clip_grad_norm_


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
        initrange = 0.1
        self.net1.weight.data.uniform_(-initrange, initrange)
        self.net2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        return self.net2(self.relu(self.net1(x)))

def noop(state: object, bucket: dist.GradBucket): # -> torch.futures.Future[torch.Tensor]
    print("fkkkkkk")
    fut = torch.futures.Future()
    fut.set_result(bucket.buffer())
    return fut

def demo_basic(rank, world_size):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel()# .to(rank)
    # model = neuron()
    # ddp_model = DDP(model, device_ids=[rank])
    ddp_model = DDP(model)
    ddp_model.register_comm_hook(state=None, hook=noop)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    with ddp_model.join():
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5)# .to(rank)
        loss_fn(outputs, labels).backward()
        for n, p in ddp_model.named_parameters():
            print(n, p.grad) 
        # optimizer.step()

        # if rank == 0:
        #     for i in range(5):
        #         print(f"Running basic DDP example on rank {rank}.")
        #         optimizer.zero_grad()
        #         outputs = ddp_model(torch.randn(20, 10))
        #         labels = torch.randn(20, 5)# .to(rank)
        #         loss_fn(outputs, labels).backward()
        #         optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    # freeze_support()
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = 3 # n_gpus 
    run_demo(demo_basic, world_size)