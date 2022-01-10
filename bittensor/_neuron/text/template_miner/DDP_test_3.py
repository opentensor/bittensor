import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def get_visible_gpus():
    ns = os.popen('nvidia-smi')
    lines_ns = ns.readlines()
    # print(lines_ns)
    for _i, _line in enumerate(lines_ns):
        if _line.find('|=') >= 0:
            break
    line_gpus = lines_ns[_i:]
    for _i, _line in enumerate(line_gpus):
        if _line.find('Processes') >= 0:
            break
    line_gpus = line_gpus[:_i-3]
    # print(line_gpus)
    idx_gpu_lines = []
    for _i, _line in enumerate(line_gpus):
        if _line.find('+') >= 0:
            idx_gpu_lines.append(_i+1)
    idx_gpus = []
    for _line_gpu in  idx_gpu_lines:
        idx_gpus.append(int(line_gpus[_line_gpu].split()[1]))
    # print(idx_gpus)
    return idx_gpus

def example(rank, world_size):
    print('rank:{}'.format(rank))
    # create default process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    # world_size = 2
    world_size = torch.cuda.device_count()
    print('world_size:{}'.format(world_size))
    print('get_visible_gpus():{}'.format(get_visible_gpus()))
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__ == "__main__":
    print(torch.__version__)
    main()
    print('Done\n\a')