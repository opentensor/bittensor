import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join, Joinable, JoinHook

class CounterJoinHook(JoinHook):
    r"""
    Join hook for :class:`Counter`.

    Arguments:
        counter (Counter): the :class:`Counter` object using this hook.
        sync_max_count (bool): whether to sync the max count once all ranks
            join.
    """
    def __init__(
        self,
        counter,
        sync_max_count
    ):
        self.counter = counter
        self.sync_max_count = sync_max_count

    def main_hook(self):
        r"""
        Shadows the counter's all-reduce by all-reducing a dim-1 zero tensor.
        """
        t = torch.zeros(1, device=self.counter.device)
        dist.all_reduce(t)

    def post_hook(self, is_last_joiner: bool):
        r"""
        Synchronizes the max count across all :class:`Counter` s if
        ``sync_max_count=True``.
        """
        if not self.sync_max_count:
            return
        rank = dist.get_rank(self.counter.process_group)
        common_rank = self.counter.find_common_rank(rank, is_last_joiner)
        if rank == common_rank:
            self.counter.max_count = self.counter.count.detach().clone()
        dist.broadcast(self.counter.max_count, src=common_rank)

class Counter(Joinable):
    r"""
    Example :class:`Joinable` that counts the number of training iterations
    that it participates in.
    """
    def __init__(self, device, process_group):
        super(Counter, self).__init__()
        self.device = device
        self.process_group = process_group
        self.count = torch.tensor([0], device=device).float()
        self.max_count = torch.tensor([0], device=device).float()

    def __call__(self):
        r"""
        Counts the number of inputs processed on this iteration by all ranks
        by all-reducing a dim-1 one tensor; increments its own internal count.
        """
        Join.notify_join_context(self)
        t = torch.ones(1, device=self.device).float()
        dist.all_reduce(t)
        self.count += t

    def join_hook(self, **kwargs) -> JoinHook:
        r"""
        Return a join hook that shadows the all-reduce in :meth:`__call__`.

        This join hook supports the following keyword arguments:
            sync_max_count (bool, optional): whether to synchronize the maximum
                count across all ranks once all ranks join; default is ``False``.
        """
        sync_max_count = kwargs.get("sync_max_count", False)
        return CounterJoinHook(self, sync_max_count)

    @property
    def join_device(self) -> torch.device:
        return self.device

    @property
    def join_process_group(self):
        return self.process_group

    def find_common_rank(self, rank, to_consider):
        r"""
        Returns the max rank of the ones to consider over the process group.
        """
        common_rank = torch.tensor([rank if to_consider else -1], device=self.device)
        dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
        common_rank = common_rank.item()
        return common_rank



