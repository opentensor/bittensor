import torch
import time
import cProfile, pstats, io
from pstats import SortKey
import concurrent
import copy

task = torch.load('tensor_split_stable.pt')
compact_topk = task['compact_topk']
prob_idx = task['prob_idx']
phrases = task['phrases']
max_len = task['max_len']
topk_tensor = task['topk_tensor']

# torch.save(task, 'tensor_split_stable.pt')

def old_split(arg):
    i, compact_topk, prob_idx = arg
    ss = torch.tensor_split(compact_topk, prob_idx)
    ss = [s.tolist() for s in ss]

def split(arg):
    start_time = time.time()
    th, compact_topk, prob_idx = arg
    org_ss = torch.tensor_split(compact_topk, prob_idx)
    diff = (prob_idx[1:] - prob_idx[:-1])
    # index_cut = prob_idx[1:][diff != 2]
    cuts = []
    # for cut, d in zip(index_cut, diff[diff != 2]):
        # cuts += [cut-d, cut]
    for i, d in enumerate(diff):
        if d != 2:
            if th == 0:
                print(d, prob_idx[i-1], prob_idx[i], prob_idx[i+1])
            cuts += [prob_idx[i], prob_idx[i+1]]
    ss = torch.tensor_split(compact_topk, cuts)
    temp_phrases = []
    for s in ss[:-1]:
        if len(s) % 2 == 0 and len(s) > 0:
            s_reshape = torch.reshape(s, (-1,2))
            zeros = torch.zeros((s_reshape.shape[0], 1))
            s_reshape = torch.cat((s_reshape, zeros), dim = 1) 
            temp_phrases.append(s_reshape)
        else:
            zeros = torch.zeros(3 - len(s))
            s_reshape = torch.cat((s, zeros)) 
            s_reshape = torch.unsqueeze(s_reshape, 0)
            temp_phrases.append(s_reshape)

    phrases = torch.cat (temp_phrases)
    if th % 100 == 0:
        print(i, 'finished')
        print('\n\nlen(compact_topk)', len(compact_topk))
        print('\n\ncompact_topk', compact_topk)
        print('\n\ncompact_topk[262188]', compact_topk[262187])
        print('\n\nprob_idx', prob_idx)
        print('\n\ndiff', diff)
        # print('\n\nindex_cut', index_cut)
        print('\n\ncuts', cuts)
        print('\n\nsplit', [s.shape for s in ss])
        print('\n\norg_split', org_ss[:3], org_ss[-3:])
        print('\n\nphrases', phrases[0], phrases[-1])
        print('\n\nphrases',phrases.shape)

n = 3

args = []
for i in range(n):
    args.append((i, copy.deepcopy(compact_topk), copy.deepcopy(prob_idx)))

# =====================================================================
# start_time = time.time()
# pr = cProfile.Profile()
# pr.enable()

# with concurrent.futures.ThreadPoolExecutor( max_workers = min(8, n) ) as executor:
#     executor.map( old_split, args )

# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats(20)
# print(s.getvalue())

# ==============================================================
start_time = time.time()
pr = cProfile.Profile()
pr.enable()

with concurrent.futures.ThreadPoolExecutor( max_workers = n ) as executor:
    executor.map( split, args )

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())

# # ===========================================================

start_time = time.time()
pr = cProfile.Profile()
pr.enable()

for i in args:
    split(i)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())
