import torch
import time
import cProfile, pstats, io
from pstats import SortKey
import concurrent
import copy
from bittensor.utils.tokenizer_utils import compact_topk_token_phrases, unravel_topk_token_phrases, prepend_tensor_legacy, prepend_tensor


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
    compact_topk, prob_idx, ignore_index = arg
    ignore_index = -100
    ignore_index += 2

    # org_ss = torch.tensor_split(compact_topk, prob_idx) # rm later
    phrase_size = prob_idx[1:] - prob_idx[:-1]
    split_idx = []
    split_idx_partial = prob_idx[1:][phrase_size!= 2]
    split_size = phrase_size[phrase_size != 2]
    max_len = max(split_size)
    for idx, size in zip(split_idx_partial, split_size):
        split_idx += [idx-size, idx]

    # make sure the last of split_idx == prob_idx, so that we dont miss out the last section cut
    if split_idx[-1] != prob_idx[-1]:
        split_idx += [prob_idx[-1]]

    split_topk = torch.tensor_split(compact_topk, split_idx)
    phrases_list = []
    for s in split_topk:
        if len(s) % 2 == 0 and len(s) > 0:
            s_reshape = torch.reshape(s, (-1,2))
            ignore = torch.ones((s_reshape.shape[0], max_len-2))
            ignore *= ignore_index
            s_reshape = torch.cat((s_reshape, ignore), dim = 1) 
            phrases_list.append(s_reshape)
        else:
            ignore = torch.ones(max_len - len(s))
            ignore *= ignore_index
            s_reshape = torch.cat((s, ignore)) 
            s_reshape = torch.unsqueeze(s_reshape, 0)
            phrases_list.append(s_reshape)

    topk = 4096
    batch_size = len(prob_idx) // (topk + 1)  # (batch_size * (topk + floor)) / (topk + floor)
    assert batch_size * (topk + 1) == len(prob_idx), f'{batch_size} * ({topk} + 1) != {len(prob_idx)}'  # decoding irregularity otherwise

    topk_tensor = phrases = torch.cat(phrases_list)
    topk_tensor -= 2
    # grafting probability tensors into first column to attach gradients
    topk_tensor[:, 0] = compact_topk[prob_idx]  # tensor([prob_k=0_b, prob_k=1_b, ..., prob_floor_b])
    topk_tensor = topk_tensor.reshape(batch_size, topk + 1, max_len)  # [batch_size, (topk + 1), max_len] reshaped


    if False and th % 100 == 0:
        # for i, (p, s) in enumerate(zip(phrases.tolist(), org_ss[1:])):
            # if p[0] != s.tolist()[0] or i % 20000 == 0:
                # print(i, p, s)
        print(i, 'finished')
        print('\n\nlen(compact_topk)', len(compact_topk))
        print('\n\ncompact_topk', compact_topk)
        print('\n\ncompact_topk[262188]', compact_topk[262187])
        print('\n\nprob_idx', prob_idx)
        print('\n\nphrase_size', phrase_size)
        # print('\n\nindex_cut', split_idx)
        print('\n\nindex_cut', split_idx)
        print('\n\nsplit', [s.shape for s in split_topk])
        # print('\n\norg_split', org_ss[:3], org_ss[-3:], org_ss[131104])
        print('\n\nphrases', phrases[:3], phrases[-3:])
        # print('\n\nphrases',len(org_ss), phrases.shape)

        task = torch.load('tensor_split_stable.pt')
        true_topk_tensor = task['topk_tensor']
        print('\n\ntopk_tensor', topk_tensor == true_topk_tensor )

n = 10

args = []
for i in range(n):
    args.append((copy.deepcopy(compact_topk), copy.deepcopy(prob_idx), -102))

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

split(args[0])
# ==============================================================
start_time = time.time()
pr = cProfile.Profile()
pr.enable()

with concurrent.futures.ThreadPoolExecutor( max_workers = min(7, n)) as executor:
    executor.map( split, args )

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())

# ==============================================================
start_time = time.time()
pr = cProfile.Profile()
pr.enable()

with concurrent.futures.ThreadPoolExecutor( max_workers = min(7, n)) as executor:
    executor.map( prepend_tensor, args )

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

with concurrent.futures.ThreadPoolExecutor( max_workers = min(7, n)) as executor:
    executor.map( prepend_tensor_legacy, args )

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())

print(torch.all(torch.eq(prepend_tensor(*args[0])[1], prepend_tensor_legacy(*args[0])[1])))