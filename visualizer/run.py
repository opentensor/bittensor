import torch
import bittensor

metagraph = bittensor.metagraph.Metagraph()
metagraph.uid = 0
metagraph.connect(12)
metagraph.sync()

print (torch.tensor(metagraph.block - metagraph.lastemit).tolist())
print (metagraph.S.tolist())
print (list(zip(metagraph.S.tolist(), metagraph.uids.tolist(), metagraph.lastemit.tolist())))


# links = []
# nodes = [ {'id': uid, 's': s, 'r': r, 'i': i, 'e': e} for (uid, s, r, i, e) in list(zip(metagraph.uids.tolist(), metagraph.S.tolist(), metagraph.R.tolist(), metagraph.I.tolist(), metagraph.lastemit.tolist()))]

# for i in range(metagraph.n):
# 	for j in range(metagraph.n):
# 		w_ij = metagraph.W[i,j]
# 		uid_i = metagraph.uids[i]
# 		uid_j = metagraph.uids[j]
# 		links.append( {'source': int(uid_i.tolist()), 'target': int(uid_j.tolist()), 'weight': float(w_ij)} )
# gdata = {'links': links, 'nodes': nodes}

# print(gdata)

# import json
# with open('graph.json', 'w') as fp:
#     json.dump(gdata, fp)