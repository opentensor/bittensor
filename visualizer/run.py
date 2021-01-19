from bittensor import metagraph
from munch import Munch

from bittensor.subtensor.interface import Keypair
from bittensor.metagraph import Metagraph
from munch import Munch
import torch


config = {'session':
              {'datapath': 'data/', 'learning_rate': 0.01, 'momentum': 0.9, 'batch_size_train': 64,
               'batch_size_test': 64, 'log_interval': 10, 'sync_interval': 100, 'priority_interval': 100,
               'name': 'mnist', 'trial_id': '1608070667'},
          'synapse': {'target_dim': 10},
          'dendrite': {'key_dim': 100, 'topk': 10, 'stale_emit_filter': 10000, 'pass_gradients': True, 'timeout': 0.5,
                       'do_backoff': True, 'max_backoff': 100}, 'axon': {'local_port': 8091, 'external_ip': '191.97.53.53', 'max_workers': 5, 'max_gradients': 1000},
          'nucleus': {'max_workers': 5, 'queue_timeout': 5, 'queue_maxsize': 1000},
          'metagraph': {'chain_endpoint': 'feynman.kusanagi.bittensor.com:9944', 'stale_emit_filter': 100000000000000},
          'meta_logger': {'log_dir': 'data/'},
          'neuron': {'keyfile': None, 'keypair': None }
          }


config = Munch.fromDict(config)
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)
config.wallet.keypair = keypair
metagraph = Metagraph(config)
metagraph.uid = 0

metagraph.connect(12)

metagraph.sync()


print (torch.tensor(metagraph.block - metagraph.lastemit).tolist())
print (metagraph.S.tolist())
print (list(zip(metagraph.S.tolist(), metagraph.uids.tolist(), metagraph.lastemit.tolist()))




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





