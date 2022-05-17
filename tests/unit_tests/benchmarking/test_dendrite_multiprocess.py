import bittensor
import torch
import time
from multiprocessing import Pool
from qqdm import qqdm

from bittensor.utils.test_utils import get_random_unused_port

wallet =  bittensor.wallet (
    path = f"/tmp/pytest{time.time()}",
    name = 'pytest',
    hotkey = 'pytest',
) 

wallet.create_new_coldkey( use_password=False, overwrite = True)
wallet.create_new_hotkey( use_password=False, overwrite = True)
logging =bittensor.logging(debug=True)
ports = [get_random_unused_port() for _ in range(5)]

inputs="""in my palm is a clear stone , and inside it is a
    small ivory statuette . a guardian angel .
    figured if you 're going to be out at night"""


def forward( inputs_x):
    return torch.zeros([1, 42, bittensor.__network_dim__])

def create_axon(port):
    axon = bittensor.axon (
        port = port,
        wallet = wallet,
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TEXT )
    axon.start()


def dendrite_delay(i):
    dend = bittensor.dendrite(wallet=wallet,max_active_receptors=10,multiprocess=True)
    for idx in range(100):
        responses, return_ops, query_times = dend.forward_text( endpoints=endpoints,inputs = inputs)
        assert all(return_ops) == 1 
        time.sleep(0.1)
    return

def main():
    global endpoints
    endpoints = []
    for i in ports:
        create_axon(i)
        wallet.create_new_hotkey( use_password=False, overwrite = True)
        endpoint = bittensor.endpoint(
            version = bittensor.__version_as_int__,
            uid = 1,
            hotkey = wallet.hotkey.ss58_address,
            ip = '0.0.0.0', 
            ip_type = 4, 
            port = i, 
            modality = 0, 
            coldkey = wallet.coldkey.ss58_address
        )
        endpoints += [endpoint]

    logging =bittensor.logging(debug=True)
    dend = bittensor.dendrite(wallet=wallet,max_active_receptors=10,multiprocess=True)
    responses, return_ops, query_times = dend.forward_text( endpoints=endpoints,inputs = inputs)
    assert all(return_ops) == 1
    
    N_processes = [1,2,3,4,5]
    N = len(N_processes)
    Num_experiments = 5
    collections = torch.zeros((Num_experiments,N))
    bittensor.logging(debug=False)
    experiments =  [i for i in range(Num_experiments)]
    for j in qqdm(experiments):
        for i in range(N):
            start = time.time()
            process =  N_processes[i]
            with Pool(process) as p:
                reps = p.map(dendrite_delay,list(range(i+1)))

            end = time.time()
            collections[j,i] = end-start
        time.sleep(1)
    
    means = torch.mean(collections,axis=0)
    error = torch.std(collections,axis=0)

    scaled_collections = torch.zeros((Num_experiments,N))
    for i in range(N):
        scaled_collections[:,i] = collections[:,i]/((i+1)*(100*len(ports)))

    means_scaled = torch.mean(scaled_collections,axis=0)
    error_scaled = torch.std(scaled_collections,axis=0)

    print ("{:<8}  {:<15}  {:<10}  {:<10}".format('# of Processes','Avg Time Elapsed','Standard Error','Time Per Payload'))
    for i in range(N):
        print ("{:^13} | {:^14.3f} | {:^14.3f} | {:^10.3f}".format(N_processes[i], means[i], error[i], means_scaled[i]))



if __name__ == "__main__":
    main()