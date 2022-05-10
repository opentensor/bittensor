import bittensor
import json

def test_ipfs_init():
    ipfs = bittensor.Ipfs()

    assert ipfs.cat == 'http://global.ipfs.opentensor.ai/api/v0/cat' 
    assert ipfs.node_get == 'http://global.ipfs.opentensor.ai/api/v0/object/get'
    assert ipfs.ipns_resolve == 'http://global.ipfs.opentensor.ai/api/v0/name/resolve'

    assert ipfs.mountain_hash == 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
    assert ipfs.latest_neurons_ipns == 'k51qzi5uqu5di1eoe0o91g32tbfsgikva6mvz0jw0414zhxzhiakana67shoh7'
    assert ipfs.historical_neurons_ipns == 'k51qzi5uqu5dhf5yxm3kqw9hyrv28q492p3t32s23059z911a23l30ai6ziceh'

    assert ipfs.refresh_corpus == False

def test_retrieve_directory():
    ipfs = bittensor.Ipfs()

    directory = ipfs.retrieve_directory(ipfs.node_get, (('arg', ipfs.mountain_hash),))
    folder_list = json.loads(directory.text)
    assert directory.status_code == 200
    assert len(folder_list) > 1
    
