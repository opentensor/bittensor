def torch_to_bytes(key):
    key = key.cpu().detach().numpy()
    key = key.tobytes()
    return key

def bytes_to_torch(key):
    torchkey = torch.Tensor(np.frombuffer(key, dtype=np.float32))
    return torchkey

def new_node_id():
    return os.urandom(12)

def new_key(dim):
    new_key = torch.rand(dim, dtype=torch.float32, requires_grad=False)
    return new_key

class Keys():
    def __init__(self, key_dim):
        self._key_dim = key_dim
        self._key_for_id = {}
        self._id_for_key = {}

    def addId(self, nid):
        key = new_key(self._key_dim)
        self._key_for_id[nid] = key
        self._id_for_key[torch_to_bytes(key)] = nid

    def addNodes(self, nodes: List[opentensor_pb2.Node]):
        

    def toKeys(self, nids):
        torch_keys = []
        for nid in nids:
            if nid not in self._key_for_id:
                addId(nid)
            torch_keys.append(self._key_for_id[nid])
        return torch_keys

    def toIds(self, keys):
        nids = []
        for k in keys:
            kb = torch_to_bytes(k)
            assert(kb in self._id_for_key)
            nids.append(self._id_for_key[kb])
        return nids

    def keys(self):
        return torch.cat(self._keys.toKeys(self._nodes_for_node_id.keys()), dim=0).view(-1, self._key_dim)



