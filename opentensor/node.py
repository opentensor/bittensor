class Node():
    def __init__(self):
        pass

    def fwd (self, key, tensor):
        raise NotImplementedError

    def bwd (self, key, tensor):
        raise NotImplementedError
 
