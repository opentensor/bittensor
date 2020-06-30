import numpy 
import pickle

def serialize(array):
    return pickle.dumpy(array, protocol=0)

def deserialize(bytes_content):
    return pickle.loads(bytes_content)
