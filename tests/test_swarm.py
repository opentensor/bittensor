from multiprocessing import Process
import os

import bittensor

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def neuron(name):
    info('function f')
    parser = argparse.ArgumentParser()
    config = bittensor.Config.add_args(parser)
    config = bittensor.Config()
    bittensor.init( config )

if __name__ == '__main__':
    info('main line')
    p = Process(target=neuron, args=('bob',))
    p.start()
    p.join()