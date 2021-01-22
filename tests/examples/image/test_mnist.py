import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("examples/image/")
from mnist import Session
def test_run_mnist():
    mnist_session_config = Session.build_config()
    mnist_session_config.metagraph.chain_endpoint = 'feynman.akira.bittensor.com:9944'
    mnist_session_config.session.n_epochs = 1
    mnist_session_config.session.epoch_length = 1
    mnist_session = Session(mnist_session_config)
    mnist_session.run()
test_run_mnist()