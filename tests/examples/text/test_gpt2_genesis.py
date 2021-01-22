import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("examples/TEXT/")
from gpt2_genesis import Session
def test_run_gpt2_genesis():
    gpt2_genesis_session_config = Session.build_config()
    gpt2_genesis_session_config.metagraph.chain_endpoint = 'feynman.akira.bittensor.com:9944'
    gpt2_genesis_session_config.session.n_epochs = 1
    gpt2_genesis_session_config.session.epoch_length = 1
    gpt2_genesis_session = Session(gpt2_genesis_session_config)
    gpt2_genesis_session.run()
test_run_gpt2_genesis()