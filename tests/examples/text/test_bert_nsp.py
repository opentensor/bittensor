import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("examples/TEXT/")
from bert_nsp import Session
def test_run_bert_nsp():
    bert_nsp_session_config = Session.build_config()
    bert_nsp_session_config.metagraph.chain_endpoint = 'feynman.akira.bittensor.com:9944'
    bert_nsp_session_config.session.n_epochs = 1
    bert_nsp_session_config.session.epoch_length = 5
    bert_nsp_session = Session(bert_nsp_session_config)
    bert_nsp_session.run()
test_run_bert_nsp()