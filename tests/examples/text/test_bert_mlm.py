import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("examples/TEXT/")
from bert_mlm import Session
def test_run_bert_mlm():
    bert_mlm_session_config = Session.build_config()
    bert_mlm_session_config.metagraph.chain_endpoint = 'feynman.akira.bittensor.com:9944'
    bert_mlm_session_config.session.n_epochs = 1
    bert_mlm_session_config.session.epoch_length = 1
    bert_mlm_session = Session(bert_mlm_session_config)
    bert_mlm_session.run()
test_run_bert_mlm()