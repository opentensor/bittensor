import bittensor as bt


w = bt.wallet( name='floppy', hotkey = '3' )
test_endpoint = bt.endpoint(
    version = bt.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 9090,
    hotkey = w.hotkey.ss58_address,
    coldkey = w.coldkeypub.ss58_address,
    modality = 0
)    

m = bt.subtensor().metagraph(1)
mod = bt.text_last_hidden_state( m.endpoint_objs[0] )
print ('done creation.')
import time
time.sleep(3)