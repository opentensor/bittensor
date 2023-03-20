import json
import bittensor
import binascii
information_str = input("Validator information: ")
signature_hex = input("Validator signature: ").encode()
information_dict = json.loads(information_str)
print (str(list(information_dict.keys())[0]))
keypair = bittensor.Keypair(ss58_address=str(list(information_dict.keys())[0]))
print ('Verified', keypair.verify( data = information_str, signature = binascii.unhexlify( signature_hex ) ))