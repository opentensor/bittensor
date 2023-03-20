import json
import bittensor
mnemonic = input("The mnemonic of your validator's Hotkey ( see file: ~/.bittensor/wallets/<coldkey>/hotkeys/<validator> ) : ")
descriptive_name = input("Your validator's descriptive name (i.e. Opentensor Foundation): ")
url = input("Your validator url (i.e. www.opentensor.org ): ")
description = input("A short description for your validator ( i.e. Build, maintain and advance Bittensor): ")
keypair = bittensor.Keypair.create_from_mnemonic(mnemonic)
dictionary = {}
dictionary[ keypair.ss58_address ] = {
    'name': descriptive_name,
    'url': url,
    'description': description,
}
message = json.dumps( dictionary )
signature = keypair.sign( data = message )
print('\n\n\tVerified', bittensor.Keypair(ss58_address=keypair.ss58_address).verify( data = message, signature = signature) )
print (
    "\tValidator information: {}\n".format(message),
    "\tValidator signature: {}\n\n".format(signature.hex()),
)