import bittensor
from bittensor.utils.balance import Balance 
from termcolor import colored

# Fill in below to name your wallet and keys.
YOUR_WALLET_NAME = 'puma'
YOUR_HOTKEY_NAME = 'default'

# Create the wallet object.
wallet = bittensor.Wallet(
    path = "~/.bittensor/wallets/",
    name = YOUR_WALLET_NAME,
    hotkey = YOUR_HOTKEY_NAME
)
# Assert before continuing
assert wallet.has_hotkey
assert wallet.has_coldkeypub

subtensor = bittensor.Subtensor(
    wallet = wallet,
    network = 'kusanagi'
)
subtensor.connect()

amount = 0.01
destination_public_key = wallet.coldkey.public_key
amount = Balance.from_float( amount )
balance = subtensor.get_balance( wallet.coldkey.public_key )
# if balance < amount:
#     print(colored("Not enough balance ({}) to transfer {}".format(balance, amount), 'red'))
#     quit()

print(colored("Requesting transfer of {} Tao, from coldkey.pub: {} to dest.pub: {}".format(amount.tao, wallet.coldkey.public_key, destination_public_key), 'blue'))
print("Waiting for finalization...",)
result = subtensor.transfer(destination_public_key, amount, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
if result:
    print(colored("Transfer finalized with amount: {} Tao to dest: {} from coldkey.pub: {}".format(amount.tao, destination_public_key, wallet.coldkey.public_key), 'green'))
    new_balance = subtensor.get_balance(wallet.coldkeypub)
    destination_balance = subtensor.get_balance(destination_public_key)
    print(colored("Your coldkey has new balance: {} Tao".format( new_balance.tao ) , 'green'))
    print(colored("The destination has new balance: {} Tao".format( new_balance.tao ) , 'green'))
else:
    print(colored("Transfer failed", 'red'))