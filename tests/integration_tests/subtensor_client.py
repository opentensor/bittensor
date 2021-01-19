import asyncio

from bittensor.balance import Balance
from bittensor.subtensor.client import WSClient
from bittensor.subtensor.interface import Keypair
from loguru import logger
import pytest

# logger.remove() # Shut up loguru

def connect(keypair):
    socket = "localhost:9944"
    client = WSClient(socket, keypair)

    client.connect()
    return client


def generate_keypair():
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

'''
connect() tests
'''

@pytest.mark.asyncio
async def test_connect_success():
    hotkeypair = Keypair.create_from_uri("//Alice")
    client = connect(hotkeypair)
    result = await client.is_connected()
    assert result is True


'''
subscribe() tests
'''

@pytest.mark.asyncio
async def test_subscribe_success():
    hotkeypair = generate_keypair()
    client = connect(hotkeypair)
    coldkeypair = Keypair.create_from_uri("//Alice")

    await client.is_connected()

    await client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)

    await asyncio.sleep(10) # Sleep for at least one block to give extrinsic the chance to be put onto the blockchain
    uid = await client.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None

'''
get_balance() tests
'''

@pytest.mark.asyncio
async def test_get_balance_success():
    hotkeypair = Keypair.create_from_uri("//Alice")
    client = connect(hotkeypair)

    await client.is_connected()

    result = await client.get_balance(hotkeypair.ss58_address)
    assert result == Balance(pow(10, 9))

@pytest.mark.asyncio
async def test_get_balance_no_balance():
    hotkeypair = generate_keypair()
    client = connect(hotkeypair)

    await client.is_connected()

    result = await client.get_balance(hotkeypair.ss58_address)
    assert result == Balance(0)

'''
get_stake() tests
'''
@pytest.mark.asyncio
async def test_add_stake_success():
    coldkeypair = Keypair.create_from_uri("//Alice")
    hotkeypair  = generate_keypair()

    client = connect(hotkeypair)
    await client.is_connected()

    # Subscribe a new neuron with the hotkey
    await client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)
    await asyncio.sleep(10)

    # Now switch the connection the use the coldkey

    client = connect(coldkeypair)
    await client.is_connected()

    # Get the uid of the new neuron
    uid = await client.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None

    # Get the amount of stake, should be 0
    result = await client.get_stake_for_uid(uid)
    assert int (result) == int(Balance(0))

    # Add stake to new neuron
    result = await client.add_stake(Balance(4000), hotkeypair.public_key)
    logger.info(result)

    assert result is not None
    assert 'extrinsic_hash' in result

    # Wait for the extrinsic to complete
    await asyncio.sleep(10)

    # Get the amount of stake
    result = await client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(4000))


'''
@todo Build more tests. The above only tests the successful putting of the extrinsic on the chain
What is needed is logic that handles errors generated when handling the extrinsic
'''

@pytest.mark.asyncio
async def test_transfer_success():
    coldkey_alice = Keypair.create_from_uri("//Alice")
    coldkey_bob = Keypair.create_from_uri("//Bob")

    client = connect(coldkey_alice)
    await client.is_connected()

    balance_alice = await client.get_balance(coldkey_alice.ss58_address)
    balance_bob   = await client.get_balance(coldkey_bob.ss58_address)

    result = await client.transfer(coldkey_bob.public_key, Balance(pow(10, 4)))
    assert result is not None
    assert 'extrinsic_hash' in result

    # Wait until extrinsic is processed
    await asyncio.sleep(10)

    balance_alice_new = await client.get_balance(coldkey_alice.ss58_address)
    balance_bob_new = await client.get_balance(coldkey_bob.ss58_address)

    assert balance_alice_new < balance_alice
    assert balance_bob_new > balance_bob

@pytest.mark.asyncio
async def test_unstake_success():
    coldkeypair = Keypair.create_from_uri("//Alice")
    hotkeypair = generate_keypair()

    hotkey_client = connect(hotkeypair)
    await hotkey_client.is_connected()

    # Subscribe a new neuron with the hotkey
    await hotkey_client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)
    await asyncio.sleep(10)

    # Now switch the connection the use the coldkey

    coldkey_client = connect(coldkeypair)
    await coldkey_client.is_connected()

    # Get the uid of the new neuron
    uid = await coldkey_client.get_uid_for_pubkey(hotkeypair.public_key)
    logger.error(uid)
    assert uid is not None

    # Get the amount of stake, should be 0
    result = await coldkey_client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(0))

    # Get the balance for the cold key, we use this for later comparison
    balance = await coldkey_client.get_balance(coldkeypair.public_key)

    # Add stake to new neuron
    result = await coldkey_client.add_stake(Balance(4000), hotkeypair.public_key)
    logger.info(result)

    assert result is not None
    assert 'extrinsic_hash' in result

    # Wait for the extrinsic to complete
    await asyncio.sleep(10)

    # Get current balance, should be 4000 less than first balance
    result = await coldkey_client.get_balance(coldkeypair.ss58_address)
    assert int(result) == int(balance) - 4000

    # Get the amount of stake, should be 4000
    result = await coldkey_client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(4000))

    # Now do the actual unstake

    # Reconnect with coldkey account
    coldkey_client =  connect(coldkeypair)
    await coldkey_client.is_connected()

    # Do unstake
    result = await coldkey_client.unstake(Balance(4000), hotkeypair.public_key)
    assert result is not None
    assert 'extrinsic_hash' in result

    await asyncio.sleep(10)

    # Check if balance is the same as what we started with
    new_balance = await coldkey_client.get_balance(coldkeypair.ss58_address)
    assert int(new_balance) == int(balance)







# @todo unstake()
# @todo set_weights()
# @todo emit() prolly kill
# @todo get_current_block()
# @todo get_active()
# @todo get_uid_for_pubkey()
# @todo get_neuron_for_uid()
# @todo neurons()
# @todo get_stake_for_uid()
# @todo weight_uids_for_uid()
# @todo weight_vals_for_uid
# @todo get_last_emit_data_for_uid()
# @todo get_last_emit_data()