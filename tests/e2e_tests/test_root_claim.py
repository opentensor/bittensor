from bittensor.core.extrinsics import root
from bittensor_wallet import Wallet
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils import (
    AdminUtils,
    NETUID,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_TEMPO,
)


PROOF_COUNTER = 3


def test_root_claim(
    subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet, fred_wallet
):
    """Tests root claim logic."""
    TEMPO_TO_SET = 10 if subtensor.chain.is_fast_blocks() else 20

    # activate ROOT net to stake on Alice
    rood_sn = TestSubnet(subtensor, 0)
    rood_sn.execute_steps(
        [
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(alice_wallet),
        ]
    )

    sn2 = TestSubnet(subtensor)
    sn2.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    response = subtensor.staking.add_stake_multiple(
        wallet=charlie_wallet,
        netuids=[sn2.netuid] * 5,
        hotkey_ss58s=[alice_wallet.hotkey.ss58_address] * 5,
        amounts=[Balance.from_tao(10)] * 5,
    )
    assert response.success, response.message

    for k, v in response.data.items():
        print(k, v.data)

    stake_balance = Balance.from_tao(10)

    # stake to Alice in ROOT
    response = subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=rood_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    # set claim type to Swap (actually it's already Swap, but just to be sure if default is changed in the future)
    assert subtensor.staking.set_root_claim_type(
        wallet=charlie_wallet, new_root_claim_type="Swap"
    ).success
    assert (
        subtensor.staking.get_root_claim_type(charlie_wallet.coldkey.ss58_address)
        == "Swap"
    )

    proof_counter = PROOF_COUNTER
    prev_root_stake = subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=rood_sn.netuid,
    )
    # proof that ROOT stake is changing each last epoch block
    while proof_counter > 0:
        next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
            rood_sn.netuid
        )
        subtensor.wait_for_block(next_epoch_start_block)
        charlie_root_stake = subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=rood_sn.netuid,
        )
        assert charlie_root_stake > prev_root_stake
        proof_counter -= 1

    # === Set claim type to Keep ===
    assert subtensor.staking.set_root_claim_type(
        wallet=charlie_wallet, new_root_claim_type="Keep"
    ).success
    assert (
        subtensor.staking.get_root_claim_type(charlie_wallet.coldkey.ss58_address)
        == "Keep"
    )

    proof_counter = PROOF_COUNTER
    prev_root_stake = subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=rood_sn.netuid,
    )
    # proof that ROOT stake isn't changes until it's claimed manually
    while proof_counter > 0:
        next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
            rood_sn.netuid
        )
        subtensor.wait_for_block(next_epoch_start_block)
        charlie_root_stake = subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=rood_sn.netuid,
        )
        assert charlie_root_stake == prev_root_stake
        proof_counter -= 1

    # claim ROOT stake
    response = subtensor.staking.claim_root(charlie_wallet)
    assert response.success, response.message
