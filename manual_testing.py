from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import tao
from bittensor_wallet.wallet import Wallet

COLDKEY = "5G4T9VnZfUDD2vD3MdKsgmbSAsKQq5rtZ73JtQuFA8SRfW14"
HOTKEY = "5GQenM6A7sWGxgmPPAorKGnFQsBWczXDhGhZjkbZq3Wr9wtX"
HOTKEY_NETUIDS = [0, 4, 6]

DEVNET = "wss://dev.chain.opentensor.ai:443"
wallet = Wallet(path="~/.bittensor/wallets", name="alice")


subtensor = Subtensor(network="test")


def main():
    # print(subtensor.get_hyperparameter("LastUpdate", 1))
    # print(subtensor.get_hyperparameter("MinDifficulty", 1))  # does not exist
    # print(subtensor.all_subnets())
    # print(subtensor.blocks_since_last_update(1, 1))
    # print(subtensor.bonds(1))  # looking into
    # print(subtensor.commit_reveal_enabled(1))
    # print(subtensor.difficulty(1))
    # print(subtensor.does_hotkey_exist(HOTKEY))
    # print(subtensor.get_all_subnets_info())
    # print(subtensor.get_balance(COLDKEY))
    # print(subtensor.get_balances(COLDKEY))

    # print(current_block := subtensor.get_current_block())

    # print(subtensor._get_block_hash(current_block))
    # print(subtensor.get_block_hash(current_block))
    # print(subtensor.get_children(HOTKEY, 1))  # maybe not working
    # print(subtensor.get_metagraph_info(1))
    # print(subtensor.subnet(1))
    # print(subtensor.get_all_metagraphs_info())
    # print(subtensor.get_netuids_for_hotkey(HOTKEY))
    # print(subtensor.get_neuron_certificate())  # untested
    # print(subtensor.get_neuron_for_pubkey_and_subnet(HOTKEY, HOTKEY_NETUIDS[1]))
    # print(subtensor.get_stake(COLDKEY, HOTKEY, HOTKEY_NETUIDS[1]))
    # print(subtensor.get_stake_for_coldkey_and_hotkey(COLDKEY, HOTKEY, HOTKEY_NETUIDS))
    # print(subtensor.get_stake_for_coldkey(COLDKEY))
    # print(type(subtensor.get_subnet_burn_cost()))

    # print(subtensor.get_subnet_hyperparameters(111))
    # print(subtensor.get_subnets(block=current_block-20))
    # print(subtensor.get_total_stake_for_hotkey(HOTKEY))
    # print(subtensor.get_total_subnets())
    # print(subtensor.get_transfer_fee(wallet=wallet, dest=COLDKEY, value=tao(1.0)))
    # print(subtensor.get_uid_for_hotkey_on_subnet(HOTKEY, HOTKEY_NETUIDS[1]))
    # print(subtensor.immunity_period(HOTKEY_NETUIDS[1]))
    # print(subtensor.is_hotkey_delegate(COLDKEY))
    # print(subtensor.is_hotkey_registered(HOTKEY, HOTKEY_NETUIDS[1]))
    # print(subtensor.is_hotkey_registered_any(HOTKEY))
    # print(subtensor.is_hotkey_registered_on_subnet(HOTKEY, HOTKEY_NETUIDS[1]))
    # print(subtensor.last_drand_round())
    # print(subtensor.max_weight_limit(HOTKEY_NETUIDS[1]))
    # print(subtensor.metagraph(HOTKEY_NETUIDS[1], lite=True))
    # print(subtensor.metagraph(HOTKEY_NETUIDS[1], lite=False))
    # print(subtensor.min_allowed_weights(HOTKEY_NETUIDS[1]))
    # print(subtensor.neuron_for_uid(1, HOTKEY_NETUIDS[1]))
    # print(subtensor.neurons(HOTKEY_NETUIDS[1]))
    # print(subtensor.neurons_lite(HOTKEY_NETUIDS[1]))
    # print(subtensor.query_identity(HOTKEY))
    # print(subtensor.recycle(HOTKEY_NETUIDS[1]))
    # print(subtensor.subnet_exists(2))
    # print(subtensor.subnet_exists(200))
    # print(subtensor.subnetwork_n(HOTKEY_NETUIDS[1]))
    # print(subtensor.tempo(HOTKEY_NETUIDS[1]))
    # print(subtensor.tx_rate_limit())
    # print(subtensor.wait_for_block())
    # print(subtensor.wait_for_block(current_block+5))
    # for subnet in subtensor.get_subnets():
    #     print(subtensor.weights(subnet))
    # print(subtensor.weights_rate_limit(HOTKEY_NETUIDS[1]))
    print(block := subtensor.block)
    # for uid in range(0, 7):
    #     print(subtensor.get_commitment(2, uid, block=block))
    # print(subtensor.get_current_weight_commit_info(1))
    # print(subtensor.get_delegate_by_hotkey(HOTKEY))
    # print(subtensor.get_delegate_identities())
    # print(subtensor.get_delegate_take("5GP7c3fFazW9GXK8Up3qgu2DJBk8inu4aK9TZy3RuoSWVCMi", block=block))
    # print(subtensor.get_delegated(COLDKEY, block=block))
    # print(subtensor.get_delegates(block=block))
    # print(subtensor.get_existential_deposit(block=block))
    # print(subtensor.get_hotkey_owner("5DcpkDYwWHgtkhLhKFCMqTQeU1ggRKjQEoLCf1e3X47VxZzw"))
    # print(subtensor.get_minimum_required_stake())


if __name__ == "__main__":
    main()
