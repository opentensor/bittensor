import bittensor
import argparse


def test_py_config_parsed_successfully_rust_wallet():
    """Verify that python based config object is successfully parsed with rust-based wallet object."""
    # Preps
    parser = argparse.ArgumentParser()

    bittensor.wallet.add_args(parser)
    bittensor.subtensor.add_args(parser)
    bittensor.axon.add_args(parser)
    bittensor.logging.add_args(parser)

    config = bittensor.config(parser)

    # since we can't apply mocking to rust implewmented object then replace those directly
    config.wallet.name = "new_wallet_name"
    config.wallet.hotkey = "new_hotkey"
    config.wallet.path = "/some/not_default/path"

    # Pass in the whole bittensor config
    wallet = bittensor.wallet(config=config)
    assert wallet.name == config.wallet.name
    assert wallet.hotkey_str == config.wallet.hotkey
    assert wallet.path == config.wallet.path

    # Pass in only the btwallet's config
    wallet_two = bittensor.wallet(config=config.wallet)
    assert wallet_two.name == config.wallet.name
    assert wallet_two.hotkey_str == config.wallet.hotkey
    assert wallet_two.path == config.wallet.path
