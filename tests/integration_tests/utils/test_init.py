import os
import shutil

from bittensor_wallet import Wallet, Keyfile, Keypair
import pytest

from bittensor import utils


def test_unlock_through_env():
    # Ensure path is clean before we run the tests
    if os.path.exists("/tmp/bittensor-tests-wallets/"):
        shutil.rmtree("/tmp/bittensor-tests-wallets")

    wallet = Wallet(path="/tmp/bittensor-tests-wallets")

    # Set up the coldkey
    cold_kf = Keyfile("/tmp/bittensor-tests-wallets/default/coldkey", name="default")
    kp = Keypair.create_from_mnemonic(
        "stool feel open east woman high can denial forget screen trust salt"
    )
    cold_kf.set_keypair(kp, False, False)
    cold_kf.encrypt("1234password1234")

    # Set up the hotkey
    hot_kf = Keyfile(
        "/tmp/bittensor-tests-wallets/default/hotkeys/default", name="default"
    )
    hkp = Keypair.create_from_mnemonic(
        "stool feel open east woman high can denial forget screen trust salt"
    )
    hot_kf.set_keypair(hkp, False, False)
    hot_kf.encrypt("1234hotkey1234")

    # Save a wrong password to the environment for CK
    cold_kf.save_password_to_env("badpassword")
    result = utils.unlock_key(wallet)
    assert result.success is False

    # Save correct password to the environment for CK
    cold_kf.save_password_to_env("1234password1234")
    result = utils.unlock_key(wallet)
    assert result.success is True

    # Save a wrong password to the environment for HK
    hot_kf.save_password_to_env("badpassword")
    result = utils.unlock_key(wallet, "hotkey")
    assert result.success is False

    # Save correct password to the environment for HK
    hot_kf.save_password_to_env("1234hotkey1234")
    result = utils.unlock_key(wallet, "hotkey")
    assert result.success is True

    with pytest.raises(ValueError):
        utils.unlock_key(wallet, "mycoldkey")

    # Ensure test wallets path is deleted after running tests
    if os.path.exists("/tmp/bittensor-tests-wallets"):
        shutil.rmtree("/tmp/bittensor-tests-wallets")
