import os
import shutil

from bittensor_wallet import Wallet, Keyfile, Keypair
import pytest

from bittensor import utils


def test_unlock_key(monkeypatch):
    # Ensure path is clean before we run the tests
    if os.path.exists("/tmp/bittensor-tests-wallets"):
        shutil.rmtree("/tmp/bittensor-tests-wallets")

    wallet = Wallet(path="/tmp/bittensor-tests-wallets")
    cold_kf = Keyfile("/tmp/bittensor-tests-wallets/default/coldkey", name="default")
    kp = Keypair.create_from_mnemonic(
        "stool feel open east woman high can denial forget screen trust salt"
    )
    cold_kf.set_keypair(kp, False, False)
    cold_kf.encrypt("1234password1234")
    hot_kf = Keyfile("/tmp/bittensor-tests-wallets/default/hotkey", name="default")
    hkp = Keypair.create_from_mnemonic(
        "stool feel open east woman high can denial forget screen trust salt"
    )
    hot_kf.set_keypair(hkp, False, False)
    hot_kf.encrypt("1234hotkey1234")
    monkeypatch.setattr("getpass.getpass", lambda _: "badpassword1234")
    result = utils.unlock_key(wallet)
    assert result.success is False
    monkeypatch.setattr("getpass.getpass", lambda _: "1234password1234")
    result = utils.unlock_key(wallet)
    assert result.success is True
    monkeypatch.setattr("getpass.getpass", lambda _: "badpassword1234")
    result = utils.unlock_key(wallet, "hot")
    assert result.success is False
    with pytest.raises(ValueError):
        utils.unlock_key(wallet, "mycoldkey")

    # Ensure test wallets path is deleted after running tests
    if os.path.exists("/tmp/bittensor-tests-wallets"):
        shutil.rmtree("/tmp/bittensor-tests-wallets")
