"""HTTP remote signer for bittensor wallets.

When ``BT_SIGNER_URL`` is set, ``wallet.hotkey`` transparently delegates signing
to an HTTP service, so no key files are required on the validator machine.

The remote signer must expose a single URL with two verbs:

* ``GET``  → ``{"ss58_address": "...", "public_key": "0x...", "ss58_format": 42, "crypto_type": 1}``
* ``POST`` → request ``{"message": "0x..."}`` → response ``{"signature": "0x..."}``

If ``BT_SIGNER_AUTH`` is set, its value is sent as the ``Authorization`` header.
"""

import os
from typing import Optional, Union

import requests
from bittensor_wallet.wallet import Wallet as _BaseWallet


class RemoteKeypair:
    """Keypair-Protocol-compatible class that delegates signing to HTTP.

    Satisfies ``async_substrate_interface.protocols.Keypair`` via duck-typing,
    so instances can be passed wherever a real ``bittensor_wallet.Keypair`` is
    expected (e.g. ``substrate.create_signed_extrinsic``).
    """

    def __init__(self, signer_url: str, signer_auth: Optional[str] = None):
        self._signer_url = signer_url
        self._signer_auth = signer_auth
        self._fetch_identity()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._signer_auth:
            h["Authorization"] = self._signer_auth
        return h

    def _fetch_identity(self) -> None:
        resp = requests.get(self._signer_url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self.ss58_address: str = data["ss58_address"]
        self.public_key: bytes = bytes.fromhex(
            data["public_key"].removeprefix("0x")
        )
        self.ss58_format: int = data.get("ss58_format", 42)
        self.crypto_type: int = data.get("crypto_type", 1)

    def sign(self, data: Union[bytes, str]) -> bytes:
        if isinstance(data, str):
            payload = data
        else:
            payload = f"0x{data.hex()}"
        resp = requests.post(
            self._signer_url,
            json={"message": payload},
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return bytes.fromhex(resp.json()["signature"].removeprefix("0x"))


def _remote_signing_enabled() -> bool:
    return bool(os.environ.get("BT_SIGNER_URL"))


class Wallet(_BaseWallet):
    """Bittensor Wallet with optional HTTP remote signing.

    When ``BT_SIGNER_URL`` is set, ``wallet.hotkey`` returns a
    :class:`RemoteKeypair` instead of loading the hotkey from disk, and
    ``unlock_hotkey()`` becomes a no-op. All other behavior is identical to
    the base ``bittensor_wallet.Wallet``.
    """

    _remote_hotkey_cache: Optional[RemoteKeypair] = None

    @property
    def hotkey(self):
        if _remote_signing_enabled():
            if self._remote_hotkey_cache is None:
                self._remote_hotkey_cache = RemoteKeypair(
                    signer_url=os.environ["BT_SIGNER_URL"],
                    signer_auth=os.environ.get("BT_SIGNER_AUTH"),
                )
            return self._remote_hotkey_cache
        return _BaseWallet.hotkey.fget(self)

    def unlock_hotkey(self, *args, **kwargs):
        if _remote_signing_enabled():
            return
        return super().unlock_hotkey(*args, **kwargs)
