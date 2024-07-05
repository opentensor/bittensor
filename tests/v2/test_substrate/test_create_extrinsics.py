import os
import pytest

from scalecodec.type_registry import load_type_registry_file
from substrateinterface import SubstrateInterface, Keypair

from bittensor.v2.utils.async_substrate import (
    AsyncSubstrateInterface,
    SubstrateRequestException,
)
import settings


@pytest.fixture
def kusama_substrate():
    return AsyncSubstrateInterface(
        url=settings.KUSAMA_NODE_URL,
    )


@pytest.fixture
def polkadot_substrate():
    return AsyncSubstrateInterface(
        url=settings.POLKADOT_NODE_URL,
    )


@pytest.fixture
def keypair():
    mnemonic = Keypair.generate_mnemonic()
    return Keypair.create_from_mnemonic(mnemonic)


def setUpClass(cls):
    cls.kusama_substrate = SubstrateInterface(
        url=settings.KUSAMA_NODE_URL, ss58_format=2, type_registry_preset="kusama"
    )

    cls.polkadot_substrate = SubstrateInterface(
        url=settings.POLKADOT_NODE_URL, ss58_format=0, type_registry_preset="polkadot"
    )

    module_path = os.path.dirname(__file__)
    cls.metadata_fixture_dict = load_type_registry_file(
        os.path.join(module_path, "fixtures", "metadata_hex.json")
    )

    # Create new keypair
    mnemonic = Keypair.generate_mnemonic()
    cls.keypair = Keypair.create_from_mnemonic(mnemonic)


@pytest.mark.asyncio
async def test_create_extrinsic_metadata_v14(kusama_substrate, keypair):
    # Create balance transfer call
    async with kusama_substrate:
        call = await kusama_substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": "EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk",
                "value": 3 * 10**3,
            },
        )

        extrinsic = await kusama_substrate.create_signed_extrinsic(
            call=call, keypair=keypair, tip=1
        )

        decoded_extrinsic = await kusama_substrate.create_scale_object("Extrinsic")
        decoded_extrinsic.decode(extrinsic.data)

        assert decoded_extrinsic["call"]["call_module"].name == "Balances"
        assert decoded_extrinsic["call"]["call_function"].name == "transfer_keep_alive"
        assert extrinsic["nonce"] == 0
        assert extrinsic["tip"] == 1


@pytest.mark.asyncio
async def test_create_mortal_extrinsic(kusama_substrate, polkadot_substrate, keypair):
    for substrate in [kusama_substrate, polkadot_substrate]:
        # Create balance transfer call
        call = await substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": "EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk",
                "value": 3 * 10**3,
            },
        )

        extrinsic = await substrate.create_signed_extrinsic(
            call=call, keypair=keypair, era={"period": 64}
        )

        try:
            await substrate.submit_extrinsic(extrinsic)
            pytest.fail("Should raise no funds to pay fees exception")

        except SubstrateRequestException:
            # Extrinsic should be successful if account had balance, otherwise 'Bad proof' error should be raised
            pass


@pytest.mark.asyncio
async def test_create_batch_extrinsic(polkadot_substrate, keypair):
    async with polkadot_substrate:
        balance_call = await polkadot_substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": "EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk",
                "value": 3 * 10**3,
            },
        )

        call = await polkadot_substrate.compose_call(
            call_module="Utility",
            call_function="batch",
            call_params={"calls": [balance_call, balance_call]},
        )

        extrinsic = await polkadot_substrate.create_signed_extrinsic(
            call=call, keypair=keypair, era={"period": 64}
        )

        # Decode extrinsic again as test
        extrinsic.decode(extrinsic.data)

    assert "Utility" == extrinsic.value["call"]["call_module"]
    assert "batch" == extrinsic.value["call"]["call_function"]


@pytest.mark.asyncio
async def test_create_unsigned_extrinsic(kusama_substrate):
    async with kusama_substrate:
        call = await kusama_substrate.compose_call(
            call_module="Timestamp",
            call_function="set",
            call_params={
                "now": 1602857508000,
            },
        )

    extrinsic = kusama_substrate.substrate.create_unsigned_extrinsic(call)
    assert str(extrinsic.data) == "0x280402000ba09cc0317501"


@pytest.mark.asyncio
async def test_create_extrinsic_bytes_signature(kusama_substrate, keypair):
    # Create balance transfer call
    async with kusama_substrate:
        call = await kusama_substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": "EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk",
                "value": 3 * 10**3,
            },
        )

        signature_hex = (
            "01741d037f6ea0c5269c6d78cde9505178ee928bb1077db49c684f9d1cad430e767e09808bc556ea2962a7b21a"
            "ada78b3aaf63a8b41e035acfdb0f650634863f83"
        )

        extrinsic = await kusama_substrate.create_signed_extrinsic(
            call=call, keypair=keypair, signature=f"0x{signature_hex}"
        )

        assert extrinsic.value["signature"]["Sr25519"] == f"0x{signature_hex[2:]}"

        extrinsic = await kusama_substrate.create_signed_extrinsic(
            call=call, keypair=keypair, signature=bytes.fromhex(signature_hex)
        )

    assert extrinsic.value["signature"]["Sr25519"] == f"0x{signature_hex[2:]}"
