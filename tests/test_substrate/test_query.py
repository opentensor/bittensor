import pytest
from bittensor.utils.async_substrate import AsyncSubstrateInterface
import settings


@pytest.fixture
def kusama_substrate():
    return AsyncSubstrateInterface(
        chain_endpoint=settings.KUSAMA_NODE_URL,
    )


@pytest.fixture
def polkadot_substrate():
    return AsyncSubstrateInterface(
        chain_endpoint=settings.POLKADOT_NODE_URL,
    )


@pytest.mark.asyncio
async def test_system_account(kusama_substrate):
    async with kusama_substrate:
        result = await kusama_substrate.query(
            module="System",
            storage_function="Account",
            params=["F4xQKRUagnSGjFqafyhajLs94e7Vvzvr8ebwYJceKpr8R7T"],
            block_hash="0xbf787e2f322080e137ed53e763b1cc97d5c5585be1f736914e27d68ac97f5f2c",
        )

        assert result.value["nonce"] == 67501
        assert result.value["data"]["free"] == 1099945000512
        assert result.meta_info["result_found"] is True


@pytest.mark.asyncio
async def test_system_account_non_existing(kusama_substrate):
    async with kusama_substrate:
        result = await kusama_substrate.query(
            module="System",
            storage_function="Account",
            params=["GSEX8kR4Kz5UZGhvRUCJG93D5hhTAoVZ5tAe6Zne7V42DSi"],
        )

        assert result.value == {
            "nonce": 0,
            "consumers": 0,
            "providers": 0,
            "sufficients": 0,
            "data": {
                "free": 0,
                "reserved": 0,
                "frozen": 0,
                "flags": 170141183460469231731687303715884105728,
            },
        }


@pytest.mark.asyncio
async def test_non_existing_query(kusama_substrate):
    async with kusama_substrate:
        with pytest.raises(Exception) as excinfo:
            await kusama_substrate.query("Unknown", "StorageFunction")
        assert str(excinfo.value) == 'Pallet "Unknown" not found'


@pytest.mark.asyncio
async def test_missing_params(kusama_substrate):
    async with kusama_substrate:
        with pytest.raises(ValueError) as excinfo:
            await kusama_substrate.query("System", "Account")
        assert "parameters" in str(excinfo.value)


@pytest.mark.asyncio
async def test_modifier_default_result(kusama_substrate):
    async with kusama_substrate:
        result = await kusama_substrate.query(
            module="Staking",
            storage_function="HistoryDepth",
            block_hash="0x4b313e72e3a524b98582c31cd3ff6f7f2ef5c38a3c899104a833e468bb1370a2",
        )

    assert result.value == 84
    assert result.meta_info["result_found"] is False


@pytest.mark.asyncio
async def test_modifier_option_result(kusama_substrate):
    async with kusama_substrate:
        result = await kusama_substrate.query(
            module="Identity",
            storage_function="IdentityOf",
            params=["DD6kXYJPHbPRbBjeR35s1AR7zDh7W2aE55EBuDyMorQZS2a"],
            block_hash="0x4b313e72e3a524b98582c31cd3ff6f7f2ef5c38a3c899104a833e468bb1370a2",
        )

    assert result.value is None
    assert result.meta_info["result_found"] is False


@pytest.mark.asyncio
async def test_identity_hasher(kusama_substrate):
    async with kusama_substrate:
        result = await kusama_substrate.query(
            "Claims", "Claims", ["0x00000a9c44f24e314127af63ae55b864a28d7aee"]
        )
    assert result.value == 45880000000000
