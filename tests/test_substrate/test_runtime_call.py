import pytest
from bittensor.utils.async_substrate import AsyncSubstrateInterface
from substrateinterface import Keypair
import settings


@pytest.fixture
def kusama_substrate():
    return AsyncSubstrateInterface(
        chain_endpoint=settings.KUSAMA_NODE_URL,
    )


@pytest.fixture
def substrate():
    return AsyncSubstrateInterface(
        chain_endpoint=settings.POLKADOT_NODE_URL,
    )


@pytest.fixture(scope="module")
def keypair():
    mnemonic = Keypair.generate_mnemonic()
    return Keypair.create_from_mnemonic(mnemonic)


@pytest.mark.asyncio
async def test_core_version(substrate):
    async with substrate:
        result = await substrate.runtime_call("Core", "version")

    assert result.value['spec_version'] > 0
    assert result.value['spec_name'] == 'polkadot'


@pytest.mark.asyncio
async def test_core_version_at_not_best_block(substrate):
    async with substrate:
        parent_hash = substrate.substrate.get_block_header()
        parent_hash = parent_hash['header']['parentHash']
        result = await substrate.runtime_call("Core", "version", block_hash=parent_hash)

    assert result.value['spec_version'] > 0
    assert result.value['spec_name'] == 'polkadot'


@pytest.mark.asyncio
async def test_metadata_call_info(substrate):
    async with substrate:
        runtime_call = substrate.substrate.get_metadata_runtime_call_function("TransactionPaymentApi", "query_fee_details")
        param_info = runtime_call.get_param_info()
    assert param_info[0] == 'Extrinsic'
    assert param_info[1] == 'u32'

    async with substrate:
        runtime_call = substrate.substrate.get_metadata_runtime_call_function("Core", "initialise_block")
        param_info = runtime_call.get_param_info()
    assert param_info[0]['number'] == 'u32'
    assert param_info[0]['parent_hash'] == 'h256'


@pytest.mark.asyncio
async def test_unknown_runtime_call(substrate):
    async with substrate:
        with pytest.raises(ValueError):
            await substrate.runtime_call("Foo", "bar")


if __name__ == '__main__':
    pytest.main()