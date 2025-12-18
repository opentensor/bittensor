from bittensor.core.chain_data.coldkey_swap import (
    ColdkeySwapAnnouncementInfo,
    ColdkeySwapConstants,
)
from async_substrate_interface.types import ScaleObj


def test_coldkey_swap_announcement_info_from_query_none(mocker):
    # Prep
    coldkey_ss58 = mocker.Mock(spec=str)
    query = mocker.Mock(spec=ScaleObj)

    # Call
    from_query = ColdkeySwapAnnouncementInfo.from_query(coldkey_ss58, query)

    # Asserts
    assert from_query is None


def test_coldkey_swap_announcement_info_from_query_happy_path(mocker):
    # Prep
    coldkey_ss58 = mocker.Mock(spec=str)
    fake_block = mocker.Mock(spec=int)
    fake_hash_data = mocker.Mock(spec=list)
    query = mocker.Mock(value=(fake_block, (fake_hash_data,)))

    mocked_bytes = mocker.patch("bittensor.core.chain_data.coldkey_swap.bytes")

    # Call
    from_query = ColdkeySwapAnnouncementInfo.from_query(coldkey_ss58, query)

    # Asserts
    mocked_bytes.assert_called_once_with(fake_hash_data)
    assert from_query is not None, "Should return ColdkeySwapAnnouncementInfo object"
    assert from_query.coldkey == coldkey_ss58
    assert from_query.execution_block == fake_block
    assert (
        from_query.new_coldkey_hash
        == mocked_bytes.return_value.hex.return_value.__radd__.return_value
    )
