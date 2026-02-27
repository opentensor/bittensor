from async_substrate_interface.types import ScaleObj

from bittensor.core.chain_data.coldkey_swap import (
    ColdkeySwapAnnouncementInfo,
    ColdkeySwapDisputeInfo,
)


def test_coldkey_swap_announcement_info_from_query_none(mocker):
    """Test from_query returns None when query has no value."""
    # Prep
    coldkey_ss58 = mocker.Mock(spec=str)
    query = mocker.Mock(spec=ScaleObj)

    # Call
    from_query = ColdkeySwapAnnouncementInfo.from_query(coldkey_ss58, query)

    # Asserts
    assert from_query is None


def test_coldkey_swap_announcement_info_from_query_happy_path(mocker):
    """Test from_query returns ColdkeySwapAnnouncementInfo when query has valid data."""
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


def test_coldkey_swap_dispute_info_from_query_none(mocker):
    """Test from_query returns None when query has no value."""
    coldkey_ss58 = mocker.Mock(spec=str)
    query = mocker.Mock(spec=ScaleObj)
    query.value = None

    from_query = ColdkeySwapDisputeInfo.from_query(coldkey_ss58, query)

    assert from_query is None


def test_coldkey_swap_dispute_info_from_query_happy_path(mocker):
    """Test from_query returns ColdkeySwapDisputeInfo when query has valid data."""
    coldkey_ss58 = mocker.Mock(spec=str)
    fake_block = 12345
    query = mocker.Mock(spec=ScaleObj, value=fake_block)

    from_query = ColdkeySwapDisputeInfo.from_query(coldkey_ss58, query)

    assert from_query is not None
    assert from_query.coldkey == coldkey_ss58
    assert from_query.disputed_block == fake_block


def test_coldkey_swap_dispute_info_from_record(mocker):
    """Test from_record returns ColdkeySwapDisputeInfo from query_map record."""
    decoded_coldkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    disputed_block = 999
    record = (mocker.Mock(), mocker.Mock(value=disputed_block))
    mocker.patch(
        "bittensor.core.chain_data.coldkey_swap.decode_account_id",
        return_value=decoded_coldkey,
    )

    from_record = ColdkeySwapDisputeInfo.from_record(record)

    assert from_record.coldkey == decoded_coldkey
    assert from_record.disputed_block == disputed_block
