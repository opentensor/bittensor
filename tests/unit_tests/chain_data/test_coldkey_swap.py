from bittensor.core.chain_data.coldkey_swap import (
    ColdkeySwapAnnouncementInfo,
    ColdkeySwapConstants,
)


def test_coldkey_swap_announcement_info_from_query_map_record(mocker):
    """Test ColdkeySwapAnnouncementInfo.from_query_map_record parses query_map record correctly."""
    # Prep
    fake_account_id = b"\x00" * 32
    fake_execution_block = 1000
    fake_hash = b"\x11" * 32

    record = (
        fake_account_id,
        mocker.Mock(value=(fake_execution_block, fake_hash)),
    )

    # Call
    coldkey_ss58, info = ColdkeySwapAnnouncementInfo.from_query_map_record(record)

    # Asserts
    assert isinstance(coldkey_ss58, str)
    assert coldkey_ss58.startswith("5")
    assert info.coldkey == coldkey_ss58
    assert info.execution_block == fake_execution_block
    assert info.new_coldkey_hash == "0x" + fake_hash.hex()


def test_coldkey_swap_announcement_info_from_query(mocker):
    """Test ColdkeySwapAnnouncementInfo.from_query parses query result correctly."""
    # Prep
    fake_execution_block = 1000
    fake_hash = b"\x11" * 32

    query = mocker.Mock(value=(fake_execution_block, fake_hash))

    # Call
    info = ColdkeySwapAnnouncementInfo.from_query(query)

    # Asserts
    assert info is not None
    assert info.execution_block == fake_execution_block
    assert info.new_coldkey_hash == "0x" + fake_hash.hex()
    assert info.coldkey == ""  # Should be empty as per implementation


def test_coldkey_swap_announcement_info_from_query_none(mocker):
    """Test ColdkeySwapAnnouncementInfo.from_query returns None when query.value is None."""
    # Prep
    query = mocker.Mock(value=None)

    # Call
    info = ColdkeySwapAnnouncementInfo.from_query(query)

    # Asserts
    assert info is None


def test_coldkey_swap_constants_from_dict():
    """Test ColdkeySwapConstants.from_dict creates instance from dictionary."""
    # Prep
    data = {
        "ColdkeySwapAnnouncementDelay": 100,
        "ColdkeySwapReannouncementDelay": 200,
        "KeySwapCost": 1000000,
    }

    # Call
    constants = ColdkeySwapConstants.from_dict(data)

    # Asserts
    assert constants.ColdkeySwapAnnouncementDelay == 100
    assert constants.ColdkeySwapReannouncementDelay == 200
    assert constants.KeySwapCost == 1000000


def test_coldkey_swap_constants_from_dict_partial():
    """Test ColdkeySwapConstants.from_dict handles partial data."""
    # Prep
    data = {
        "ColdkeySwapAnnouncementDelay": 100,
        # Missing other fields
    }

    # Call
    constants = ColdkeySwapConstants.from_dict(data)

    # Asserts
    assert constants.ColdkeySwapAnnouncementDelay == 100
    assert constants.ColdkeySwapReannouncementDelay is None
    assert constants.KeySwapCost is None


def test_coldkey_swap_constants_to_dict():
    """Test ColdkeySwapConstants.to_dict converts instance to dictionary."""
    # Prep
    constants = ColdkeySwapConstants(
        ColdkeySwapAnnouncementDelay=100,
        ColdkeySwapReannouncementDelay=200,
        KeySwapCost=1000000,
    )

    # Call
    result = constants.to_dict()

    # Asserts
    assert isinstance(result, dict)
    assert result["ColdkeySwapAnnouncementDelay"] == 100
    assert result["ColdkeySwapReannouncementDelay"] == 200
    assert result["KeySwapCost"] == 1000000


def test_coldkey_swap_constants_names():
    """Test ColdkeySwapConstants.constants_names returns list of constant names."""
    # Call
    names = ColdkeySwapConstants.constants_names()

    # Asserts
    assert isinstance(names, list)
    assert "ColdkeySwapAnnouncementDelay" in names
    assert "ColdkeySwapReannouncementDelay" in names
    assert "KeySwapCost" in names
    assert len(names) == 3
