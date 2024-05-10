import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import subtensor as Subtensor
from bittensor.wallet import wallet as Wallet


# Mock the bittensor and related modules to avoid real network calls and wallet operations
@pytest.fixture
def mock_subtensor():
    subtensor = MagicMock(spec=Subtensor)
    subtensor.get_balance.return_value = 100
    subtensor.get_subnet_burn_cost.return_value = 10
    return subtensor


@pytest.fixture
def mock_wallet():
    wallet = MagicMock(spec=Wallet)
    wallet.coldkeypub.ss58_address = "fake_address"
    wallet.coldkey = MagicMock()
    return wallet


@pytest.mark.parametrize(
    "test_id, wait_for_inclusion, wait_for_finalization, prompt, expected",
    [
        ("happy-path-01", False, True, False, True),
        ("happy-path-02", True, False, False, True),
        ("happy-path-03", False, False, False, True),
        ("happy-path-04", True, True, False, True),
    ],
)
def test_register_subnetwork_extrinsic_happy_path(
    mock_subtensor,
    mock_wallet,
    test_id,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    expected,
):
    # Arrange
    mock_subtensor.substrate = MagicMock(
        get_block_hash=MagicMock(return_value="0x" + "0" * 64),
        submit_extrinsic=MagicMock(),
    )
    mock_subtensor.substrate.submit_extrinsic.return_value.is_success = True

    # Act
    from bittensor.extrinsics.network import register_subnetwork_extrinsic

    result = register_subnetwork_extrinsic(
        mock_subtensor, mock_wallet, wait_for_inclusion, wait_for_finalization, prompt
    )

    # Assert
    assert result == expected


# Edge cases
@pytest.mark.parametrize(
    "test_id, balance, burn_cost, prompt_input, expected",
    [
        ("edge-case-01", 0, 10, False, False),  # Balance is zero
        ("edge-case-02", 10, 10, False, False),  # Balance equals burn cost
        ("edge-case-03", 9, 10, False, False),  # Balance less than burn cost
        ("edge-case-04", 100, 10, True, True),  # User declines prompt
    ],
)
def test_register_subnetwork_extrinsic_edge_cases(
    mock_subtensor,
    mock_wallet,
    test_id,
    balance,
    burn_cost,
    prompt_input,
    expected,
    monkeypatch,
):
    # Arrange
    mock_subtensor.get_balance.return_value = balance
    mock_subtensor.get_subnet_burn_cost.return_value = burn_cost
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda x: prompt_input)
    mock_subtensor.substrate = MagicMock()

    # Act
    from bittensor.extrinsics.network import register_subnetwork_extrinsic

    result = register_subnetwork_extrinsic(mock_subtensor, mock_wallet, prompt=True)

    # Assert
    assert result == expected
