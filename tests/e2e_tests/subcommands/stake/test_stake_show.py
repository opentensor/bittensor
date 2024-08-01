from bittensor.commands.stake import StakeShow
from ...utils import setup_wallet


def test_stake_show(local_chain, capsys):
    keypair, exec_command, wallet = setup_wallet("//Alice")

    # Execute the command
    exec_command(StakeShow, ["stake", "show"])
    captured = capsys.readouterr()
    lines = captured.out.split("\n")

    # Ensure there are enough lines
    assert len(lines) >= 5, "Output has fewer than 5 lines."

    # Check the header line
    header = lines[0]
    assert "Coldkey" in header, "Header missing 'Coldkey'."
    assert "Balance" in header, "Header missing 'Balance'."
    assert "Account" in header, "Header missing 'Account'."
    assert "Stake" in header, "Header missing 'Stake'."
    assert "Rate" in header, "Header missing 'Rate'."

    # Check the first line of data
    values1 = lines[1].strip().split()
    assert values1[0] == "default", f"Expected 'default', got {values1[0]}."
    assert (
        values1[1].replace("τ", "") == "1000000.000000"
    ), f"Expected '1000000.000000', got {values1[1]}."

    # Check the second line of data
    values2 = lines[2].strip().split()
    assert values2[0] == "default", f"Expected 'default', got {values2[0]}."
    assert (
        values2[1].replace("τ", "") == "0.000000"
    ), f"Expected '0.000000', got {values2[1]}."
    assert values2[2] == "0/d", f"Expected '0/d', got {values2[2]}."

    # Check the third line of data
    values3 = lines[3].strip().split()
    assert (
        values3[0].replace("τ", "") == "1000000.00000"
    ), f"Expected '1000000.00000', got {values3[0]}."
    assert (
        values3[1].replace("τ", "") == "0.00000"
    ), f"Expected '0.00000', got {values3[1]}."
    assert (
        values3[2].replace("τ", "") == "0.00000/d"
    ), f"Expected '0.00000/d', got {values3[2]}."
