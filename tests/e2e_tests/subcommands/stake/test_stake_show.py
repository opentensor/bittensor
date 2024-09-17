from bittensor import logging
from bittensor.commands.stake import StakeShow

from ...utils import setup_wallet


def test_stake_show(local_chain, capsys):
    logging.info("Testing test_stake_show")
    keypair, exec_command, wallet = setup_wallet("//Alice")

    # Execute the command
    exec_command(StakeShow, ["stake", "show"])
    captured = capsys.readouterr()
    output = captured.out

    # Check the header line
    assert "Coldkey" in output, "Output missing 'Coldkey'."
    assert "Balance" in output, "Output missing 'Balance'."
    assert "Account" in output, "Output missing 'Account'."
    assert "Stake" in output, "Output missing 'Stake'."
    assert "Rate" in output, "Output missing 'Rate'."

    # Check the first line of data
    assert "default" in output, "Output missing 'default'."
    assert "1000000.000000" in output.replace(
        "τ", ""
    ), "Output missing '1000000.000000'."

    # Check the second line of data
    assert "0.000000" in output.replace("τ", ""), "Output missing '0.000000'."
    assert "0/d" in output, "Output missing '0/d'."

    # Check the third line of data

    assert "1000000.00000" in output.replace("τ", ""), "Output missing '1000000.00000'."
    assert "0.00000" in output.replace("τ", ""), "Output missing '0.00000'."
    assert "0.00000/d" in output.replace("τ", ""), "Output missing '0.00000/d'."
