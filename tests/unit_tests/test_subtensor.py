# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import unittest.mock as mock
from unittest.mock import MagicMock
import pytest

import bittensor
import unittest


class TestSubtensorWithExternalAxon(unittest.TestCase):
    """
    Test the subtensor with external axon in the config
    """

    def test_serve_axon_with_external_ip_set(self):
        internal_ip: str = "this is an internal ip"
        external_ip: str = "this is an external ip"

        mock_serve_axon = MagicMock(return_value=True)

        mock_subtensor = MagicMock(spec=bittensor.subtensor, serve_axon=mock_serve_axon)

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_wallet = MagicMock(
            spec=bittensor.wallet,
            coldkey=MagicMock(),
            coldkeypub=MagicMock(
                # mock ss58 address
                ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
            ),
            hotkey=MagicMock(
                ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
            ),
        )

        mock_config = bittensor.axon.config()
        mock_axon_with_external_ip_set = bittensor.axon(
            wallet=mock_wallet,
            ip=internal_ip,
            external_ip=external_ip,
            config=mock_config,
        )

        mock_subtensor.serve_axon(
            netuid=-1,
            axon=mock_axon_with_external_ip_set,
        )

        mock_serve_axon.assert_called_once()

        # verify that the axon is served to the network with the external ip
        _, kwargs = mock_serve_axon.call_args
        axon_info = kwargs["axon"].info()
        self.assertEqual(axon_info.ip, external_ip)

    def test_serve_axon_with_external_port_set(self):
        external_ip: str = "this is an external ip"

        internal_port: int = 1234
        external_port: int = 5678

        mock_serve = MagicMock(return_value=True)

        mock_serve_axon = MagicMock(return_value=True)

        mock_subtensor = MagicMock(
            spec=bittensor.subtensor,
            serve=mock_serve,
            serve_axon=mock_serve_axon,
        )

        mock_wallet = MagicMock(
            spec=bittensor.wallet,
            coldkey=MagicMock(),
            coldkeypub=MagicMock(
                # mock ss58 address
                ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
            ),
            hotkey=MagicMock(
                ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
            ),
        )

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_config = bittensor.axon.config()

        mock_axon_with_external_port_set = bittensor.axon(
            wallet=mock_wallet,
            port=internal_port,
            external_port=external_port,
            config=mock_config,
        )

        with mock.patch(
            "bittensor.utils.networking.get_external_ip", return_value=external_ip
        ):
            # mock the get_external_ip function to return the external ip
            mock_subtensor.serve_axon(
                netuid=-1,
                axon=mock_axon_with_external_port_set,
            )

        mock_serve_axon.assert_called_once()
        # verify that the axon is served to the network with the external port
        _, kwargs = mock_serve_axon.call_args
        axon_info = kwargs["axon"].info()
        self.assertEqual(axon_info.port, external_port)


class ExitEarly(Exception):
    """Mock exception to exit early from the called code"""

    pass


class TestStakeMultiple(unittest.TestCase):
    """
    Test the stake_multiple function
    """

    def test_stake_multiple(self):
        mock_amount: bittensor.Balance = bittensor.Balance.from_tao(1.0)

        mock_wallet = MagicMock(
            spec=bittensor.wallet,
            coldkey=MagicMock(),
            coldkeypub=MagicMock(
                # mock ss58 address
                ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
            ),
            hotkey=MagicMock(
                ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
            ),
        )

        mock_hotkey_ss58s = ["5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"]

        mock_amounts = [mock_amount]  # more than 1000 RAO

        mock_neuron = MagicMock(
            is_null=False,
        )

        mock_do_stake = MagicMock(side_effect=ExitEarly)

        mock_subtensor = MagicMock(
            spec=bittensor.subtensor,
            network="mock_net",
            get_balance=MagicMock(
                return_value=bittensor.Balance.from_tao(mock_amount.tao + 20.0)
            ),  # enough balance to stake
            get_neuron_for_pubkey_and_subnet=MagicMock(return_value=mock_neuron),
            _do_stake=mock_do_stake,
        )

        with pytest.raises(ExitEarly):
            bittensor.subtensor.add_stake_multiple(
                mock_subtensor,
                wallet=mock_wallet,
                hotkey_ss58s=mock_hotkey_ss58s,
                amounts=mock_amounts,
            )

            mock_do_stake.assert_called_once()
            # args, kwargs
            _, kwargs = mock_do_stake.call_args
            self.assertAlmostEqual(
                kwargs["ammount"], mock_amount.rao, delta=1.0 * 1e9
            )  # delta of 1.0 TAO


if __name__ == "__main__":
    unittest.main()
