# The MIT License (MIT)
# Copyright © 2022 Yuma Rao

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

import unittest
from unittest.mock import patch, MagicMock
import pytest
import bittensor


class TestWalletReregister(unittest.TestCase):
    def test_wallet_reregister_use_cuda_flag_none(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        config.subtensor.register.cuda.use_cuda = None # don't set the argument, but do specify the flag
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], None) # should be None when no argument, but flag set

    def test_wallet_reregister_use_cuda_flag_true(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        config.subtensor.register.cuda.use_cuda = True
        config.subtensor.register.cuda.dev_id = 0
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], True) # should be default when no argument

    def test_wallet_reregister_use_cuda_flag_false(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        config.subtensor.register.cuda.use_cuda = False
        config.subtensor.register.cuda.dev_id = 0
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], False) # should be default when no argument

    def test_wallet_reregister_use_cuda_flag_not_specified_false(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        #config.subtensor.register.cuda.use_cuda # don't specify the flag
        config.subtensor.register.cuda.dev_id = 0
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], False) # should be False when no flag was set


if __name__ == '__main__':
    unittest.main()