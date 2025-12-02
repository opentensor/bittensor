"""
Comprehensive unit tests for the WeightCommitInfo class.

This test suite validates the WeightCommitInfo dataclass structure and its
class methods (from_vec_u8 and from_vec_u8_v2) which are critical for
decoding weight commit information from the blockchain.

The WeightCommitInfo class is used throughout the Bittensor codebase to
represent weight commit data, which is essential for the commit-reveal
scheme used in weight setting operations.
"""

import pytest
from unittest.mock import patch, MagicMock

from bittensor.core.chain_data.weight_commit_info import WeightCommitInfo


class TestWeightCommitInfoDataclass:
    """
    Test suite for WeightCommitInfo dataclass instantiation and attributes.

    These tests verify that the dataclass can be properly instantiated with
    valid data and that all attributes are correctly assigned and accessible.
    """

    def test_weight_commit_info_instantiation_with_all_fields(self):
        """
        Test that WeightCommitInfo can be instantiated with all required fields.

        This test verifies the basic dataclass functionality, ensuring that
        all four attributes (ss58, commit_block, commit_hex, reveal_round)
        can be set during instantiation and are correctly stored.
        """
        # Arrange: Prepare test data with all fields populated
        ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        commit_block = 12345
        commit_hex = "0x1234567890abcdef"
        reveal_round = 42

        # Act: Create a WeightCommitInfo instance
        weight_commit_info = WeightCommitInfo(
            ss58=ss58_address,
            commit_block=commit_block,
            commit_hex=commit_hex,
            reveal_round=reveal_round,
        )

        # Assert: Verify all attributes are correctly assigned
        assert weight_commit_info.ss58 == ss58_address, (
            "SS58 address should match input"
        )
        assert weight_commit_info.commit_block == commit_block, (
            "Commit block should match input"
        )
        assert weight_commit_info.commit_hex == commit_hex, (
            "Commit hex should match input"
        )
        assert weight_commit_info.reveal_round == reveal_round, (
            "Reveal round should match input"
        )

    def test_weight_commit_info_instantiation_with_none_commit_block(self):
        """
        Test that WeightCommitInfo can be instantiated with None commit_block.

        The commit_block field is Optional[int], so it should accept None values.
        This is important because some weight commit operations may not have
        an associated block number at the time of creation.
        """
        # Arrange: Prepare test data with commit_block as None
        ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        commit_hex = "0xabcdef1234567890"
        reveal_round = 10

        # Act: Create a WeightCommitInfo instance with None commit_block
        weight_commit_info = WeightCommitInfo(
            ss58=ss58_address,
            commit_block=None,
            commit_hex=commit_hex,
            reveal_round=reveal_round,
        )

        # Assert: Verify attributes are correctly assigned, including None commit_block
        assert weight_commit_info.ss58 == ss58_address, (
            "SS58 address should match input"
        )
        assert weight_commit_info.commit_block is None, "Commit block should be None"
        assert weight_commit_info.commit_hex == commit_hex, (
            "Commit hex should match input"
        )
        assert weight_commit_info.reveal_round == reveal_round, (
            "Reveal round should match input"
        )

    def test_weight_commit_info_instantiation_with_zero_values(self):
        """
        Test that WeightCommitInfo handles zero values correctly.

        This test ensures that zero is a valid value for commit_block and
        reveal_round, which is important for edge cases in blockchain operations.
        """
        # Arrange: Prepare test data with zero values
        ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        commit_block = 0
        commit_hex = "0x0000000000000000"
        reveal_round = 0

        # Act: Create a WeightCommitInfo instance with zero values
        weight_commit_info = WeightCommitInfo(
            ss58=ss58_address,
            commit_block=commit_block,
            commit_hex=commit_hex,
            reveal_round=reveal_round,
        )

        # Assert: Verify zero values are correctly stored
        assert weight_commit_info.commit_block == 0, (
            "Zero commit block should be accepted"
        )
        assert weight_commit_info.reveal_round == 0, (
            "Zero reveal round should be accepted"
        )
        assert weight_commit_info.commit_hex == commit_hex, (
            "Commit hex should match input"
        )

    def test_weight_commit_info_instantiation_with_large_values(self):
        """
        Test that WeightCommitInfo handles large integer values correctly.

        This test verifies that the dataclass can handle large block numbers
        and round numbers that might occur in production blockchain environments.
        """
        # Arrange: Prepare test data with large values
        ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        commit_block = 999999999  # Large block number
        commit_hex = "0x" + "ff" * 32  # 32 bytes of 0xff
        reveal_round = 2147483647  # Max 32-bit signed integer

        # Act: Create a WeightCommitInfo instance with large values
        weight_commit_info = WeightCommitInfo(
            ss58=ss58_address,
            commit_block=commit_block,
            commit_hex=commit_hex,
            reveal_round=reveal_round,
        )

        # Assert: Verify large values are correctly stored
        assert weight_commit_info.commit_block == commit_block, (
            "Large commit block should be accepted"
        )
        assert weight_commit_info.reveal_round == reveal_round, (
            "Large reveal round should be accepted"
        )
        assert len(weight_commit_info.commit_hex) == 66, (
            "Commit hex should be 0x + 64 hex chars"
        )


class TestWeightCommitInfoFromVecU8:
    """
    Test suite for WeightCommitInfo.from_vec_u8 class method.

    The from_vec_u8 method is used when querying blocks where the storage
    function CRV3WeightCommitsV2 does not exist in the Subtensor module.
    This method returns a tuple of (ss58, commit_hex, round_number) without
    the commit_block information.
    """

    def test_from_vec_u8_with_tuple_account_id(self):
        """
        Test from_vec_u8 with account_id as a tuple.

        When account_id comes from the blockchain, it may be wrapped in a
        tuple. This test verifies that the method correctly handles this
        nested tuple structure.
        """
        # Arrange: Prepare test data with account_id as a tuple
        # AccountId is typically 32 bytes, represented as a tuple of integers
        account_id_tuple = (
            (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ),
        )
        commit_data_tuple = ((0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0),)
        round_number = 5
        data = (account_id_tuple, commit_data_tuple, round_number)

        # Mock decode_account_id to return a predictable SS58 address
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0x123456789abcdef0"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8 with tuple-wrapped account_id
            result = WeightCommitInfo.from_vec_u8(data)

            # Assert: Verify the result is a tuple with correct structure
            assert isinstance(result, tuple), "Result should be a tuple"
            assert len(result) == 3, (
                "Result should contain 3 elements (ss58, commit_hex, round_number)"
            )

            ss58, commit_hex, round_num = result
            assert ss58 == expected_ss58, "SS58 address should match decoded account_id"
            assert commit_hex == expected_commit_hex, (
                "Commit hex should be correctly formatted"
            )
            assert round_num == round_number, "Round number should match input"

            # Verify decode_account_id was called with the unwrapped account_id
            mock_decode.assert_called_once()
            call_arg = mock_decode.call_args[0][0]
            assert call_arg == account_id_tuple[0], (
                "decode_account_id should receive unwrapped account_id"
            )

    def test_from_vec_u8_with_non_tuple_account_id(self):
        """
        Test from_vec_u8 with account_id as a direct value (not wrapped in tuple).

        Sometimes account_id may come as a direct value rather than wrapped
        in a tuple. This test ensures the method handles both cases correctly.
        """
        # Arrange: Prepare test data with account_id as direct value
        account_id_direct = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_data_direct = (0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF)
        round_number = 10
        data = (account_id_direct, commit_data_direct, round_number)

        # Mock decode_account_id
        expected_ss58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        expected_commit_hex = "0xaabbccddeeff"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8 with non-tuple account_id
            result = WeightCommitInfo.from_vec_u8(data)

            # Assert: Verify the result is correct
            ss58, commit_hex, round_num = result
            assert ss58 == expected_ss58, "SS58 address should match decoded account_id"
            assert commit_hex == expected_commit_hex, (
                "Commit hex should be correctly formatted"
            )
            assert round_num == round_number, "Round number should match input"

    def test_from_vec_u8_with_tuple_commit_data(self):
        """
        Test from_vec_u8 with commit_data wrapped in a tuple.

        Commit data may come wrapped in a tuple from the blockchain.
        This test verifies that the method correctly unwraps and processes
        tuple-wrapped commit data.
        """
        # Arrange: Prepare test data with commit_data as a tuple
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_data_tuple = ((0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF),)
        round_number = 15
        data = (account_id, commit_data_tuple, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0x0123456789abcdef"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8 with tuple-wrapped commit_data
            result = WeightCommitInfo.from_vec_u8(data)

            # Assert: Verify commit_data was correctly unwrapped and formatted
            ss58, commit_hex, round_num = result
            assert commit_hex == expected_commit_hex, (
                "Commit hex should be correctly formatted from tuple data"
            )
            assert round_num == round_number, "Round number should match input"

    def test_from_vec_u8_with_non_tuple_commit_data(self):
        """
        Test from_vec_u8 with commit_data as a direct tuple (not wrapped).

        Commit data may come as a direct tuple of bytes. This test ensures
        the method handles both wrapped and unwrapped commit data correctly.
        """
        # Arrange: Prepare test data with commit_data as direct tuple
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_data_direct = (0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA)
        round_number = 20
        data = (account_id, commit_data_direct, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0xffeeddccbbaa"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8 with non-tuple commit_data
            result = WeightCommitInfo.from_vec_u8(data)

            # Assert: Verify commit_data was correctly processed
            ss58, commit_hex, round_num = result
            assert commit_hex == expected_commit_hex, (
                "Commit hex should be correctly formatted from direct data"
            )
            assert round_num == round_number, "Round number should match input"

    def test_from_vec_u8_with_empty_commit_data(self):
        """
        Test from_vec_u8 with empty commit_data.

        Edge case: commit_data may be empty. This test verifies that the
        method correctly handles empty commit data and produces a valid
        hex string (0x with no data).
        """
        # Arrange: Prepare test data with empty commit_data
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_data_empty = ()
        round_number = 0
        data = (account_id, commit_data_empty, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0x"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8 with empty commit_data
            result = WeightCommitInfo.from_vec_u8(data)

            # Assert: Verify empty commit_data produces empty hex string
            ss58, commit_hex, round_num = result
            assert commit_hex == expected_commit_hex, (
                "Empty commit data should produce '0x' hex string"
            )
            assert round_num == round_number, "Round number should match input"

    def test_from_vec_u8_hex_formatting(self):
        """
        Test that from_vec_u8 correctly formats commit_data as hex string.

        This test verifies the hex formatting logic, ensuring that each
        byte is correctly converted to a two-digit hexadecimal representation
        and prefixed with '0x'.
        """
        # Arrange: Prepare test data with specific byte values to verify formatting
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        # Use bytes that will produce specific hex patterns
        commit_data = (0x00, 0x0F, 0xF0, 0xFF, 0x10, 0xAB)
        round_number = 1
        data = (account_id, commit_data, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        # Expected hex: each byte formatted as two hex digits
        expected_commit_hex = "0x000ff0ff10ab"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8
            result = WeightCommitInfo.from_vec_u8(data)

            # Assert: Verify hex formatting is correct
            ss58, commit_hex, round_num = result
            assert commit_hex == expected_commit_hex, (
                "Hex formatting should be correct with proper zero-padding"
            )
            assert commit_hex.startswith("0x"), (
                "Hex string should start with '0x' prefix"
            )
            # Verify each byte is represented as two hex digits
            hex_part = commit_hex[2:]  # Remove '0x' prefix
            assert len(hex_part) == len(commit_data) * 2, (
                "Each byte should be represented as two hex digits"
            )


class TestWeightCommitInfoFromVecU8V2:
    """
    Test suite for WeightCommitInfo.from_vec_u8_v2 class method.

    The from_vec_u8_v2 method is the newer version that includes commit_block
    information. It returns a tuple of (ss58, commit_block, commit_hex, round_number).
    This method is used when the CRV3WeightCommitsV2 storage function is available.
    """

    def test_from_vec_u8_v2_with_all_fields(self):
        """
        Test from_vec_u8_v2 with all fields populated.

        This is the primary use case for from_vec_u8_v2, where all data
        including commit_block is available from the blockchain.
        """
        # Arrange: Prepare test data with all fields
        account_id_tuple = (
            (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ),
        )
        commit_block_tuple = (12345,)
        commit_data_tuple = ((0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88),)
        round_number = 7
        data = (account_id_tuple, commit_block_tuple, commit_data_tuple, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_block = 12345
        expected_commit_hex = "0x1122334455667788"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify all four elements are returned correctly
            assert isinstance(result, tuple), "Result should be a tuple"
            assert len(result) == 4, (
                "Result should contain 4 elements (ss58, commit_block, commit_hex, round_number)"
            )

            ss58, commit_block, commit_hex, round_num = result
            assert ss58 == expected_ss58, "SS58 address should match decoded account_id"
            assert commit_block == expected_commit_block, (
                "Commit block should match input"
            )
            assert commit_hex == expected_commit_hex, (
                "Commit hex should be correctly formatted"
            )
            assert round_num == round_number, "Round number should match input"

    def test_from_vec_u8_v2_with_non_tuple_commit_block(self):
        """
        Test from_vec_u8_v2 with commit_block as direct value.

        Commit block may come as a direct integer value rather than wrapped
        in a tuple. This test ensures the method handles both cases.
        """
        # Arrange: Prepare test data with commit_block as direct value
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_block_direct = 54321
        commit_data = (0xAA, 0xBB, 0xCC, 0xDD)
        round_number = 12
        data = (account_id, commit_block_direct, commit_data, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0xaabbccdd"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2 with non-tuple commit_block
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify commit_block is correctly handled
            ss58, commit_block, commit_hex, round_num = result
            assert commit_block == commit_block_direct, (
                "Commit block should match input when not wrapped in tuple"
            )
            assert commit_hex == expected_commit_hex, (
                "Commit hex should be correctly formatted"
            )

    def test_from_vec_u8_v2_with_zero_commit_block(self):
        """
        Test from_vec_u8_v2 with commit_block equal to zero.

        Edge case: commit_block may be zero. This test verifies that zero
        is correctly handled and returned.
        """
        # Arrange: Prepare test data with zero commit_block
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_block_zero = 0
        commit_data = (0x01, 0x02, 0x03)
        round_number = 0
        data = (account_id, commit_block_zero, commit_data, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2 with zero commit_block
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify zero commit_block is correctly returned
            ss58, commit_block, commit_hex, round_num = result
            assert commit_block == 0, "Zero commit block should be correctly returned"
            assert round_num == 0, "Zero round number should be correctly returned"

    def test_from_vec_u8_v2_with_large_commit_block(self):
        """
        Test from_vec_u8_v2 with large commit_block value.

        This test verifies that large block numbers (which may occur in
        production) are correctly handled and returned.
        """
        # Arrange: Prepare test data with large commit_block
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_block_large = 999999999
        commit_data = (0xFF, 0xFF, 0xFF, 0xFF)
        round_number = 100
        data = (account_id, commit_block_large, commit_data, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2 with large commit_block
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify large commit_block is correctly returned
            ss58, commit_block, commit_hex, round_num = result
            assert commit_block == commit_block_large, (
                "Large commit block should be correctly returned"
            )
            assert isinstance(commit_block, int), "Commit block should be an integer"

    def test_from_vec_u8_v2_hex_formatting(self):
        """
        Test that from_vec_u8_v2 correctly formats commit_data as hex string.

        This test verifies the hex formatting logic is consistent with
        from_vec_u8, ensuring proper conversion of bytes to hexadecimal.
        """
        # Arrange: Prepare test data with specific byte values
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_block = 1000
        # Use bytes that require zero-padding in hex representation
        commit_data = (0x00, 0x01, 0x0A, 0x0F, 0xA0, 0xF0, 0xFF)
        round_number = 5
        data = (account_id, commit_block, commit_data, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0x00010a0fa0f0ff"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify hex formatting is correct
            ss58, commit_block, commit_hex, round_num = result
            assert commit_hex == expected_commit_hex, (
                "Hex formatting should match expected value"
            )
            assert commit_hex.startswith("0x"), (
                "Hex string should start with '0x' prefix"
            )
            # Verify each byte is two hex digits
            hex_part = commit_hex[2:]
            assert len(hex_part) == len(commit_data) * 2, (
                "Each byte should be two hex digits"
            )

    def test_from_vec_u8_v2_with_empty_commit_data(self):
        """
        Test from_vec_u8_v2 with empty commit_data.

        Edge case: commit_data may be empty. This test verifies that empty
        commit data produces a valid hex string (0x with no data).
        """
        # Arrange: Prepare test data with empty commit_data
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_block = 5000
        commit_data_empty = ()
        round_number = 3
        data = (account_id, commit_block, commit_data_empty, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        expected_commit_hex = "0x"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2 with empty commit_data
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify empty commit_data produces empty hex string
            ss58, commit_block, commit_hex, round_num = result
            assert commit_hex == expected_commit_hex, (
                "Empty commit data should produce '0x' hex string"
            )
            assert commit_block == 5000, (
                "Commit block should still be returned correctly"
            )

    def test_from_vec_u8_v2_tuple_unwrapping_consistency(self):
        """
        Test that from_vec_u8_v2 correctly unwraps all tuple-wrapped fields.

        This test verifies that the method correctly handles tuple-wrapped
        values for account_id, commit_block, and commit_data simultaneously.
        """
        # Arrange: Prepare test data with all fields wrapped in tuples
        account_id_tuple = (
            (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ),
        )
        commit_block_tuple = (98765,)
        commit_data_tuple = ((0x99, 0x88, 0x77, 0x66),)
        round_number = 25
        data = (account_id_tuple, commit_block_tuple, commit_data_tuple, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call from_vec_u8_v2 with all fields wrapped
            result = WeightCommitInfo.from_vec_u8_v2(data)

            # Assert: Verify all fields are correctly unwrapped
            ss58, commit_block, commit_hex, round_num = result
            assert ss58 == expected_ss58, (
                "SS58 should be decoded from unwrapped account_id"
            )
            assert commit_block == 98765, "Commit block should be unwrapped from tuple"
            assert commit_hex == "0x99887766", (
                "Commit hex should be from unwrapped commit_data"
            )
            assert round_num == 25, "Round number should match input"


class TestWeightCommitInfoEdgeCases:
    """
    Test suite for edge cases and error conditions in WeightCommitInfo.

    These tests verify that the class methods handle unusual inputs and
    edge cases gracefully, ensuring robustness in production environments.
    """

    def test_from_vec_u8_vs_from_vec_u8_v2_difference(self):
        """
        Test that from_vec_u8 and from_vec_u8_v2 return different structures.

        This test verifies the key difference between the two methods:
        from_vec_u8 returns 3 elements (no commit_block), while from_vec_u8_v2
        returns 4 elements (includes commit_block).
        """
        # Arrange: Prepare identical test data (except commit_block for v2)
        account_id = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        )
        commit_data = (0x11, 0x22, 0x33)
        round_number = 1

        data_v1 = (account_id, commit_data, round_number)
        data_v2 = (account_id, 100, commit_data, round_number)

        # Mock decode_account_id
        expected_ss58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

        with patch(
            "bittensor.core.chain_data.weight_commit_info.decode_account_id"
        ) as mock_decode:
            mock_decode.return_value = expected_ss58

            # Act: Call both methods
            result_v1 = WeightCommitInfo.from_vec_u8(data_v1)
            result_v2 = WeightCommitInfo.from_vec_u8_v2(data_v2)

            # Assert: Verify structural differences
            assert len(result_v1) == 3, "from_vec_u8 should return 3 elements"
            assert len(result_v2) == 4, "from_vec_u8_v2 should return 4 elements"

            # Verify commit_hex and round_number are the same
            assert result_v1[1] == result_v2[2], (
                "Commit hex should be the same in both methods"
            )
            assert result_v1[2] == result_v2[3], (
                "Round number should be the same in both methods"
            )

            # Verify v2 has commit_block but v1 doesn't
            assert result_v2[1] == 100, "from_vec_u8_v2 should include commit_block"
