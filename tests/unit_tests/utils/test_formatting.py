

from bittensor.utils.formatting import convert_blocks_to_time


def test_convert_blocks_to_time():
    assert convert_blocks_to_time(6301) == (21, 0, 12)
    assert convert_blocks_to_time(300) == (1, 0, 0)
    assert convert_blocks_to_time(10) == (0, 2, 0)
    assert convert_blocks_to_time(1) == (0, 0, 12)
    assert convert_blocks_to_time(186, block_time=3) == (0, 9, 18)
    assert convert_blocks_to_time(0) == (0, 0, 0)
