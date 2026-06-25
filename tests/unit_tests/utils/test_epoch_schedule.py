import pytest

from bittensor.utils.epoch_schedule import (
    blocks_until_next_auto_epoch,
    is_in_admin_freeze_window,
)


def test_blocks_until_next_auto_epoch_before_next_epoch():
    """Verify blocks remaining before the next auto epoch."""
    assert blocks_until_next_auto_epoch(100, 50, 120) == 30


def test_blocks_until_next_auto_epoch_at_next_epoch():
    """Verify zero remaining at the next auto epoch boundary."""
    assert blocks_until_next_auto_epoch(100, 50, 150) == 0


def test_blocks_until_next_auto_epoch_after_next_epoch():
    """Verify zero remaining after the next auto epoch boundary."""
    assert blocks_until_next_auto_epoch(100, 50, 200) == 0


def test_is_in_admin_freeze_window_tempo_zero():
    """Verify tempo zero disables the admin freeze window."""
    assert (
        is_in_admin_freeze_window(
            tempo=0,
            pending_epoch_at=0,
            last_epoch_block=100,
            block_number=100,
            admin_freeze_window=10,
        )
        is False
    )


def test_is_in_admin_freeze_window_pending_epoch_in_future():
    """Verify pending triggered epoch in the future blocks owner operations."""
    assert (
        is_in_admin_freeze_window(
            tempo=20,
            pending_epoch_at=200,
            last_epoch_block=80,
            block_number=91,
            admin_freeze_window=10,
        )
        is True
    )


def test_is_in_admin_freeze_window_auto_epoch_inside_window():
    """Verify auto epoch inside the freeze window blocks owner operations."""
    assert (
        is_in_admin_freeze_window(
            tempo=20,
            pending_epoch_at=0,
            last_epoch_block=80,
            block_number=91,
            admin_freeze_window=10,
        )
        is True
    )


def test_is_in_admin_freeze_window_auto_epoch_at_boundary():
    """Verify remaining == window is not frozen (strict less-than)."""
    assert (
        is_in_admin_freeze_window(
            tempo=20,
            pending_epoch_at=0,
            last_epoch_block=80,
            block_number=90,
            admin_freeze_window=10,
        )
        is False
    )


@pytest.mark.parametrize(
    "block_number, expected",
    [
        (89, False),
        (90, False),
        (91, True),
        (99, True),
        (100, True),
    ],
)
def test_is_in_admin_freeze_window_edge_cases(block_number, expected):
    """Sweep around the freeze boundary for last=80, tempo=20, window=10."""
    assert (
        is_in_admin_freeze_window(
            tempo=20,
            pending_epoch_at=0,
            last_epoch_block=80,
            block_number=block_number,
            admin_freeze_window=10,
        )
        is expected
    )
