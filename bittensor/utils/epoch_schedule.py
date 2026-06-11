"""Pure-function ports of subtensor's epoch scheduling logic."""


def blocks_until_next_auto_epoch(
    last_epoch_block: int, tempo: int, block_number: int
) -> int:
    """Returns the number of blocks remaining before the next automatic epoch.

    Port of ``run_coinbase.rs::blocks_until_next_auto_epoch``. Does not account for ``PendingEpochAt``, the
    ``BlocksSinceLastStep > MAX_TEMPO`` safety-net, or per-block-cap deferral. Caller must guard against ``tempo == 0``
    upstream.

    Parameters:
        last_epoch_block: The block at which the last epoch fired for this subnet.
        tempo: The subnet's tempo (epoch period in blocks).
        block_number: The current (or reference) block number.

    Returns:
        blocks_remaining: Non-negative number of blocks until ``last_epoch_block + tempo``.
    """
    next_auto = last_epoch_block + tempo
    return max(0, next_auto - block_number)


def is_in_admin_freeze_window(
    *,
    tempo: int,
    pending_epoch_at: int,
    last_epoch_block: int,
    block_number: int,
    admin_freeze_window: int,
) -> bool:
    """Returns whether owner operations are blocked because an epoch is imminent.

    Returns ``True`` when the current block is within the terminal ``admin_freeze_window`` blocks before the next auto
    epoch, or a pending manual trigger is armed (``pending_epoch_at > block_number``).

    Parameters:
        tempo: The subnet's tempo (epoch period in blocks). Returns ``False`` when zero.
        pending_epoch_at: Block at which an owner-triggered epoch is scheduled (``0`` = none).
        last_epoch_block: The block at which the last epoch fired for this subnet.
        block_number: The current (or reference) block number.
        admin_freeze_window: How many blocks before an epoch owner operations are frozen.

    Returns:
        is_frozen: ``True`` if owner operations should be blocked.
    """
    if tempo == 0:
        return False
    if pending_epoch_at > 0 and pending_epoch_at > block_number:
        return True
    remaining = blocks_until_next_auto_epoch(last_epoch_block, tempo, block_number)
    return remaining < admin_freeze_window
