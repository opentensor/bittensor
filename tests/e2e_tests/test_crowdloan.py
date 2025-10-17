from bittensor import Balance
from bittensor.core.extrinsics.registration import RegistrationParams
from bittensor_wallet import Wallet
import pytest
import asyncio


def test_crowdloan_with_target(
    subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet, fred_wallet
):
    """Tests crowdloan creation with target.

    Steps:
    - Verify initial empty state
    - Validate crowdloan constants
    - Check InvalidCrowdloanId errors
    - Test creation validation errors
    - Create valid crowdloan with target
    - Verify creation and parameters
    - Update end block, cap, and min contribution
    - Test low contribution rejection
    - Add contributions from Alice and Charlie
    - Test withdrawal and re-contribution
    - Validate CapRaised behavior
    - Finalize crowdloan successfully
    - Confirm target (Fred) received funds
    - Validate post-finalization errors
    - Create second crowdloan for refund test
    - Contribute from Alice and Dave
    - Verify that refund imposable from non creator account
    - Refund all contributors
    - Verify balances after refund
    - Dissolve refunded crowdloan
    - Confirm only finalized crowdloan remains
    """
    # no one crowdloan has been created yet
    next_crowdloan = subtensor.crowdloans.get_crowdloan_next_id()
    assert next_crowdloan == 0

    # no crowdloans before creation
    assert subtensor.crowdloans.get_crowdloans() == []
    # no contributions before creation
    assert subtensor.crowdloans.get_crowdloan_contributions(next_crowdloan) == {}
    # no crowdloan with next ID before creation
    assert subtensor.crowdloans.get_crowdloan_by_id(next_crowdloan) is None

    # fetch crowdloan constants
    crowdloan_constants = subtensor.crowdloans.get_crowdloan_constants(next_crowdloan)
    assert crowdloan_constants.AbsoluteMinimumContribution == Balance.from_rao(
        100000000
    )
    assert crowdloan_constants.MaxContributors == 500
    assert crowdloan_constants.MinimumBlockDuration == 50
    assert crowdloan_constants.MaximumBlockDuration == 20000
    assert crowdloan_constants.MinimumDeposit == Balance.from_rao(10000000000)
    assert crowdloan_constants.RefundContributorsLimit == 50

    # All extrinsics expected to fail with InvalidCrowdloanId error
    invalid_calls = [
        lambda: subtensor.crowdloans.contribute_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(10)
        ),
        lambda: subtensor.crowdloans.withdraw_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan
        ),
        lambda: subtensor.crowdloans.update_min_contribution_crowdloan(
            wallet=bob_wallet,
            crowdloan_id=next_crowdloan,
            new_min_contribution=Balance.from_tao(10),
        ),
        lambda: subtensor.crowdloans.update_cap_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan, new_cap=Balance.from_tao(10)
        ),
        lambda: subtensor.crowdloans.update_end_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan, new_end=10000
        ),
        lambda: subtensor.crowdloans.dissolve_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan
        ),
        lambda: subtensor.crowdloans.finalize_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan
        ),
    ]

    for call in invalid_calls:
        response = call()
        assert response.success is False
        assert "InvalidCrowdloanId" in response.message
        assert response.error["name"] == "InvalidCrowdloanId"

    # create crowdloan to raise funds to send to wallet
    current_block = subtensor.block
    crowdloan_cap = Balance.from_tao(15)

    # check DepositTooLow error
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(5),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=current_block + 240,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "DepositTooLow" in response.message
    assert response.error["name"] == "DepositTooLow"

    # check CapTooLow error
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=Balance.from_tao(10),
        end=current_block + 240,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "CapTooLow" in response.message
    assert response.error["name"] == "CapTooLow"

    # check CannotEndInPast error
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=current_block,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "CannotEndInPast" in response.message
    assert response.error["name"] == "CannotEndInPast"

    # check BlockDurationTooShort error
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=subtensor.block + 10,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "BlockDurationTooShort" in response.message
    assert response.error["name"] == "BlockDurationTooShort"

    # check BlockDurationTooLong error
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=subtensor.block + crowdloan_constants.MaximumBlockDuration + 100,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "BlockDurationTooLong" in response.message
    assert response.error["name"] == "BlockDurationTooLong"

    # === SUCCESSFUL creation ===
    fred_balance = subtensor.wallets.get_balance(fred_wallet.hotkey.ss58_address)
    assert fred_balance == Balance.from_tao(0)

    end_block = subtensor.block + 240
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=end_block,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert response.success, response.message

    # check crowdloan created successfully
    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.contributors_count == 1
    assert crowdloan.min_contribution == Balance.from_tao(1)
    assert crowdloan.end == end_block

    # check update end block
    new_end_block = end_block + 100
    response = subtensor.crowdloans.update_end_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan, new_end=new_end_block
    )
    assert response.success, response.message

    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.end == new_end_block

    # check update crowdloan cap
    updated_crowdloan_cap = Balance.from_tao(20)
    response = subtensor.crowdloans.update_cap_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan, new_cap=updated_crowdloan_cap
    )
    assert response.success, response.message

    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.cap == updated_crowdloan_cap

    # check min contribution update
    response = subtensor.crowdloans.update_min_contribution_crowdloan(
        wallet=bob_wallet,
        crowdloan_id=next_crowdloan,
        new_min_contribution=Balance.from_tao(5),
    )
    assert response.success, response.message

    # check contribution not enough
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(1)
    )
    assert "ContributionTooLow" in response.message
    assert response.error["name"] == "ContributionTooLow"

    # check successful contribution crowdloan
    # contribution from alice
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # contribution from charlie
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # check charlie_wallet withdraw amount back
    charlie_balance_before = subtensor.wallets.get_balance(
        charlie_wallet.hotkey.ss58_address
    )
    response = subtensor.crowdloans.withdraw_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message
    charlie_balance_after = subtensor.wallets.get_balance(
        charlie_wallet.hotkey.ss58_address
    )
    assert (
        charlie_balance_after
        == charlie_balance_before + Balance.from_tao(5) - response.extrinsic_fee
    )

    # contribution from charlie again
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # check over contribution with CapRaised error
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(1)
    )
    assert "CapRaised" in response.message
    assert response.error["name"] == "CapRaised"

    crowdloan_contributions = subtensor.crowdloans.get_crowdloan_contributions(
        next_crowdloan
    )
    assert len(crowdloan_contributions) == 3
    assert crowdloan_contributions[bob_wallet.hotkey.ss58_address] == Balance.from_tao(
        10
    )
    assert crowdloan_contributions[
        alice_wallet.hotkey.ss58_address
    ] == Balance.from_tao(5)
    assert crowdloan_contributions[
        charlie_wallet.hotkey.ss58_address
    ] == Balance.from_tao(5)

    # check finalization
    response = subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # make sure fred received raised amount
    fred_balance_after_finalize = subtensor.wallets.get_balance(
        fred_wallet.hotkey.ss58_address
    )
    assert fred_balance_after_finalize == updated_crowdloan_cap

    # check AlreadyFinalized error after finalization
    response = subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert "AlreadyFinalized" in response.message
    assert response.error["name"] == "AlreadyFinalized"

    # check error after finalization
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert "CapRaised" in response.message
    assert response.error["name"] == "CapRaised"

    # check dissolve crowdloan error after finalization
    response = subtensor.crowdloans.dissolve_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert "AlreadyFinalized" in response.message
    assert response.error["name"] == "AlreadyFinalized"

    crowdloans = subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 1

    # === check refund crowdloan (create + contribute + refund + dissolve) ===
    next_crowdloan = subtensor.crowdloans.get_crowdloan_next_id()
    assert next_crowdloan == 1

    bob_deposit = Balance.from_tao(10)
    crowdloan_cap = Balance.from_tao(20)

    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=bob_deposit,
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=subtensor.block + 240,
        target_address=dave_wallet.hotkey.ss58_address,
    )
    assert response.success, response.message

    crowdloans = subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 2

    # check crowdloan's raised amount decreased after refund
    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert crowdloan.raised == bob_deposit

    alice_balance_before = subtensor.wallets.get_balance(
        alice_wallet.hotkey.ss58_address
    )
    alice_contribute_amount = Balance.from_tao(5)
    dave_balance_before = subtensor.wallets.get_balance(dave_wallet.hotkey.ss58_address)
    dave_contribution_amount = Balance.from_tao(5)

    # contribution from alice
    response_alice_contrib = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=alice_contribute_amount
    )
    assert response_alice_contrib.success, response_alice_contrib.message

    # check alice balance decreased
    alice_balance_after_contrib = subtensor.wallets.get_balance(
        alice_wallet.hotkey.ss58_address
    )
    assert (
        alice_balance_after_contrib
        == alice_balance_before
        - alice_contribute_amount
        - response_alice_contrib.extrinsic_fee
    )

    # contribution from dave
    response_dave_contrib = subtensor.crowdloans.contribute_crowdloan(
        wallet=dave_wallet, crowdloan_id=next_crowdloan, amount=dave_contribution_amount
    )
    assert response_dave_contrib.success, response_dave_contrib.message

    # check dave balance decreased
    dave_balance_after_contrib = subtensor.wallets.get_balance(
        dave_wallet.hotkey.ss58_address
    )
    assert (
        dave_balance_after_contrib
        == dave_balance_before
        - dave_contribution_amount
        - response_dave_contrib.extrinsic_fee
    )

    # check crowdloan's raised amount
    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert (
        crowdloan.raised
        == bob_deposit + alice_contribute_amount + dave_contribution_amount
    )

    # refund crowdloan from wrong account
    response = subtensor.crowdloans.refund_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan,
    )
    assert "InvalidOrigin" in response.message
    assert response.error["name"] == "InvalidOrigin"

    # refund crowdloan from creator account
    response = subtensor.crowdloans.refund_crowdloan(
        wallet=bob_wallet,
        crowdloan_id=next_crowdloan,
    )
    assert response.success, response.message

    # check crowdloan's raised amount decreased after refund
    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert crowdloan.raised == bob_deposit

    # check alice balance increased after refund
    alice_balance_after_refund = subtensor.wallets.get_balance(
        alice_wallet.hotkey.ss58_address
    )
    assert (
        alice_balance_after_refund
        == alice_balance_after_contrib + alice_contribute_amount
    )

    # check dave balance increased after refund
    dave_balance_after_refund = subtensor.wallets.get_balance(
        dave_wallet.hotkey.ss58_address
    )
    assert (
        dave_balance_after_refund
        == dave_balance_after_contrib + dave_contribution_amount
    )

    # dissolve crowdloan
    response = subtensor.crowdloans.dissolve_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # check that chain has just one finalized crowdloan
    crowdloans = subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 1


@pytest.mark.asyncio
async def test_crowdloan_with_target_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet, fred_wallet
):
    """Async tests crowdloan creation with target.

    Steps:
    - Verify initial empty state
    - Validate crowdloan constants
    - Check InvalidCrowdloanId errors
    - Test creation validation errors
    - Create valid crowdloan with target
    - Verify creation and parameters
    - Update end block, cap, and min contribution
    - Test low contribution rejection
    - Add contributions from Alice and Charlie
    - Test withdrawal and re-contribution
    - Validate CapRaised behavior
    - Finalize crowdloan successfully
    - Confirm target (Fred) received funds
    - Validate post-finalization errors
    - Create second crowdloan for refund test
    - Contribute from Alice and Dave
    - Verify that refund imposable from non creator account
    - Refund all contributors
    - Verify balances after refund
    - Dissolve refunded crowdloan
    - Confirm only finalized crowdloan remains
    """
    # no one crowdloan has been created yet
    (
        next_crowdloan,
        crowdloans,
        crowdloan_contributions,
        crowdloan_by_id,
    ) = await asyncio.gather(
        async_subtensor.crowdloans.get_crowdloan_next_id(),
        async_subtensor.crowdloans.get_crowdloans(),
        async_subtensor.crowdloans.get_crowdloan_contributions(0),
        async_subtensor.crowdloans.get_crowdloan_by_id(0),
    )
    # no created crowdloans yet
    assert next_crowdloan == 0
    # no crowdloans before creation
    assert len(crowdloans) == 0
    # no contributions before creation
    assert crowdloan_contributions == {}
    # no crowdloan with next ID before creation
    assert crowdloan_by_id is None

    # fetch crowdloan constants
    crowdloan_constants = await async_subtensor.crowdloans.get_crowdloan_constants(
        next_crowdloan
    )
    assert crowdloan_constants.AbsoluteMinimumContribution == Balance.from_rao(
        100000000
    )
    assert crowdloan_constants.MaxContributors == 500
    assert crowdloan_constants.MinimumBlockDuration == 50
    assert crowdloan_constants.MaximumBlockDuration == 20000
    assert crowdloan_constants.MinimumDeposit == Balance.from_rao(10000000000)
    assert crowdloan_constants.RefundContributorsLimit == 50

    # All extrinsics expected to fail with InvalidCrowdloanId error
    invalid_calls = [
        lambda: async_subtensor.crowdloans.contribute_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(10)
        ),
        lambda: async_subtensor.crowdloans.withdraw_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan
        ),
        lambda: async_subtensor.crowdloans.update_min_contribution_crowdloan(
            wallet=bob_wallet,
            crowdloan_id=next_crowdloan,
            new_min_contribution=Balance.from_tao(10),
        ),
        lambda: async_subtensor.crowdloans.update_cap_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan, new_cap=Balance.from_tao(10)
        ),
        lambda: async_subtensor.crowdloans.update_end_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan, new_end=10000
        ),
        lambda: async_subtensor.crowdloans.dissolve_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan
        ),
        lambda: async_subtensor.crowdloans.finalize_crowdloan(
            wallet=bob_wallet, crowdloan_id=next_crowdloan
        ),
    ]

    for call in invalid_calls:
        response = await call()
        assert response.success is False
        assert "InvalidCrowdloanId" in response.message
        assert response.error["name"] == "InvalidCrowdloanId"

    # create crowdloan to raise funds to send to wallet
    current_block = await async_subtensor.block
    crowdloan_cap = Balance.from_tao(15)

    # check DepositTooLow error
    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(5),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=current_block + 240,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "DepositTooLow" in response.message
    assert response.error["name"] == "DepositTooLow"

    # check CapTooLow error
    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=Balance.from_tao(10),
        end=current_block + 240,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "CapTooLow" in response.message
    assert response.error["name"] == "CapTooLow"

    # check CannotEndInPast error
    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=current_block,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "CannotEndInPast" in response.message
    assert response.error["name"] == "CannotEndInPast"

    # check BlockDurationTooShort error
    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=await async_subtensor.block + 49,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "BlockDurationTooShort" in response.message
    assert response.error["name"] == "BlockDurationTooShort"

    # check BlockDurationTooLong error
    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=await async_subtensor.block
        + crowdloan_constants.MaximumBlockDuration
        + 100,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert "BlockDurationTooLong" in response.message
    assert response.error["name"] == "BlockDurationTooLong"

    # === SUCCESSFUL creation ===
    fred_balance = await async_subtensor.wallets.get_balance(
        fred_wallet.hotkey.ss58_address
    )
    assert fred_balance == Balance.from_tao(0)

    end_block = await async_subtensor.block + 240
    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=end_block,
        target_address=fred_wallet.hotkey.ss58_address,
    )
    assert response.success, response.message

    # check crowdloan created successfully
    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.contributors_count == 1
    assert crowdloan.min_contribution == Balance.from_tao(1)
    assert crowdloan.end == end_block

    # check update end block
    new_end_block = end_block + 100
    response = await async_subtensor.crowdloans.update_end_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan, new_end=new_end_block
    )
    assert response.success, response.message

    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.end == new_end_block

    # check update crowdloan cap
    updated_crowdloan_cap = Balance.from_tao(20)
    response = await async_subtensor.crowdloans.update_cap_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan, new_cap=updated_crowdloan_cap
    )
    assert response.success, response.message

    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.cap == updated_crowdloan_cap

    # check min contribution update
    response = await async_subtensor.crowdloans.update_min_contribution_crowdloan(
        wallet=bob_wallet,
        crowdloan_id=next_crowdloan,
        new_min_contribution=Balance.from_tao(5),
    )
    assert response.success, response.message

    # check contribution not enough
    response = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(1)
    )
    assert "ContributionTooLow" in response.message
    assert response.error["name"] == "ContributionTooLow"

    # check successful contribution crowdloan
    # contribution from alice
    response = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # contribution from charlie
    response = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # check charlie_wallet withdraw amount back
    charlie_balance_before = await async_subtensor.wallets.get_balance(
        charlie_wallet.hotkey.ss58_address
    )
    response = await async_subtensor.crowdloans.withdraw_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message
    charlie_balance_after = await async_subtensor.wallets.get_balance(
        charlie_wallet.hotkey.ss58_address
    )
    assert (
        charlie_balance_after
        == charlie_balance_before + Balance.from_tao(5) - response.extrinsic_fee
    )

    # contribution from charlie again
    response = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # check over contribution with CapRaised error
    response = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(1)
    )
    assert "CapRaised" in response.message
    assert response.error["name"] == "CapRaised"

    crowdloan_contributions = (
        await async_subtensor.crowdloans.get_crowdloan_contributions(next_crowdloan)
    )
    assert len(crowdloan_contributions) == 3
    assert crowdloan_contributions[bob_wallet.hotkey.ss58_address] == Balance.from_tao(
        10
    )
    assert crowdloan_contributions[
        alice_wallet.hotkey.ss58_address
    ] == Balance.from_tao(5)
    assert crowdloan_contributions[
        charlie_wallet.hotkey.ss58_address
    ] == Balance.from_tao(5)

    # check finalization
    response = await async_subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # make sure fred received raised amount
    fred_balance_after_finalize = await async_subtensor.wallets.get_balance(
        fred_wallet.hotkey.ss58_address
    )
    assert fred_balance_after_finalize == updated_crowdloan_cap

    # check AlreadyFinalized error after finalization
    response = await async_subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert "AlreadyFinalized" in response.message
    assert response.error["name"] == "AlreadyFinalized"

    # check error after finalization
    response = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet, crowdloan_id=next_crowdloan, amount=Balance.from_tao(5)
    )
    assert "CapRaised" in response.message
    assert response.error["name"] == "CapRaised"

    # check dissolve crowdloan error after finalization
    response = await async_subtensor.crowdloans.dissolve_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert "AlreadyFinalized" in response.message
    assert response.error["name"] == "AlreadyFinalized"

    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 1

    # === check refund crowdloan (create + contribute + refund + dissolve) ===
    next_crowdloan = await async_subtensor.crowdloans.get_crowdloan_next_id()
    assert next_crowdloan == 1

    bob_deposit = Balance.from_tao(10)
    crowdloan_cap = Balance.from_tao(20)

    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=bob_deposit,
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=await async_subtensor.block + 240,
        target_address=dave_wallet.hotkey.ss58_address,
    )
    assert response.success, response.message

    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 2

    # check crowdloan's raised amount decreased after refund
    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert crowdloan.raised == bob_deposit

    alice_balance_before = await async_subtensor.wallets.get_balance(
        alice_wallet.hotkey.ss58_address
    )
    alice_contribute_amount = Balance.from_tao(5)
    dave_balance_before = await async_subtensor.wallets.get_balance(
        dave_wallet.hotkey.ss58_address
    )
    dave_contribution_amount = Balance.from_tao(5)

    # contribution from alice
    response_alice_contrib = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=alice_contribute_amount
    )
    assert response_alice_contrib.success, response_alice_contrib.message

    # check alice balance decreased
    alice_balance_after_contrib = await async_subtensor.wallets.get_balance(
        alice_wallet.hotkey.ss58_address
    )
    assert (
        alice_balance_after_contrib
        == alice_balance_before
        - alice_contribute_amount
        - response_alice_contrib.extrinsic_fee
    )

    # contribution from dave
    response_dave_contrib = await async_subtensor.crowdloans.contribute_crowdloan(
        wallet=dave_wallet, crowdloan_id=next_crowdloan, amount=dave_contribution_amount
    )
    assert response_dave_contrib.success, response_dave_contrib.message

    # check dave balance decreased
    dave_balance_after_contrib = await async_subtensor.wallets.get_balance(
        dave_wallet.hotkey.ss58_address
    )
    assert (
        dave_balance_after_contrib
        == dave_balance_before
        - dave_contribution_amount
        - response_dave_contrib.extrinsic_fee
    )

    # check crowdloan's raised amount
    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert (
        crowdloan.raised
        == bob_deposit + alice_contribute_amount + dave_contribution_amount
    )

    # refund crowdloan from wrong account
    response = await subtensor.crowdloans.refund_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan,
    )
    assert "InvalidOrigin" in response.message
    assert response.error["name"] == "InvalidOrigin"

    # refund crowdloan from creator account
    response = await async_subtensor.crowdloans.refund_crowdloan(
        wallet=bob_wallet,
        crowdloan_id=next_crowdloan,
    )
    assert response.success, response.message

    # check crowdloan's raised amount decreased after refund
    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert crowdloan.raised == bob_deposit

    # check alice balance increased after refund
    alice_balance_after_refund = await async_subtensor.wallets.get_balance(
        alice_wallet.hotkey.ss58_address
    )
    assert (
        alice_balance_after_refund
        == alice_balance_after_contrib + alice_contribute_amount
    )

    # check dave balance increased after refund
    dave_balance_after_refund = await async_subtensor.wallets.get_balance(
        dave_wallet.hotkey.ss58_address
    )
    assert (
        dave_balance_after_refund
        == dave_balance_after_contrib + dave_contribution_amount
    )

    # dissolve crowdloan
    response = await async_subtensor.crowdloans.dissolve_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # check that chain has just one finalized crowdloan
    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 1


def test_crowdloan_with_call(
    subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet, fred_wallet
):
    """Tests crowdloan creation with call.

    Steps:
        - Compose subnet registration call
        - Create new crowdloan
        - Verify creation and balance change
        - Alice contributes to crowdloan
        - Charlie contributes to crowdloan
        - Verify total raised and contributors
        - Finalize crowdloan campaign
        - Verify new subnet created   (composed crowdloan call executed)
        - Confirm subnet owner is Fred
    """
    # create crowdloan's call
    crowdloan_call = subtensor.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params=RegistrationParams.register_network(
            hotkey_ss58=fred_wallet.hotkey.ss58_address
        ),
    )

    next_crowdloan = subtensor.crowdloans.get_crowdloan_next_id()
    subnets_before = subtensor.subnets.get_all_subnets_netuid()
    crowdloan_cap = Balance.from_tao(30)
    crowdloan_deposit = Balance.from_tao(10)

    bob_balance_before = subtensor.wallets.get_balance(bob_wallet.hotkey.ss58_address)

    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=crowdloan_deposit,
        min_contribution=Balance.from_tao(5),
        cap=crowdloan_cap,
        end=subtensor.block + 2400,
        call=crowdloan_call,
    )

    # keep it until `scalecodec` has a fix for `wait_for_inclusion=True` and `wait_for_finalization=True`
    subtensor.wait_for_block(subtensor.block + 10)

    # check creation was successful
    assert response.success, response.message

    # check bob balance decreased
    bob_balance_after = subtensor.wallets.get_balance(bob_wallet.hotkey.ss58_address)
    assert (
        bob_balance_after
        == bob_balance_before - crowdloan_deposit - response.extrinsic_fee
    )

    # contribution from alice
    alice_contribute_amount = Balance.from_tao(10)
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet, crowdloan_id=next_crowdloan, amount=alice_contribute_amount
    )
    assert response.success, response.message

    # contribution from charlie
    charlie_contribute_amount = Balance.from_tao(10)
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan,
        amount=charlie_contribute_amount,
    )
    assert response.success, response.message

    # make sure the crowdloan company is ready to finalize
    crowdloans = subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.contributors_count == 3
    assert (
        crowdloan.raised
        == crowdloan_deposit + alice_contribute_amount + charlie_contribute_amount
    )
    assert crowdloan.cap == crowdloan_cap

    # finalize crowdloan
    response = subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # check new subnet exist
    subnets_after = subtensor.subnets.get_all_subnets_netuid()
    assert len(subnets_after) == len(subnets_before) + 1

    # get new subnet id and owner
    new_subnet_id = subnets_after[-1]
    new_subnet_owner_hk = subtensor.subnets.get_subnet_owner_hotkey(new_subnet_id)

    # make sure subnet owner is fred
    assert new_subnet_owner_hk == fred_wallet.hotkey.ss58_address


@pytest.mark.asyncio
async def test_crowdloan_with_call_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet, fred_wallet
):
    """Async tests crowdloan creation with call.

    Steps:
        - Compose subnet registration call
        - Create new crowdloan
        - Verify creation and balance change
        - Alice contributes to crowdloan
        - Charlie contributes to crowdloan
        - Verify total raised and contributors
        - Finalize crowdloan campaign
        - Verify new subnet created   (composed crowdloan call executed)
        - Confirm subnet owner is Fred
    """
    # create crowdloan's call
    crowdloan_call = await async_subtensor.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params=RegistrationParams.register_network(
            hotkey_ss58=fred_wallet.hotkey.ss58_address
        ),
    )

    crowdloan_cap = Balance.from_tao(30)
    crowdloan_deposit = Balance.from_tao(10)

    (
        next_crowdloan,
        subnets_before,
        bob_balance_before,
        current_block,
    ) = await asyncio.gather(
        async_subtensor.crowdloans.get_crowdloan_next_id(),
        async_subtensor.subnets.get_all_subnets_netuid(),
        async_subtensor.wallets.get_balance(bob_wallet.hotkey.ss58_address),
        async_subtensor.block,
    )
    end_block = current_block + 2400

    response = await async_subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=crowdloan_deposit,
        min_contribution=Balance.from_tao(5),
        cap=crowdloan_cap,
        end=end_block,
        call=crowdloan_call,
    )

    # keep it until `scalecodec` has a fix for `wait_for_inclusion=True` and `wait_for_finalization=True`
    await async_subtensor.wait_for_block(current_block + 20)

    # check creation was successful
    assert response.success, response.message

    # check bob balance decreased
    bob_balance_after = await async_subtensor.wallets.get_balance(
        bob_wallet.hotkey.ss58_address
    )
    assert (
        bob_balance_after
        == bob_balance_before - crowdloan_deposit - response.extrinsic_fee
    )

    # contribution from alice and charlie
    alice_contribute_amount = Balance.from_tao(10)
    charlie_contribute_amount = Balance.from_tao(10)

    a_response, c_response = await asyncio.gather(
        async_subtensor.crowdloans.contribute_crowdloan(
            wallet=alice_wallet,
            crowdloan_id=next_crowdloan,
            amount=alice_contribute_amount,
        ),
        async_subtensor.crowdloans.contribute_crowdloan(
            wallet=charlie_wallet,
            crowdloan_id=next_crowdloan,
            amount=charlie_contribute_amount,
        ),
    )
    assert a_response.success, a_response.message
    assert c_response.success, c_response.message

    # make sure the crowdloan company is ready to finalize
    crowdloans = await async_subtensor.crowdloans.get_crowdloans()
    crowdloan = [c for c in crowdloans if c.id == next_crowdloan][0]
    assert len(crowdloans) == 1
    assert crowdloan.id == next_crowdloan
    assert crowdloan.contributors_count == 3
    assert (
        crowdloan.raised
        == crowdloan_deposit + alice_contribute_amount + charlie_contribute_amount
    )
    assert crowdloan.cap == crowdloan_cap

    # finalize crowdloan
    response = await async_subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # check new subnet exist
    subnets_after = await async_subtensor.subnets.get_all_subnets_netuid()
    assert len(subnets_after) == len(subnets_before) + 1

    # get new subnet id and owner
    new_subnet_id = subnets_after[-1]
    new_subnet_owner_hk = await async_subtensor.subnets.get_subnet_owner_hotkey(
        new_subnet_id
    )

    # make sure subnet owner is fred
    assert new_subnet_owner_hk == fred_wallet.hotkey.ss58_address
