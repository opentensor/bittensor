from bittensor import Balance
from bittensor.core.extrinsics.registration import RegistrationParams
from bittensor_wallet import Wallet


def test_crowdloan_with_target(subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet):

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
    assert crowdloan_constants.AbsoluteMinimumContribution == Balance.from_rao(100000000)
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
            wallet=bob_wallet, crowdloan_id=next_crowdloan, new_min_contribution=Balance.from_tao(10)
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
    crowdloan_cap = Balance.from_tao(20)

    # check DepositTooLow error
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(5),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=current_block + 240,
        target_address=dave_wallet.hotkey.ss58_address,
        wait_for_inclusion=True,
        wait_for_finalization=True,
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
        target_address=dave_wallet.hotkey.ss58_address,
        wait_for_inclusion=True,
        wait_for_finalization=True,
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
        target_address=dave_wallet.hotkey.ss58_address,
        wait_for_inclusion=True,
        wait_for_finalization=True,
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
        target_address=dave_wallet.hotkey.ss58_address,
        wait_for_inclusion=True,
        wait_for_finalization=True,
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
        target_address=dave_wallet.hotkey.ss58_address,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert "BlockDurationTooLong" in response.message
    assert response.error["name"] == "BlockDurationTooLong"

    # successful creation
    dave_balance_before = subtensor.wallets.get_balance(dave_wallet.hotkey.ss58_address)
    response = subtensor.crowdloans.create_crowdloan(
        wallet=bob_wallet,
        deposit=Balance.from_tao(10),
        min_contribution=Balance.from_tao(1),
        cap=crowdloan_cap,
        end=subtensor.block + 240,
        target_address=dave_wallet.hotkey.ss58_address,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success, response.message

    crowdloans = subtensor.crowdloans.get_crowdloans()
    assert len(crowdloans) == 1
    assert crowdloans[0].id == next_crowdloan

    # check contribute crowdloan
    # from alice
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet,
        crowdloan_id=next_crowdloan,
        amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # from charlie
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan,
        amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # check charlie_wallet withdraw amount back
    charlie_balance_before = subtensor.wallets.get_balance(charlie_wallet.hotkey.ss58_address)
    response = subtensor.crowdloans.withdraw_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan
    )
    assert response.success, response.message
    charlie_balance_after = subtensor.wallets.get_balance(charlie_wallet.hotkey.ss58_address)
    assert charlie_balance_after == charlie_balance_before + Balance.from_tao(5) - response.extrinsic_fee

    # from charlie again
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan,
        amount=Balance.from_tao(5)
    )
    assert response.success, response.message

    # check over contribution with CapRaised error
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=alice_wallet,
        crowdloan_id=next_crowdloan,
        amount=Balance.from_tao(1)
    )
    assert "CapRaised" in response.message
    assert response.error["name"] == "CapRaised"

    crowdloan_contributions = subtensor.crowdloans.get_crowdloan_contributions(next_crowdloan)
    assert len(crowdloan_contributions) == 3
    assert crowdloan_contributions[bob_wallet.hotkey.ss58_address] == Balance.from_tao(10)
    assert crowdloan_contributions[alice_wallet.hotkey.ss58_address] == Balance.from_tao(5)
    assert crowdloan_contributions[charlie_wallet.hotkey.ss58_address] == Balance.from_tao(5)

    # check finalization
    response = subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert response.success, response.message

    # check AlreadyFinalized error after finalization
    response = subtensor.crowdloans.finalize_crowdloan(
        wallet=bob_wallet, crowdloan_id=next_crowdloan
    )
    assert "AlreadyFinalized" in response.message
    assert response.error["name"] == "AlreadyFinalized"

    dave_balance_after = subtensor.wallets.get_balance(dave_wallet.hotkey.ss58_address)
    assert dave_balance_after == dave_balance_before + crowdloan_cap

    # check error after finalization
    response = subtensor.crowdloans.contribute_crowdloan(
        wallet=charlie_wallet,
        crowdloan_id=next_crowdloan,
        amount=Balance.from_tao(5)
    )
    assert "CapRaised" in response.message
    assert response.error["name"] == "CapRaised"

    # need add cases
    # 1. creation using call
    # 2. update_min_contribution_crowdloan_extrinsic
    # 3. update_end_crowdloan_extrinsic
    # 4. update_cap_crowdloan_extrinsic
    # 5. refund_crowdloan_extrinsic
    # 6. dissolve_crowdloan_extrinsic
    # 7. add docstring with steps
    # 8. duplicate test for async impl


# def test_crowdloan_with_call(subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet, fred_wallet):
#
#     assert subtensor.wallets.get_balance(fred_wallet.hotkey.ss58_address) == Balance.from_tao(0)
#
#     crowdloan_call = subtensor.compose_call(
#         call_module="SubtensorModule",
#         call_function="register_network",
#         call_params=RegistrationParams.register_network(
#             hotkey_ss58=fred_wallet.hotkey.ss58_address
#         ),
#     )
#
#     next_crowdloan = subtensor.crowdloans.get_crowdloan_next_id()
#
#     response = subtensor.crowdloans.create_crowdloan(
#         wallet=bob_wallet,
#         deposit=Balance.from_tao(10),
#         min_contribution=Balance.from_tao(5),
#         cap=Balance.from_tao(30),
#         end=subtensor.block + 2400,
#         call=crowdloan_call,
#         raise_error=True,
#         wait_for_inclusion=False,
#         wait_for_finalization=False,
#     )
#
#     assert response.success, response.message
#
#     crowdloans = subtensor.crowdloans.get_crowdloans()
#     assert len(crowdloans) == 1
#     assert crowdloans[0].id == next_crowdloan
#

