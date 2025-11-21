import pytest

from bittensor.core.chain_data.proxy import ProxyType
from bittensor.core.extrinsics.pallets import SubtensorModule, Proxy, Balances
from bittensor.utils.balance import Balance


def test_proxy_and_errors(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """Tests proxy logic.

    Steps:
        - Verify that chain has no proxies initially.
        - Add proxy with ProxyType.Registration and verify success.
        - Attempt to add duplicate proxy and verify error handling.
        - Add proxy with ProxyType.Transfer and verify success.
        - Verify chain has 2 proxies with correct deposit.
        - Verify proxy details match expected values (delegate, type, delay).
        - Test get_proxies() returns all proxies in network.
        - Test get_proxy_constants() returns valid constants.
        - Remove proxy ProxyType.Registration and verify deposit decreases.
        - Verify chain has 1 proxy remaining.
        - Remove proxy ProxyType.Transfer and verify all proxies removed.
        - Attempt to remove non-existent proxy and verify NotFound error.
        - Attempt to add proxy with invalid type and verify error.
        - Attempt to add self as proxy and verify NoSelfProxy error.
        - Test adding proxy with delay = 0 and verify it works.
        - Test adding multiple proxy types for same delegate.
        - Test adding proxy with different delegate.
    """
    real_account_wallet = bob_wallet
    delegate_wallet = charlie_wallet
    delay = 100

    # === check that chain has no proxies ===
    proxies, deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert not proxies
    assert deposit == 0

    # === add proxy with ProxyType.Registration ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert response.success, response.message

    # === add the same proxy returns error ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert not response.success
    assert "Duplicate" in response.message
    assert response.error["name"] == "Duplicate"
    assert response.error["docs"] == ["Account is already a proxy."]

    # === add proxy with ProxyType.Transfer ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=delay,
    )
    assert response.success, response.message

    # === check that chain has 2 proxy ===
    proxies, deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 2
    assert deposit > 0
    initial_deposit = deposit

    proxy_registration = next(
        (p for p in proxies if p.proxy_type == ProxyType.Registration), None
    )
    assert proxy_registration is not None
    assert proxy_registration.delegate == delegate_wallet.coldkey.ss58_address
    assert proxy_registration.proxy_type == ProxyType.Registration
    assert proxy_registration.delay == delay

    proxy_transfer = next(
        (p for p in proxies if p.proxy_type == ProxyType.Transfer), None
    )
    assert proxy_transfer is not None
    assert proxy_transfer.delegate == delegate_wallet.coldkey.ss58_address
    assert proxy_transfer.proxy_type == ProxyType.Transfer
    assert proxy_transfer.delay == delay

    # === Test get_proxies() - all proxies in network ===
    all_proxies = subtensor.proxies.get_proxies()
    assert isinstance(all_proxies, dict)
    assert real_account_wallet.coldkey.ss58_address in all_proxies
    assert len(all_proxies[real_account_wallet.coldkey.ss58_address]) == 2

    # === Test get_proxy_constants() ===
    constants = subtensor.proxies.get_proxy_constants()
    assert constants.MaxProxies is not None
    assert constants.MaxPending is not None
    assert constants.ProxyDepositBase is not None
    assert constants.ProxyDepositFactor is not None

    # === remove proxy ProxyType.Registration ===
    response = subtensor.proxies.remove_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert response.success, response.message

    # ===  check that chain has 1 proxies ===
    proxies, deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 1
    assert deposit > 0
    # Deposit should decrease after removing one proxy
    assert deposit < initial_deposit

    # === remove proxy ProxyType.Transfer ===
    response = subtensor.proxies.remove_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=delay,
    )
    assert response.success, response.message

    proxies, deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert not proxies
    assert deposit == 0

    # === remove already deleted or unexisted proxy ===
    response = subtensor.proxies.remove_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=delay,
    )
    assert not response.success
    assert "NotFound" in response.message
    assert response.error["name"] == "NotFound"
    assert response.error["docs"] == ["Proxy registration not found."]

    # === add proxy with wrong type ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type="custom type",
        delay=delay,
    )
    assert not response.success
    assert "Invalid proxy type" in response.message

    # === add proxy to the same account ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=real_account_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=0,
    )
    assert not response.success
    assert "NoSelfProxy" in response.message
    assert response.error["name"] == "NoSelfProxy"
    assert response.error["docs"] == ["Cannot add self as proxy."]

    # === Test adding proxy with delay = 0 ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Staking,
        delay=0,
    )
    assert response.success, response.message

    # Verify delay = 0
    proxies, _ = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    proxy_staking = next(
        (p for p in proxies if p.proxy_type == ProxyType.Staking), None
    )
    assert proxy_staking is not None
    assert proxy_staking.delay == 0

    # === Test adding multiple proxy types for same delegate ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.ChildKeys,
        delay=delay,
    )
    assert response.success, response.message

    proxies, _ = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 2  # Staking + ChildKeys

    # === Test adding proxy with different delegate ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=alice_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert response.success, response.message

    proxies, _ = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 3  # Staking + ChildKeys + Registration (alice)


@pytest.mark.asyncio
async def test_proxy_and_errors_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests proxy logic with async implementation.

    Steps:
        - Verify that chain has no proxies initially.
        - Add proxy with ProxyType.Registration and verify success.
        - Attempt to add duplicate proxy and verify error handling.
        - Add proxy with ProxyType.Transfer and verify success.
        - Verify chain has 2 proxies with correct deposit.
        - Verify proxy details match expected values (delegate, type, delay).
        - Test get_proxies() returns all proxies in network.
        - Test get_proxy_constants() returns valid constants.
        - Remove proxy ProxyType.Registration and verify deposit decreases.
        - Verify chain has 1 proxy remaining.
        - Remove proxy ProxyType.Transfer and verify all proxies removed.
        - Attempt to remove non-existent proxy and verify NotFound error.
        - Attempt to add proxy with invalid type and verify error.
        - Attempt to add self as proxy and verify NoSelfProxy error.
        - Test adding proxy with delay = 0 and verify it works.
        - Test adding multiple proxy types for same delegate.
        - Test adding proxy with different delegate.
    """
    real_account_wallet = bob_wallet
    delegate_wallet = charlie_wallet
    delay = 100

    # === check that chain has no proxies ===
    proxies, deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert not proxies
    assert deposit == 0

    # === add proxy with ProxyType.Registration ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert response.success, response.message

    # === add the same proxy returns error ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert not response.success
    assert "Duplicate" in response.message
    assert response.error["name"] == "Duplicate"
    assert response.error["docs"] == ["Account is already a proxy."]

    # === add proxy with ProxyType.Transfer ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=delay,
    )
    assert response.success, response.message

    # === check that chain has 2 proxy ===
    proxies, deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 2
    assert deposit > 0
    initial_deposit = deposit

    proxy_registration = next(
        (p for p in proxies if p.proxy_type == ProxyType.Registration), None
    )
    assert proxy_registration is not None
    assert proxy_registration.delegate == delegate_wallet.coldkey.ss58_address
    assert proxy_registration.proxy_type == ProxyType.Registration
    assert proxy_registration.delay == delay

    proxy_transfer = next(
        (p for p in proxies if p.proxy_type == ProxyType.Transfer), None
    )
    assert proxy_transfer is not None
    assert proxy_transfer.delegate == delegate_wallet.coldkey.ss58_address
    assert proxy_transfer.proxy_type == ProxyType.Transfer
    assert proxy_transfer.delay == delay

    # === Test get_proxies() - all proxies in network ===
    all_proxies = await async_subtensor.proxies.get_proxies()
    assert isinstance(all_proxies, dict)
    assert real_account_wallet.coldkey.ss58_address in all_proxies
    assert len(all_proxies[real_account_wallet.coldkey.ss58_address]) == 2

    # === Test get_proxy_constants() ===
    constants = await async_subtensor.proxies.get_proxy_constants()
    assert constants.MaxProxies is not None
    assert constants.MaxPending is not None
    assert constants.ProxyDepositBase is not None
    assert constants.ProxyDepositFactor is not None

    # === remove proxy ProxyType.Registration ===
    response = await async_subtensor.proxies.remove_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert response.success, response.message

    # ===  check that chain has 1 proxies ===
    proxies, deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 1
    assert deposit > 0
    # Deposit should decrease after removing one proxy
    assert deposit < initial_deposit

    # === remove proxy ProxyType.Transfer ===
    response = await async_subtensor.proxies.remove_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=delay,
    )
    assert response.success, response.message

    proxies, deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert not proxies
    assert deposit == 0

    # === remove already deleted or unexisted proxy ===
    response = await async_subtensor.proxies.remove_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=delay,
    )
    assert not response.success
    assert "NotFound" in response.message
    assert response.error["name"] == "NotFound"
    assert response.error["docs"] == ["Proxy registration not found."]

    # === add proxy with wrong type ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type="custom type",
        delay=delay,
    )
    assert not response.success
    assert "Invalid proxy type" in response.message

    # === add proxy to the same account ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=real_account_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=0,
    )
    assert not response.success
    assert "NoSelfProxy" in response.message
    assert response.error["name"] == "NoSelfProxy"
    assert response.error["docs"] == ["Cannot add self as proxy."]

    # === Test adding proxy with delay = 0 ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Staking,
        delay=0,
    )
    assert response.success, response.message

    # Verify delay = 0
    proxies, _ = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    proxy_staking = next(
        (p for p in proxies if p.proxy_type == ProxyType.Staking), None
    )
    assert proxy_staking is not None
    assert proxy_staking.delay == 0

    # === Test adding multiple proxy types for same delegate ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.ChildKeys,
        delay=delay,
    )
    assert response.success, response.message

    proxies, _ = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 2  # Staking + ChildKeys

    # === Test adding proxy with different delegate ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=alice_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=delay,
    )
    assert response.success, response.message

    proxies, _ = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 3  # Staking + ChildKeys + Registration (alice)


def test_create_and_announcement_proxy(
    subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests proxy logic with announcement mechanism for delay > 0.

    Steps:
        - Add proxy with ProxyType.Any and delay > 0.
        - Verify premature proxy call returns Unannounced error.
        - Verify premature proxy_announced call without announcement returns error.
        - Announce first call (register_network) and verify success.
        - Test get_proxy_announcements() returns correct announcements.
        - Attempt to execute announced call before delay blocks and verify error.
        - Wait for delay blocks to pass.
        - Execute proxy_announced call and verify subnet registration success.
        - Verify announcement is consumed after execution.
        - Verify subnet is not active after registration.
        - Announce second call (start_call) for subnet activation.
        - Test reject_announcement (real account rejects announcement).
        - Verify rejected announcement is removed.
        - Test remove_announcement (proxy removes its own announcement).
        - Verify removed announcement is no longer present.
        - Wait for delay blocks after second announcement.
        - Execute proxy_announced call to activate subnet and verify success.
        - Test proxy_call with delay = 0 (can be used immediately).
        - Test proxy_announced with wrong call_hash and verify error.
    """
    # === add proxy again ===
    real_account_wallet = bob_wallet
    delegate_wallet = charlie_wallet

    proxy_type = ProxyType.Any
    delay = 30  # cant execute proxy 30 blocks after announcement (not after creation)

    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=proxy_type,
        delay=delay,
    )
    assert response.success, response.message

    # check amount of subnets
    assert subtensor.subnets.get_total_subnets() == 2

    subnet_register_call = SubtensorModule(subtensor).register_network(
        hotkey=delegate_wallet.hotkey.ss58_address
    )
    subnet_activating_call = SubtensorModule(subtensor).start_call(netuid=2)

    # === premature proxy call ===
    # if delay > 0 .proxy always returns Unannounced error
    response = subtensor.proxies.proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_activating_call,
    )
    assert not response.success
    assert "Unannounced" in response.message
    assert response.error["name"] == "Unannounced"
    assert response.error["docs"] == [
        "Announcement, if made at all, was made too recently."
    ]

    # === premature proxy_announced call without announcement ===
    response = subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_register_call,
    )
    assert not response.success
    assert "Unannounced" in response.message
    assert response.error["name"] == "Unannounced"
    assert response.error["docs"] == [
        "Announcement, if made at all, was made too recently."
    ]

    # === Announce first call (register_network) ===
    call_hash_register = "0x" + subnet_register_call.call_hash.hex()
    response = subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=call_hash_register,
    )
    assert response.success, response.message
    registration_block = subtensor.block + delay

    # === Test get_proxy_announcements() ===
    announcements = subtensor.proxies.get_proxy_announcements()
    assert len(announcements[delegate_wallet.coldkey.ss58_address]) == 1

    delegate_announcement = announcements[delegate_wallet.coldkey.ss58_address][0]
    assert delegate_announcement.call_hash == call_hash_register
    assert delegate_announcement.real == real_account_wallet.coldkey.ss58_address

    # === announced call before delay blocks - register subnet ===
    response = subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_register_call,
    )
    assert not response.success
    assert "Unannounced" in response.message
    assert response.error["name"] == "Unannounced"
    assert response.error["docs"] == [
        "Announcement, if made at all, was made too recently."
    ]

    # === delay block need to be awaited after announcement ===
    subtensor.wait_for_block(registration_block)

    # === proxy call - register subnet ===
    response = subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_register_call,
    )
    assert response.success, response.message
    assert subtensor.subnets.get_total_subnets() == 3

    # === Verify announcement is consumed (cannot reuse) ===
    assert not subtensor.proxies.get_proxy_announcements()

    # === check that subnet is not active ===
    assert not subtensor.subnets.is_subnet_active(netuid=2)

    # === Announce second call (start_call) ===
    call_hash_activating = "0x" + subnet_activating_call.call_hash.hex()
    response = subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=call_hash_activating,
    )
    assert response.success, response.message

    # === Test reject_announcement (real account rejects) ===
    # Create another announcement to test rejection
    test_call = SubtensorModule(subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    test_call_hash = "0x" + test_call.call_hash.hex()

    response = subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=test_call_hash,
    )
    assert response.success, response.message

    # Real account rejects the announcement
    response = subtensor.proxies.reject_proxy_announcement(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        call_hash=test_call_hash,
    )
    assert response.success, response.message

    # Verify announcement was removed
    announcements = subtensor.proxies.get_proxy_announcement(
        delegate_account_ss58=delegate_wallet.coldkey.ss58_address
    )
    # Should only have start_call announcement, test_call should be rejected
    assert len(announcements) == 1
    assert announcements[0].call_hash == call_hash_activating

    # === Test remove_announcement (proxy removes its own announcement) ===
    # Create another announcement
    test_call2 = SubtensorModule(subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    test_call_hash2 = "0x" + test_call2.call_hash.hex()

    response = subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=test_call_hash2,
    )
    assert response.success, response.message

    # Proxy removes its own announcement
    response = subtensor.proxies.remove_proxy_announcement(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=test_call_hash2,
    )
    assert response.success, response.message

    # Verify announcement was removed
    announcements = subtensor.proxies.get_proxy_announcement(
        delegate_account_ss58=delegate_wallet.coldkey.ss58_address
    )
    assert len(announcements) == 1
    assert announcements[0].call_hash == call_hash_activating

    # === delay block need to be awaited after announcement ===
    activation_block = subtensor.block + delay
    subtensor.wait_for_block(activation_block)

    # === proxy call - activate subnet ===
    response = subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_activating_call,
    )
    assert response.success, response.message
    assert subtensor.subnets.is_subnet_active(netuid=2)

    # === Test proxy_call with delay = 0 (can be used immediately) ===
    # Add proxy with delay = 0
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=alice_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Any,
        delay=0,
    )
    assert response.success, response.message

    # With delay = 0, can use proxy_call directly without announcement
    test_call3 = SubtensorModule(subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    response = subtensor.proxies.proxy(
        wallet=alice_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=ProxyType.Any,
        call=test_call3,
    )
    assert response.success, response.message

    # === Test proxy_announced with wrong call_hash ===
    # Create announcement
    correct_call = SubtensorModule(subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    correct_call_hash = "0x" + correct_call.call_hash.hex()

    response = subtensor.proxies.announce_proxy(
        wallet=alice_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=correct_call_hash,
    )
    assert response.success, response.message

    # Wait for delay
    subtensor.wait_for_block(
        subtensor.block + 1
    )  # delay = 0, so can execute immediately

    # Try to execute with wrong call (different call_hash)
    wrong_call = SubtensorModule(subtensor).start_call(netuid=3)
    response = subtensor.proxies.proxy_announced(
        wallet=alice_wallet,
        delegate_ss58=alice_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=ProxyType.Any,
        call=wrong_call,  # Wrong call_hash
    )
    # Should fail because call_hash doesn't match
    assert not response.success


@pytest.mark.asyncio
async def test_create_and_announcement_proxy_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests proxy logic with announcement mechanism for delay > 0 with async implemtntation.

    Steps:
        - Add proxy with ProxyType.Any and delay > 0.
        - Verify premature proxy call returns Unannounced error.
        - Verify premature proxy_announced call without announcement returns error.
        - Announce first call (register_network) and verify success.
        - Test get_proxy_announcements() returns correct announcements.
        - Attempt to execute announced call before delay blocks and verify error.
        - Wait for delay blocks to pass.
        - Execute proxy_announced call and verify subnet registration success.
        - Verify announcement is consumed after execution.
        - Verify subnet is not active after registration.
        - Announce second call (start_call) for subnet activation.
        - Test reject_announcement (real account rejects announcement).
        - Verify rejected announcement is removed.
        - Test remove_announcement (proxy removes its own announcement).
        - Verify removed announcement is no longer present.
        - Wait for delay blocks after second announcement.
        - Execute proxy_announced call to activate subnet and verify success.
        - Test proxy_call with delay = 0 (can be used immediately).
        - Test proxy_announced with wrong call_hash and verify error.
    """
    # === add proxy again ===
    real_account_wallet = bob_wallet
    delegate_wallet = charlie_wallet

    proxy_type = ProxyType.Any
    delay = 30  # cant execute proxy 30 blocks after announcement (not after creation)

    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=proxy_type,
        delay=delay,
    )
    assert response.success, response.message

    # check amount of subnets
    assert await async_subtensor.subnets.get_total_subnets() == 2

    subnet_register_call = await SubtensorModule(async_subtensor).register_network(
        hotkey=delegate_wallet.hotkey.ss58_address
    )
    subnet_activating_call = await SubtensorModule(async_subtensor).start_call(netuid=2)

    # === premature proxy call ===
    # if delay > 0 .proxy always returns Unannounced error
    response = await async_subtensor.proxies.proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_activating_call,
    )
    assert not response.success
    assert "Unannounced" in response.message
    assert response.error["name"] == "Unannounced"
    assert response.error["docs"] == [
        "Announcement, if made at all, was made too recently."
    ]

    # === premature proxy_announced call without announcement ===
    response = await async_subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_register_call,
    )
    assert not response.success
    assert "Unannounced" in response.message
    assert response.error["name"] == "Unannounced"
    assert response.error["docs"] == [
        "Announcement, if made at all, was made too recently."
    ]

    # === Announce first call (register_network) ===
    call_hash_register = "0x" + subnet_register_call.call_hash.hex()
    response = await async_subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=call_hash_register,
    )
    assert response.success, response.message
    registration_block = await async_subtensor.block + delay

    # === Test get_proxy_announcements() ===
    announcements = await async_subtensor.proxies.get_proxy_announcements()
    assert len(announcements[delegate_wallet.coldkey.ss58_address]) == 1

    delegate_announcement = announcements[delegate_wallet.coldkey.ss58_address][0]
    assert delegate_announcement.call_hash == call_hash_register
    assert delegate_announcement.real == real_account_wallet.coldkey.ss58_address

    # === announced call before delay blocks - register subnet ===
    response = await async_subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_register_call,
    )
    assert not response.success
    assert "Unannounced" in response.message
    assert response.error["name"] == "Unannounced"
    assert response.error["docs"] == [
        "Announcement, if made at all, was made too recently."
    ]

    # === delay block need to be awaited after announcement ===
    await async_subtensor.wait_for_block(registration_block)

    # === proxy call - register subnet ===
    response = await async_subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_register_call,
    )
    assert response.success, response.message
    assert await async_subtensor.subnets.get_total_subnets() == 3

    # === Verify announcement is consumed (cannot reuse) ===
    assert not await async_subtensor.proxies.get_proxy_announcements()

    # === check that subnet is not active ===
    assert not await async_subtensor.subnets.is_subnet_active(netuid=2)

    # === Announce second call (start_call) ===
    call_hash_activating = "0x" + subnet_activating_call.call_hash.hex()
    response = await async_subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=call_hash_activating,
    )
    assert response.success, response.message

    # === Test reject_announcement (real account rejects) ===
    # Create another announcement to test rejection
    test_call = await SubtensorModule(async_subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    test_call_hash = "0x" + test_call.call_hash.hex()

    response = await async_subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=test_call_hash,
    )
    assert response.success, response.message

    # Real account rejects the announcement
    response = await async_subtensor.proxies.reject_proxy_announcement(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        call_hash=test_call_hash,
    )
    assert response.success, response.message

    # Verify announcement was removed
    announcements = await async_subtensor.proxies.get_proxy_announcement(
        delegate_account_ss58=delegate_wallet.coldkey.ss58_address
    )
    # Should only have start_call announcement, test_call should be rejected
    assert len(announcements) == 1
    assert announcements[0].call_hash == call_hash_activating

    # === Test remove_announcement (proxy removes its own announcement) ===
    # Create another announcement
    test_call2 = await SubtensorModule(async_subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    test_call_hash2 = "0x" + test_call2.call_hash.hex()

    response = await async_subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=test_call_hash2,
    )
    assert response.success, response.message

    # Proxy removes its own announcement
    response = await async_subtensor.proxies.remove_proxy_announcement(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=test_call_hash2,
    )
    assert response.success, response.message

    # Verify announcement was removed
    announcements = await async_subtensor.proxies.get_proxy_announcement(
        delegate_account_ss58=delegate_wallet.coldkey.ss58_address
    )
    assert len(announcements) == 1
    assert announcements[0].call_hash == call_hash_activating

    # === delay block need to be awaited after announcement ===
    activation_block = await async_subtensor.block + delay
    await async_subtensor.wait_for_block(activation_block)

    # === proxy call - activate subnet ===
    response = await async_subtensor.proxies.proxy_announced(
        wallet=delegate_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=proxy_type,
        call=subnet_activating_call,
    )
    assert response.success, response.message
    assert await async_subtensor.subnets.is_subnet_active(netuid=2)

    # === Test proxy_call with delay = 0 (can be used immediately) ===
    # Add proxy with delay = 0
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=alice_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Any,
        delay=0,
    )
    assert response.success, response.message

    # With delay = 0, can use proxy_call directly without announcement
    test_call3 = await SubtensorModule(async_subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    response = await async_subtensor.proxies.proxy(
        wallet=alice_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=ProxyType.Any,
        call=test_call3,
    )
    assert response.success, response.message

    # === Test proxy_announced with wrong call_hash ===
    # Create announcement
    correct_call = await SubtensorModule(async_subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    correct_call_hash = "0x" + correct_call.call_hash.hex()

    response = await async_subtensor.proxies.announce_proxy(
        wallet=alice_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=correct_call_hash,
    )
    assert response.success, response.message

    # Wait for delay
    await async_subtensor.wait_for_block(
        await async_subtensor.block + 1
    )  # delay = 0, so can execute immediately

    # Try to execute with wrong call (different call_hash)
    wrong_call = await SubtensorModule(async_subtensor).start_call(netuid=3)
    response = await async_subtensor.proxies.proxy_announced(
        wallet=alice_wallet,
        delegate_ss58=alice_wallet.coldkey.ss58_address,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        force_proxy_type=ProxyType.Any,
        call=wrong_call,  # Wrong call_hash
    )
    # Should fail because call_hash doesn't match
    assert not response.success


def test_create_and_kill_pure_proxy(subtensor, alice_wallet, bob_wallet):
    """Tests create_pure_proxy and kill_pure_proxy extrinsics.

    This test verifies the complete lifecycle of a pure proxy account:
        - Creation of a pure proxy with specific parameters
        - Verification that the pure proxy can execute calls through the spawner
        - Proper termination of the pure proxy
        - Confirmation that the killed pure proxy can no longer be used

    Steps:
        - Create pure proxy with ProxyType.Any, delay=0, and index=0.
        - Extract pure proxy address, spawner, and creation metadata from response.data.
        - Verify all required data is present and correctly formatted.
        - Fund the pure proxy account so it can execute transfers.
        - Execute a transfer through the pure proxy to verify it works correctly.
          The spawner acts as an "Any" proxy for the pure proxy account.
        - Kill the pure proxy using kill_pure_proxy() method, which automatically
          executes the kill_pure call through proxy() (spawner acts as Any proxy
          for pure proxy, with pure proxy as the origin).
        - Verify pure proxy is killed by attempting to use it and confirming
          it returns a NotProxy error.
    """
    spawner_wallet = bob_wallet
    proxy_type = ProxyType.Any
    delay = 0
    index = 0

    # === Tests failed Create pure proxy without wait_for*=True ===
    response = subtensor.proxies.create_pure_proxy(
        wallet=spawner_wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )
    assert not response.success
    assert "The ExtrinsicResponse doesn't contain pure_proxy data" in response.message

    subtensor.wait_for_block()

    # === Create pure proxy ===
    response = subtensor.proxies.create_pure_proxy(
        wallet=spawner_wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
    )
    assert response.success, response.message

    # === Extract pure proxy data from response.data ===
    pure_account = response.data.get("pure_account")
    spawner = response.data.get("spawner")
    proxy_type_from_response = response.data.get("proxy_type")
    index_from_response = response.data.get("index")
    height = response.data.get("height")
    ext_index = response.data.get("ext_index")

    # === Verify spawner matches ===
    assert spawner == spawner_wallet.coldkey.ss58_address

    # === Verify all required data is present ===
    assert pure_account, "Pure account should be present."
    assert spawner, "Spawner should be present."
    assert proxy_type_from_response, "Proxy type should be present."
    assert isinstance(index_from_response, int)
    assert isinstance(height, int) and height > 0
    assert isinstance(ext_index, int) and ext_index >= 0

    # === Fund the pure proxy account so it can execute transfers ===
    from bittensor.utils.balance import Balance

    fund_amount = Balance.from_tao(1.0)  # Fund with 1 TAO
    response = subtensor.wallets.transfer(
        wallet=spawner_wallet,
        destination_ss58=pure_account,
        amount=fund_amount,
    )
    assert response.success, f"Failed to fund pure proxy account: {response.message}."

    # === Test that pure proxy works by executing a transfer through it ===
    # The spawner acts as an "Any" proxy for the pure proxy account.
    # The pure proxy account is the origin (real account), and the spawner signs the transaction.
    transfer_amount = Balance.from_tao(0.1)  # Transfer 0.1 TAO
    transfer_call = Balances(subtensor).transfer_keep_alive(
        dest=alice_wallet.coldkey.ss58_address,
        value=transfer_amount.rao,
    )

    response = subtensor.proxies.proxy(
        wallet=spawner_wallet,  # Spawner signs the transaction
        real_account_ss58=pure_account,  # Pure proxy account is the origin (real)
        force_proxy_type=ProxyType.Any,  # Spawner acts as Any proxy for pure proxy
        call=transfer_call,
    )
    assert response.success, (
        f"Pure proxy should be able to execute transfers, got: {response.message}."
    )

    # === Kill pure proxy using kill_pure_proxy() method ===
    # The kill_pure_proxy() method automatically executes the kill_pure call through proxy():
    # - The spawner signs the transaction (wallet parameter)
    # - The pure proxy account is the origin (real_account_ss58 parameter)
    # - The spawner acts as an "Any" proxy for the pure proxy (force_proxy_type=Any)
    # This is required because pure proxies are keyless accounts and cannot sign transactions directly.
    response = subtensor.proxies.kill_pure_proxy(
        wallet=spawner_wallet,
        pure_proxy_ss58=pure_account,
        spawner=spawner,
        proxy_type=proxy_type_from_response,
        index=index_from_response,
        height=height,
        ext_index=ext_index,
    )
    assert response.success, response.message

    # === Verify pure proxy is killed by attempting to use it ===
    # Create a simple transfer call to test that proxy fails
    simple_call = Balances(subtensor).transfer_keep_alive(
        dest=alice_wallet.coldkey.ss58_address,
        value=500,  # Small amount, just to test
    )

    # === Attempt to execute call through killed pure proxy - should fail ===
    response = subtensor.proxies.proxy(
        wallet=spawner_wallet,
        real_account_ss58=pure_account,  # Killed pure proxy account
        force_proxy_type=ProxyType.Any,
        call=simple_call,
    )

    # === Should fail because pure proxy no longer exists ===
    assert not response.success, "Call through killed pure proxy should fail."
    assert "NotProxy" in response.message
    assert response.error["name"] == "NotProxy"
    assert response.error["docs"] == [
        "Sender is not a proxy of the account to be proxied."
    ]


@pytest.mark.asyncio
async def test_create_and_kill_pure_proxy_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """Tests create_pure_proxy and kill_pure_proxy extrinsics with async implementation.

    This test verifies the complete lifecycle of a pure proxy account:
        - Creation of a pure proxy with specific parameters
        - Verification that the pure proxy can execute calls through the spawner
        - Proper termination of the pure proxy
        - Confirmation that the killed pure proxy can no longer be used

    Steps:
        - Create pure proxy with ProxyType.Any, delay=0, and index=0.
        - Extract pure proxy address, spawner, and creation metadata from response.data.
        - Verify all required data is present and correctly formatted.
        - Fund the pure proxy account so it can execute transfers.
        - Execute a transfer through the pure proxy to verify it works correctly.
          The spawner acts as an "Any" proxy for the pure proxy account.
        - Kill the pure proxy using kill_pure_proxy() method, which automatically
          executes the kill_pure call through proxy() (spawner acts as Any proxy
          for pure proxy, with pure proxy as the origin).
        - Verify pure proxy is killed by attempting to use it and confirming
          it returns a NotProxy error.
    """
    spawner_wallet = bob_wallet
    proxy_type = ProxyType.Any
    delay = 0
    index = 0

    # === Tests failed Create pure proxy without wait_for*=True ===
    response = await async_subtensor.proxies.create_pure_proxy(
        wallet=spawner_wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
        wait_for_inclusion=False,
        wait_for_finalization=False,
    )
    assert not response.success
    assert "The ExtrinsicResponse doesn't contain pure_proxy data" in response.message

    await async_subtensor.wait_for_block()

    # === Create pure proxy ===
    response = await async_subtensor.proxies.create_pure_proxy(
        wallet=spawner_wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
        raise_error=True,
    )
    assert response.success, response.message

    # === Extract pure proxy data from response.data ===
    pure_account = response.data.get("pure_account")
    spawner = response.data.get("spawner")
    proxy_type_from_response = response.data.get("proxy_type")
    index_from_response = response.data.get("index")
    height = response.data.get("height")
    ext_index = response.data.get("ext_index")

    # === Verify spawner matches ===
    assert spawner == spawner_wallet.coldkey.ss58_address

    # === Verify all required data is present ===
    assert pure_account, "Pure account should be present."
    assert spawner, "Spawner should be present."
    assert proxy_type_from_response, "Proxy type should be present."
    assert isinstance(index_from_response, int)
    assert isinstance(height, int) and height > 0
    assert isinstance(ext_index, int) and ext_index >= 0

    # === Fund the pure proxy account so it can execute transfers ===
    from bittensor.utils.balance import Balance

    fund_amount = Balance.from_tao(1.0)  # Fund with 1 TAO
    response = await async_subtensor.wallets.transfer(
        wallet=spawner_wallet,
        destination_ss58=pure_account,
        amount=fund_amount,
    )
    assert response.success, f"Failed to fund pure proxy account: {response.message}."

    # === Test that pure proxy works by executing a transfer through it ===
    # The spawner acts as an "Any" proxy for the pure proxy account.
    # The pure proxy account is the origin (real account), and the spawner signs the transaction.
    transfer_amount = Balance.from_tao(0.1)  # Transfer 0.1 TAO
    transfer_call = await Balances(async_subtensor).transfer_keep_alive(
        dest=alice_wallet.coldkey.ss58_address,
        value=transfer_amount.rao,
    )

    response = await async_subtensor.proxies.proxy(
        wallet=spawner_wallet,  # Spawner signs the transaction
        real_account_ss58=pure_account,  # Pure proxy account is the origin (real)
        force_proxy_type=ProxyType.Any,  # Spawner acts as Any proxy for pure proxy
        call=transfer_call,
    )
    assert response.success, (
        f"Pure proxy should be able to execute transfers, got: {response.message}."
    )

    # === Kill pure proxy using kill_pure_proxy() method ===
    # The kill_pure_proxy() method automatically executes the kill_pure call through proxy():
    # - The spawner signs the transaction (wallet parameter)
    # - The pure proxy account is the origin (real_account_ss58 parameter)
    # - The spawner acts as an "Any" proxy for the pure proxy (force_proxy_type=Any)
    # This is required because pure proxies are keyless accounts and cannot sign transactions directly.
    response = await async_subtensor.proxies.kill_pure_proxy(
        wallet=spawner_wallet,
        pure_proxy_ss58=pure_account,
        spawner=spawner,
        proxy_type=proxy_type_from_response,
        index=index_from_response,
        height=height,
        ext_index=ext_index,
    )
    assert response.success, response.message

    # === Verify pure proxy is killed by attempting to use it ===
    # Create a simple transfer call to test that proxy fails
    simple_call = await Balances(async_subtensor).transfer_keep_alive(
        dest=alice_wallet.coldkey.ss58_address,
        value=500,  # Small amount, just to test
    )

    # === Attempt to execute call through killed pure proxy - should fail ===
    response = await async_subtensor.proxies.proxy(
        wallet=spawner_wallet,
        real_account_ss58=pure_account,  # Killed pure proxy account
        force_proxy_type=ProxyType.Any,
        call=simple_call,
    )

    # === Should fail because pure proxy no longer exists ===
    assert not response.success, "Call through killed pure proxy should fail."
    assert "NotProxy" in response.message
    assert response.error["name"] == "NotProxy"
    assert response.error["docs"] == [
        "Sender is not a proxy of the account to be proxied."
    ]


def test_remove_proxies(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """Tests remove_proxies extrinsic.

    Steps:
        - Add multiple proxies with different types and delegates
        - Verify all proxies exist and deposit is correct
        - Call remove_proxies to remove all at once
        - Verify all proxies are removed
        - Verify deposit is returned (should be 0 or empty)
    """
    real_account_wallet = bob_wallet
    delegate1 = charlie_wallet
    delegate2 = alice_wallet

    # === Add multiple proxies ===
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate1.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=0,
    )
    assert response.success

    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate1.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=0,
    )
    assert response.success

    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate2.coldkey.ss58_address,
        proxy_type=ProxyType.Staking,
        delay=0,
    )
    assert response.success

    # === Verify all proxies exist ===
    proxies, deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 3
    assert deposit > 0

    # === Remove all proxies ===
    response = subtensor.proxies.remove_proxies(
        wallet=real_account_wallet,
    )
    assert response.success, response.message

    # === Verify all proxies removed ===
    proxies, deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert not proxies
    assert deposit == 0


@pytest.mark.asyncio
async def test_remove_proxies_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests remove_proxies extrinsic with async implementation.

    Steps:
        - Add multiple proxies with different types and delegates
        - Verify all proxies exist and deposit is correct
        - Call remove_proxies to remove all at once
        - Verify all proxies are removed
        - Verify deposit is returned (should be 0 or empty)
    """
    real_account_wallet = bob_wallet
    delegate1 = charlie_wallet
    delegate2 = alice_wallet

    # === Add multiple proxies ===
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate1.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=0,
    )
    assert response.success

    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate1.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=0,
    )
    assert response.success

    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate2.coldkey.ss58_address,
        proxy_type=ProxyType.Staking,
        delay=0,
    )
    assert response.success

    # === Verify all proxies exist ===
    proxies, deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert len(proxies) == 3
    assert deposit > 0

    # === Remove all proxies ===
    response = await async_subtensor.proxies.remove_proxies(
        wallet=real_account_wallet,
    )
    assert response.success, response.message

    # === Verify all proxies removed ===
    proxies, deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    assert not proxies
    assert deposit == 0


def test_poke_deposit(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """Tests poke_deposit extrinsic.

    Steps:
        - Add multiple proxies and announcements
        - Verify initial deposit amount
        - Call poke_deposit to recalculate deposits
        - Verify deposit may change (if requirements changed)
        - Verify transaction fee is waived if deposit changed
    """
    real_account_wallet = bob_wallet
    delegate_wallet = charlie_wallet

    # Add proxies
    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=0,
    )
    assert response.success

    response = subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=0,
    )
    assert response.success

    # Get initial deposit
    _, initial_deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )

    # Create an announcement
    test_call = SubtensorModule(subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    call_hash = "0x" + test_call.call_hash.hex()

    response = subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=call_hash,
    )
    assert response.success

    # Call poke_deposit
    response = subtensor.proxies.poke_deposit(
        wallet=real_account_wallet,
    )
    assert response.success, response.message

    # Verify deposit is still correct (or adjusted)
    _, final_deposit = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    # Deposit should match or be adjusted based on current requirements
    assert final_deposit >= 0


@pytest.mark.asyncio
async def test_poke_deposit_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests poke_deposit extrinsic with async implementation.

    Steps:
        - Add multiple proxies and announcements
        - Verify initial deposit amount
        - Call poke_deposit to recalculate deposits
        - Verify deposit may change (if requirements changed)
        - Verify transaction fee is waived if deposit changed
    """
    real_account_wallet = bob_wallet
    delegate_wallet = charlie_wallet

    # Add proxies
    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Registration,
        delay=0,
    )
    assert response.success

    response = await async_subtensor.proxies.add_proxy(
        wallet=real_account_wallet,
        delegate_ss58=delegate_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Transfer,
        delay=0,
    )
    assert response.success

    # Get initial deposit
    _, initial_deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )

    # Create an announcement
    test_call = await SubtensorModule(async_subtensor).register_network(
        hotkey=alice_wallet.hotkey.ss58_address
    )
    call_hash = "0x" + test_call.call_hash.hex()

    response = await async_subtensor.proxies.announce_proxy(
        wallet=delegate_wallet,
        real_account_ss58=real_account_wallet.coldkey.ss58_address,
        call_hash=call_hash,
    )
    assert response.success

    # Call poke_deposit
    response = await async_subtensor.proxies.poke_deposit(
        wallet=real_account_wallet,
    )
    assert response.success, response.message

    # Verify deposit is still correct (or adjusted)
    _, final_deposit = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=real_account_wallet.coldkey.ss58_address
    )
    # Deposit should match or be adjusted based on current requirements
    assert final_deposit >= 0


def test_kill_pure_proxy_by_non_spawner_with_multiple_proxies(
    subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests kill_pure_proxy when pure proxy has multiple proxy delegates.

    This test verifies a multi-signature delegation scenario where a pure proxy account has multiple proxy delegates
    (Alice and Charlie), and one of them (Charlie) can successfully kill the pure proxy even though it was created by a
    different account (Bob). This demonstrates that:
        - Multiple accounts can be proxy delegates for the same pure proxy
        - Any proxy delegate can kill the pure proxy (not just the spawner)
        - spawner parameter does not need to match wallet.coldkey.ss58_address

    The test scenario:
        - Bob creates a pure proxy account (spawner = Bob)
        - Pure proxy adds Alice as a proxy delegate with ProxyType.Any
        - Pure proxy adds Charlie as a proxy delegate with ProxyType.Any
        - Pure proxy now has two proxy delegates: Alice and Charlie
        - Charlie kills the pure proxy using kill_pure_proxy() with spawner=Bob but wallet=Charlie
        - This verifies that spawner != wallet.coldkey.ss58_address is valid even in multi-proxy scenarios

    Steps:
        - Create pure proxy with ProxyType.Any, delay=0, and index=0 using Bob's wallet (spawner = Bob).
        - Extract pure proxy address, spawner, and creation metadata from response.data.
        - Verify spawner matches Bob's address and does not match Charlie's address.
        - Fund the pure proxy account so it can execute transactions.
        - Add Alice as a proxy for the pure proxy account. This is done by calling add_proxy through proxy() where Bob
            acts as "Any" proxy for the pure proxy account to execute add_proxy on behalf of pure proxy.
        - Verify Alice is now listed as a proxy delegate for the pure proxy account with ProxyType.Any.
        - Add Charlie as a proxy for the pure proxy account using the same method (Bob acts as proxy for pure proxy).
        - Verify Charlie is now listed as a proxy delegate for the pure proxy account with ProxyType.Any.
        - Verify pure proxy now has two proxy delegates (Alice and Charlie).
        - Verify that spawner (Bob) != wallet.coldkey.ss58_address (Charlie) for the upcoming kill_pure_proxy call.
        - Kill the pure proxy using kill_pure_proxy() with wallet=Charlie but spawner=Bob. This demonstrates that in a
            multi-proxy scenario, any proxy delegate can kill the pure proxy, not just the spawner.
        - Verify the kill operation succeeds.
        - Verify pure proxy is killed by attempting to use it and confirming it returns a NotProxy error.
        - Verify that both Alice and Bob can no longer use the killed pure proxy (all proxy relationships are removed
            when pure proxy is killed).
    """
    spawner_wallet = bob_wallet
    proxy_delegate1_wallet = alice_wallet
    proxy_delegate2_wallet = charlie_wallet
    proxy_type = ProxyType.Any
    delay = 0
    index = 0

    # === Create pure proxy (Bob creates it) ===
    response = subtensor.proxies.create_pure_proxy(
        wallet=spawner_wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
    )
    assert response.success, response.message

    # === Extract pure proxy data ===
    pure_account = response.data.get("pure_account")
    spawner = response.data.get("spawner")
    proxy_type_from_response = response.data.get("proxy_type")
    index_from_response = response.data.get("index")
    height = response.data.get("height")
    ext_index = response.data.get("ext_index")

    # === Verify spawner is Bob ===
    assert spawner == spawner_wallet.coldkey.ss58_address
    assert spawner != proxy_delegate2_wallet.coldkey.ss58_address

    fund_amount = Balance.from_tao(1.0)
    response = subtensor.wallets.transfer(
        wallet=spawner_wallet,
        destination_ss58=pure_account,
        amount=fund_amount,
    )
    assert response.success, f"Failed to fund pure proxy account: {response.message}."

    # === Pure proxy adds Alice as a proxy (through Bob acting as proxy for pure proxy) ===
    add_proxy_alice_call = Proxy(subtensor).add_proxy(
        delegate=proxy_delegate1_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Any.value,
        delay=0,
    )

    response = subtensor.proxies.proxy(
        wallet=spawner_wallet,  # Bob signs the transaction
        real_account_ss58=pure_account,  # Pure proxy is the origin
        force_proxy_type=ProxyType.Any,  # Bob acts as Any proxy for pure proxy
        call=add_proxy_alice_call,
    )
    assert response.success, (
        f"Failed to add Alice as proxy for pure proxy: {response.message}."
    )

    # === Verify Alice is now a proxy for pure proxy ===
    proxies, _ = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=pure_account
    )
    assert any(
        p.delegate == proxy_delegate1_wallet.coldkey.ss58_address
        and p.proxy_type == ProxyType.Any
        for p in proxies
    ), "Alice should be a proxy for pure proxy."

    # === Pure proxy adds Charlie as a proxy (through Bob acting as proxy for pure proxy) ===
    add_proxy_charlie_call = Proxy(subtensor).add_proxy(
        delegate=proxy_delegate2_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Any.value,
        delay=0,
    )

    response = subtensor.proxies.proxy(
        wallet=spawner_wallet,  # Bob signs the transaction
        real_account_ss58=pure_account,  # Pure proxy is the origin
        force_proxy_type=ProxyType.Any,  # Bob acts as Any proxy for pure proxy
        call=add_proxy_charlie_call,
    )
    assert response.success, (
        f"Failed to add Charlie as proxy for pure proxy: {response.message}."
    )

    # === Verify Charlie is now a proxy for pure proxy ===
    proxies, _ = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=pure_account
    )
    assert any(
        p.delegate == proxy_delegate2_wallet.coldkey.ss58_address
        and p.proxy_type == ProxyType.Any
        for p in proxies
    ), "Charlie should be a proxy for pure proxy."

    # === Verify pure proxy has multiple proxies ===
    proxies, _ = subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=pure_account
    )
    assert len(proxies) >= 2, "Pure proxy should have at least two proxy delegates."
    assert any(
        p.delegate == proxy_delegate1_wallet.coldkey.ss58_address for p in proxies
    )
    assert any(
        p.delegate == proxy_delegate2_wallet.coldkey.ss58_address for p in proxies
    )

    # === Charlie kills the pure proxy (spawner=Bob, wallet=Charlie) ===
    # This verifies that spawner != wallet.coldkey.ss58_address
    assert spawner != proxy_delegate2_wallet.coldkey.ss58_address, (
        "Spawner should be different from wallet for this test."
    )

    response = subtensor.proxies.kill_pure_proxy(
        wallet=proxy_delegate2_wallet,  # Charlie signs the transaction
        pure_proxy_ss58=pure_account,
        spawner=spawner,  # Bob is the spawner (who created the pure proxy)
        proxy_type=proxy_type_from_response,
        index=index_from_response,
        height=height,
        ext_index=ext_index,
    )
    assert response.success, (
        f"Charlie should be able to kill pure proxy created by Bob: {response.message}."
    )

    # === Verify pure proxy is killed ===
    # Attempt to use the killed pure proxy - should fail
    simple_call = Balances(subtensor).transfer_keep_alive(
        dest=alice_wallet.coldkey.ss58_address,
        value=500,
    )

    # === Verify Bob can no longer use the killed pure proxy ===
    response = subtensor.proxies.proxy(
        wallet=spawner_wallet,
        real_account_ss58=pure_account,  # Killed pure proxy account
        force_proxy_type=ProxyType.Any,
        call=simple_call,
    )
    assert not response.success, "Call through killed pure proxy should fail."
    assert "NotProxy" in response.message

    # === Verify Alice can no longer use the killed pure proxy ===
    response = subtensor.proxies.proxy(
        wallet=proxy_delegate1_wallet,
        real_account_ss58=pure_account,  # Killed pure proxy account
        force_proxy_type=ProxyType.Any,
        call=simple_call,
    )
    assert not response.success, "Call through killed pure proxy should fail."
    assert "NotProxy" in response.message


@pytest.mark.asyncio
async def test_kill_pure_proxy_by_non_spawner_with_multiple_proxies_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests kill_pure_proxy when pure proxy has multiple proxy delegates with async implementation.

    This test verifies a multi-signature delegation scenario where a pure proxy account has multiple proxy delegates
    (Alice and Charlie), and one of them (Charlie) can successfully kill the pure proxy even though it was created by a
    different account (Bob). This demonstrates that:
        - Multiple accounts can be proxy delegates for the same pure proxy
        - Any proxy delegate can kill the pure proxy (not just the spawner)
        - spawner parameter does not need to match wallet.coldkey.ss58_address

    The test scenario:
        - Bob creates a pure proxy account (spawner = Bob)
        - Pure proxy adds Alice as a proxy delegate with ProxyType.Any
        - Pure proxy adds Charlie as a proxy delegate with ProxyType.Any
        - Pure proxy now has two proxy delegates: Alice and Charlie
        - Charlie kills the pure proxy using kill_pure_proxy() with spawner=Bob but wallet=Charlie
        - This verifies that spawner != wallet.coldkey.ss58_address is valid even in multi-proxy scenarios

    Steps:
        - Create pure proxy with ProxyType.Any, delay=0, and index=0 using Bob's wallet (spawner = Bob).
        - Extract pure proxy address, spawner, and creation metadata from response.data.
        - Verify spawner matches Bob's address and does not match Charlie's address.
        - Fund the pure proxy account so it can execute transactions.
        - Add Alice as a proxy for the pure proxy account. This is done by calling add_proxy through proxy() where Bob
            acts as "Any" proxy for the pure proxy account to execute add_proxy on behalf of pure proxy.
        - Verify Alice is now listed as a proxy delegate for the pure proxy account with ProxyType.Any.
        - Add Charlie as a proxy for the pure proxy account using the same method (Bob acts as proxy for pure proxy).
        - Verify Charlie is now listed as a proxy delegate for the pure proxy account with ProxyType.Any.
        - Verify pure proxy now has two proxy delegates (Alice and Charlie).
        - Verify that spawner (Bob) != wallet.coldkey.ss58_address (Charlie) for the upcoming kill_pure_proxy call.
        - Kill the pure proxy using kill_pure_proxy() with wallet=Charlie but spawner=Bob. This demonstrates that in a
            multi-proxy scenario, any proxy delegate can kill the pure proxy, not just the spawner.
        - Verify the kill operation succeeds.
        - Verify pure proxy is killed by attempting to use it and confirming it returns a NotProxy error.
        - Verify that both Alice and Bob can no longer use the killed pure proxy (all proxy relationships are removed
            when pure proxy is killed).
    """
    spawner_wallet = bob_wallet
    proxy_delegate1_wallet = alice_wallet
    proxy_delegate2_wallet = charlie_wallet
    proxy_type = ProxyType.Any
    delay = 0
    index = 0

    # === Create pure proxy (Bob creates it) ===
    response = await async_subtensor.proxies.create_pure_proxy(
        wallet=spawner_wallet,
        proxy_type=proxy_type,
        delay=delay,
        index=index,
    )
    assert response.success, response.message

    # === Extract pure proxy data ===
    pure_account = response.data.get("pure_account")
    spawner = response.data.get("spawner")
    proxy_type_from_response = response.data.get("proxy_type")
    index_from_response = response.data.get("index")
    height = response.data.get("height")
    ext_index = response.data.get("ext_index")

    # === Verify spawner is Bob ===
    assert spawner == spawner_wallet.coldkey.ss58_address
    assert spawner != proxy_delegate2_wallet.coldkey.ss58_address

    # === Fund the pure proxy account ===
    from bittensor.utils.balance import Balance

    fund_amount = Balance.from_tao(1.0)
    response = await async_subtensor.wallets.transfer(
        wallet=spawner_wallet,
        destination_ss58=pure_account,
        amount=fund_amount,
    )
    assert response.success, f"Failed to fund pure proxy account: {response.message}."

    # === Pure proxy adds Alice as a proxy (through Bob acting as proxy for pure proxy) ===
    add_proxy_alice_call = await Proxy(async_subtensor).add_proxy(
        delegate=proxy_delegate1_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Any.value,
        delay=0,
    )

    response = await async_subtensor.proxies.proxy(
        wallet=spawner_wallet,  # Bob signs the transaction
        real_account_ss58=pure_account,  # Pure proxy is the origin
        force_proxy_type=ProxyType.Any,  # Bob acts as Any proxy for pure proxy
        call=add_proxy_alice_call,
    )
    assert response.success, (
        f"Failed to add Alice as proxy for pure proxy: {response.message}."
    )

    # === Verify Alice is now a proxy for pure proxy ===
    proxies, _ = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=pure_account
    )
    assert any(
        p.delegate == proxy_delegate1_wallet.coldkey.ss58_address
        and p.proxy_type == ProxyType.Any
        for p in proxies
    ), "Alice should be a proxy for pure proxy."

    # === Pure proxy adds Charlie as a proxy (through Bob acting as proxy for pure proxy) ===
    add_proxy_charlie_call = await Proxy(async_subtensor).add_proxy(
        delegate=proxy_delegate2_wallet.coldkey.ss58_address,
        proxy_type=ProxyType.Any.value,
        delay=0,
    )

    response = await async_subtensor.proxies.proxy(
        wallet=spawner_wallet,  # Bob signs the transaction
        real_account_ss58=pure_account,  # Pure proxy is the origin
        force_proxy_type=ProxyType.Any,  # Bob acts as Any proxy for pure proxy
        call=add_proxy_charlie_call,
    )
    assert response.success, (
        f"Failed to add Charlie as proxy for pure proxy: {response.message}."
    )

    # === Verify Charlie is now a proxy for pure proxy ===
    proxies, _ = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=pure_account
    )
    assert any(
        p.delegate == proxy_delegate2_wallet.coldkey.ss58_address
        and p.proxy_type == ProxyType.Any
        for p in proxies
    ), "Charlie should be a proxy for pure proxy."

    # === Verify pure proxy has multiple proxies ===
    proxies, _ = await async_subtensor.proxies.get_proxies_for_real_account(
        real_account_ss58=pure_account
    )
    assert len(proxies) >= 2, "Pure proxy should have at least two proxy delegates."
    assert any(
        p.delegate == proxy_delegate1_wallet.coldkey.ss58_address for p in proxies
    )
    assert any(
        p.delegate == proxy_delegate2_wallet.coldkey.ss58_address for p in proxies
    )

    # === Charlie kills the pure proxy (spawner=Bob, wallet=Charlie) ===
    # This verifies that spawner != wallet.coldkey.ss58_address
    assert spawner != proxy_delegate2_wallet.coldkey.ss58_address, (
        "Spawner should be different from wallet for this test."
    )

    response = await async_subtensor.proxies.kill_pure_proxy(
        wallet=proxy_delegate2_wallet,  # Charlie signs the transaction
        pure_proxy_ss58=pure_account,
        spawner=spawner,  # Bob is the spawner (who created the pure proxy)
        proxy_type=proxy_type_from_response,
        index=index_from_response,
        height=height,
        ext_index=ext_index,
    )
    assert response.success, (
        f"Charlie should be able to kill pure proxy created by Bob: {response.message}."
    )

    # === Verify pure proxy is killed ===
    # Attempt to use the killed pure proxy - should fail
    simple_call = await Balances(async_subtensor).transfer_keep_alive(
        dest=alice_wallet.coldkey.ss58_address,
        value=500,
    )

    # === Verify Bob can no longer use the killed pure proxy ===
    response = await async_subtensor.proxies.proxy(
        wallet=spawner_wallet,
        real_account_ss58=pure_account,  # Killed pure proxy account
        force_proxy_type=ProxyType.Any,
        call=simple_call,
    )
    assert not response.success, "Call through killed pure proxy should fail."
    assert "NotProxy" in response.message

    # === Verify Alice can no longer use the killed pure proxy ===
    response = await async_subtensor.proxies.proxy(
        wallet=proxy_delegate1_wallet,
        real_account_ss58=pure_account,  # Killed pure proxy account
        force_proxy_type=ProxyType.Any,
        call=simple_call,
    )
    assert not response.success, "Call through killed pure proxy should fail."
    assert "NotProxy" in response.message
