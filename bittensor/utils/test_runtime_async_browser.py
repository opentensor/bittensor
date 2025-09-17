#!/usr/bin/env python

# testing code for runtime async browser

import bittensor as bt
import asyncio
import sys

calls = [l for l in '''
# subnet registration info
await bt.async_runtime.SubtensorModule.NetworkLastRegistered()
await bt.async_runtime.SubtensorModule.NetworkLastLockCost()
await bt.async_runtime.SubtensorModule.NetworkImmunityPeriod()
await bt.async_runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost()
await bt.async_runtime.SubtensorModule.SubnetOwner[29]
await bt.async_runtime.SubtensorModule.SubnetLocked[29]
await bt.async_runtime.SubtensorModule.NetworkRegisteredAt[29]
await bt.async_runtime.SubtensorModule.NetworkLastLockCost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29])
await bt.async_runtime.SubtensorModule.NetworkLastLockCost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29]-1)
await bt.async_runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29])
await bt.async_runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29]-1)
# hotkey registration and commitment info
await bt.async_runtime.SubtensorModule.NetworkRegistrationAllowed[29]
await bt.async_runtime.SubtensorModule.Burn(9)
await bt.async_runtime.SubtensorModule.Burn(29)
await bt.async_runtime.SubtensorModule.BlockAtRegistration[29,5]
await bt.async_runtime.SubtensorModule.AdjustmentInterval[29]
await bt.async_runtime.SubtensorModule.LastAdjustmentBlock[29]
await bt.async_runtime.SubtensorModule.LastAdjustmentBlock(29,block=6000000)
await bt.async_runtime.SubtensorModule.ImmunityPeriod[29]
await bt.async_runtime.Commitments.LastCommitment[3, '5HTRBzc7CZvASdVZc7Fwzqnp1jXeP9z3KpoRKDZWhYSAksKq']
await bt.async_runtime.Commitments.CommitmentOf[3, '5HTRBzc7CZvASdVZc7Fwzqnp1jXeP9z3KpoRKDZWhYSAksKq']
await bt.async_runtime.Commitments.MaxFields()
await bt.async_runtime.SubtensorModule.Weights[0,1]
await bt.async_runtime.SubtensorModule.Weights(0,1)
await bt.async_runtime.SubtensorModule.Weights(0,1,block=6000000)
await bt.async_runtime.SubtensorModule.TotalHotkeyAlpha['5HmmmmhaoDbmt4KcYmAXETvfGBwFmcNgVCYSBH5JZRv6wrFf',29]
await bt.async_runtime.SubtensorModule.TotalHotkeyAlpha('5HmmmmhaoDbmt4KcYmAXETvfGBwFmcNgVCYSBH5JZRv6wrFf',29,block=6000000)
# various
await bt.async_runtime.Timestamp.Now()
await bt.async_runtime.Timestamp.Now(block=6000000)
await bt.async_runtime.Timestamp.Now(block=5900000)
(await bt.async_runtime.SubtensorModule.Alpha['5HmmmmhaoDbmt4KcYmAXETvfGBwFmcNgVCYSBH5JZRv6wrFf','5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn',29])['bits']/(1<<64)
(await bt.async_runtime.SubtensorModule.Alpha['5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn','5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn',29])['bits']/(1<<64)
await bt.async_runtime.System.Account['5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn']
await bt.async_runtime.SubtensorModule.Owner['5GuNCPiUQBVoXCJgrMeXmc7rVjwT5zUEK5RzjhGmVqbQA5me']
await bt.async_runtime.Proxy.Proxies['5Enrkn6rRLGz4tK3TWamnbKYECKpz3hffNk9RBaDC426jLgb']
# init runtime interface with particular fixed block:
await bt.async_runtime.System.Number()
await bt.async_runtime(block=6000000).System.Number()
await bt.async_runtime.System.Number()
await bt.async_runtime(block=None).System.Number()
# init runtime interface with particular network:
await bt.async_runtime(network='test').System.Number()

# exceptions, containing usage help:
await bt.async_runtime.NonExistantPallet[0]
await bt.async_runtime.System.NonExistentItem[0]
await bt.async_runtime.SubnetRegistrationRuntimeApi.no_such_call()
await bt.async_runtime.Timestamp.Now[1]
await bt.async_runtime.SubtensorModule.AdjustmentInterval[0,0]
await bt.async_runtime.System.Account()
await bt.async_runtime.System.Account[0]
await bt.async_runtime.System.Account['abc']
await bt.async_runtime.System.Account['5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn',0]

# exposed metadata, lists and __repr__ containing usage help:
bt.async_runtime
bt.async_runtime.dir()
bt.async_runtime.SubnetRegistrationRuntimeApi.dir()
bt.async_runtime.Commitments.dir()
bt.async_runtime.Commitments._metadata
bt.async_runtime.Commitments.LastCommitment._metadata
bt.async_runtime.Commitments
bt.async_runtime.Commitments.LastCommitment
bt.async_runtime.Commitments.MaxFields
'''.split('\n')]

if len(sys.argv)>1:
    block = int(sys.argv[1])
    bt.async_runtime.init(block=block)

async def tests(calls):
    # async_runtime requires explicit init call, or we'd have to await
    # bt.async_runtime.SubtensorModule before we could use its properties.
    await bt.async_runtime.init_metadata()

    # small demonstration of typical async usage with gather, matches first
    # three test cases:
    res3 = await asyncio.gather(
        bt.async_runtime.SubtensorModule.NetworkLastRegistered(),
        bt.async_runtime.SubtensorModule.NetworkLastLockCost(),
        bt.async_runtime.SubtensorModule.NetworkImmunityPeriod()
    )
    print('gather result:',res3)

    for call in calls:
        if len(call) == 0 or call[0] == '#':
            print(call)
            continue
        print(f'evaluating: {call}')
        try:
            if 'await' in call:
                exec(f'async def fn():\n return {call}')
                result = await locals()['fn']()
            else:
                result = eval(call)
            print(f'--> {result}')
        except Exception as e:
            print(f'--> exception: {e}')
            print()

asyncio.run(tests(calls))
