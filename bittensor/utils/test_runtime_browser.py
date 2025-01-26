#!/usr/bin/env python

# testing code for runtime browser

calls = [l for l in '''
# subnet registration info
bt.runtime.SubtensorModule.SubnetLimit()
bt.runtime.SubtensorModule.NetworkLastRegistered()
bt.runtime.SubtensorModule.NetworkLastLockCost()
bt.runtime.SubtensorModule.NetworkImmunityPeriod()
bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost()
bt.runtime.SubtensorModule.SubnetOwner[29]
bt.runtime.SubtensorModule.SubnetLocked[29]
bt.runtime.SubtensorModule.NetworkRegisteredAt[29]
bt.runtime.SubtensorModule.NetworkLastLockCost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29])
bt.runtime.SubtensorModule.NetworkLastLockCost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29]-1)
bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29])
bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost(block=bt.runtime.SubtensorModule.NetworkRegisteredAt[29]-1)
# hotkey registration and commitment info
bt.runtime.SubtensorModule.NetworkRegistrationAllowed[29]
bt.runtime.SubtensorModule.Burn(9)
bt.runtime.SubtensorModule.Burn(29)
bt.runtime.SubtensorModule.BlockAtRegistration[29,5]
bt.runtime.SubtensorModule.AdjustmentInterval[29]
bt.runtime.SubtensorModule.LastAdjustmentBlock[29]
bt.runtime.SubtensorModule.LastAdjustmentBlock(29,block=4785582)
bt.runtime.SubtensorModule.ImmunityPeriod[29]
bt.runtime.Commitments.LastCommitment[3, '5HTRBzc7CZvASdVZc7Fwzqnp1jXeP9z3KpoRKDZWhYSAksKq']
bt.runtime.Commitments.CommitmentOf[3, '5HTRBzc7CZvASdVZc7Fwzqnp1jXeP9z3KpoRKDZWhYSAksKq']
bt.runtime.Commitments.RateLimit()
bt.runtime.SubtensorModule.Weights[0,1]
bt.runtime.SubtensorModule.Weights(0,1)
bt.runtime.SubtensorModule.Weights(0,1,block=4000000)
bt.runtime.SubtensorModule.PendingdHotkeyEmission['5GuNCPiUQBVoXCJgrMeXmc7rVjwT5zUEK5RzjhGmVqbQA5me']
bt.runtime.SubtensorModule.PendingdHotkeyEmission['5DwJ2vu8xmwW61qP3qyYXKeb6ANpZVgggCZz15Ev2DvncHwz']
bt.runtime.SubtensorModule.PendingdHotkeyEmission('5DwJ2vu8xmwW61qP3qyYXKeb6ANpZVgggCZz15Ev2DvncHwz',block=4700000)
# various
bt.runtime.Timestamp.Now()
bt.runtime.Timestamp.Now(block=4000000)
bt.runtime.Timestamp.Now(block=3000000)
bt.runtime.SubtensorModule.Stake['5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp','5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn']
bt.runtime.System.Account['5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn']
bt.runtime.SubtensorModule.Owner['5GuNCPiUQBVoXCJgrMeXmc7rVjwT5zUEK5RzjhGmVqbQA5me']
bt.runtime.Proxy.Proxies['5Enrkn6rRLGz4tK3TWamnbKYECKpz3hffNk9RBaDC426jLgb']
# init runtime interface with particular fixed block:
bt.runtime.System.Number()
bt.runtime(block=4000000).System.Number()
bt.runtime.System.Number()
bt.runtime(block=None).System.Number()
# init runtime interface with particular network:
bt.runtime(network='test').System.Number()

# exceptions, containing usage help:
bt.runtime.NonExistantPallet[0]
bt.runtime.System.NonExistentItem[0]
bt.runtime.SubnetRegistrationRuntimeApi.no_such_call()
bt.runtime.Timestamp.Now[1]
bt.runtime.SubtensorModule.AdjustmentInterval[0,0]
bt.runtime.System.Account()
bt.runtime.System.Account[0]
bt.runtime.System.Account['abc']
bt.runtime.System.Account['5HHHHHzgLnYRvnKkHd45cRUDMHXTSwx7MjUzxBrKbY4JfZWn',0]

# exposed metadata, lists and __repr__ containing usage help:
bt.runtime
bt.runtime.dir()
bt.runtime.SubnetRegistrationRuntimeApi.dir()
bt.runtime.Commitments.dir()
bt.runtime.Commitments._metadata
bt.runtime.Commitments.LastCommitment._metadata
bt.runtime.Commitments
bt.runtime.Commitments.LastCommitment
bt.runtime.Commitments.RateLimit
'''.split('\n')]

import bittensor as bt
for call in calls:
    if len(call) == 0 or call[0] == '#':
        print(call)
        continue
    print(f'evaluating: {call}')
    try:
        result = eval(call)
        print(f'--> {result}')
    except Exception as e:
        print(f'--> exception: {e}')
        print()
