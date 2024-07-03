import asyncio

from bittensor.v2.mock.subtensor_mock import (
    __GLOBAL_MOCK_STATE__, AxonServeCallParams, PrometheusServeCallParams, BlockNumber,
    InfoDict, AxonInfoDict, PrometheusInfoDict, MockSubtensorValue, MockMapResult, 
    MockSystemState, MockSubtensorState, MockChainState, MockSubtensor as ms
)

__GLOBAL_MOCK_STATE__ = __GLOBAL_MOCK_STATE__
AxonServeCallParams = AxonServeCallParams
PrometheusServeCallParams = PrometheusServeCallParams
BlockNumber = BlockNumber
InfoDict = InfoDict
AxonInfoDict = AxonInfoDict
PrometheusInfoDict = PrometheusInfoDict
MockSubtensorValue = MockSubtensorValue
MockMapResult = MockMapResult
MockSystemState = MockSystemState
MockSubtensorState = MockSubtensorState
MockChainState = MockChainState


class MockSubtensor:
    def __init__(self, *args, **kwargs):
        self._async_instance = ms(*args, **kwargs)

    def __getattr__(self, item):
        attr = getattr(self._async_instance, item)
        if asyncio.iscoroutinefunction(attr):
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(attr(*args, **kwargs))

            return sync_wrapper
        return attr
