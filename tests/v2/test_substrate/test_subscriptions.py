import pytest
from bittensor.v2.utils.async_substrate import AsyncSubstrateInterface
import settings


@pytest.fixture
def substrate():
    return AsyncSubstrateInterface(
        url=settings.POLKADOT_NODE_URL,
    )


@pytest.mark.asyncio
async def test_query_subscription(substrate):
    async def subscription_handler(obj, subscription_id):
        return {"subscription_id": subscription_id}, True

    async with substrate:
        result = await substrate.query(
            "System", "Events", [], subscription_handler=subscription_handler
        )

    assert result["subscription_id"] is not None


if __name__ == "__main__":
    pytest.main()
