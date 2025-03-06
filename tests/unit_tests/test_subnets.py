import pytest

from bittensor.utils import subnets


class MySubnetsAPI(subnets.SubnetsAPI):
    """Example of user class inherited from SubnetsAPI."""

    def prepare_synapse(self, *args, **kwargs):
        """Prepare the synapse-specific payload."""

    def process_responses(self, responses):
        """Process the responses from the network."""
        return responses


def test_instance_creation(mocker):
    """Test the creation of a MySubnetsAPI instance."""
    # Prep
    mocked_dendrite = mocker.patch.object(subnets, "Dendrite")
    fake_wallet = mocker.MagicMock()

    # Call
    instance = MySubnetsAPI(fake_wallet)

    # Asserts
    assert isinstance(instance, subnets.SubnetsAPI)
    mocked_dendrite.assert_called_once_with(wallet=fake_wallet)
    assert instance.dendrite == mocked_dendrite.return_value
    assert instance.wallet == fake_wallet


@pytest.mark.asyncio
async def test_query_api(mocker):
    """Test querying the MySubnetsAPI instance asynchronously."""
    # Prep
    mocked_async_dendrite = mocker.AsyncMock()
    mocked_dendrite = mocker.patch.object(
        subnets, "Dendrite", return_value=mocked_async_dendrite
    )

    fake_wallet = mocker.MagicMock()
    fake_axon = mocker.MagicMock()

    mocked_synapse = mocker.MagicMock()
    mocked_synapse.return_value.name = "test synapse"
    mocked_prepare_synapse = mocker.patch.object(
        MySubnetsAPI, "prepare_synapse", return_value=mocked_synapse
    )

    # Call
    instance = MySubnetsAPI(fake_wallet)
    result = await instance.query_api(fake_axon, **{"key": "val"})

    # Asserts
    mocked_prepare_synapse.assert_called_once_with(key="val")
    mocked_dendrite.assert_called_once_with(wallet=fake_wallet)
    assert result == mocked_async_dendrite.return_value


@pytest.mark.asyncio
async def test_test_instance_call(mocker):
    """Test the MySubnetsAPI instance call with asynchronous handling."""
    # Prep
    mocked_async_dendrite = mocker.AsyncMock()
    mocked_dendrite = mocker.patch.object(
        subnets, "Dendrite", return_value=mocked_async_dendrite
    )
    mocked_query_api = mocker.patch.object(
        MySubnetsAPI, "query_api", new=mocker.AsyncMock()
    )
    fake_wallet = mocker.MagicMock()
    fake_axon = mocker.MagicMock()

    # Call
    instance = MySubnetsAPI(fake_wallet)
    await instance(fake_axon)

    # Asserts
    mocked_dendrite.assert_called_once_with(wallet=fake_wallet)
    mocked_query_api.assert_called_once_with(fake_axon)
