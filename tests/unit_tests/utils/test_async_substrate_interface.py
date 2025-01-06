import pytest
from websockets.exceptions import InvalidURI

from bittensor.utils import substrate_interface


@pytest.mark.asyncio
async def test_invalid_url_raises_exception():
    """Test that invalid URI raises an InvalidURI exception."""
    with pytest.raises(InvalidURI):
        substrate_interface.AsyncSubstrateInterface("non_existent_entry_point")
