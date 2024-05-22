import pytest
from aioresponses import aioresponses


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m
