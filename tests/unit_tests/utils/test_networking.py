import os
import urllib
import pytest
import requests
import unittest.mock as mock
from bittensor import utils
from unittest.mock import MagicMock


# Test conversion functions for IPv4
def test_int_to_ip_zero():
    """Test converting integer to IPv4 address for 0."""
    assert utils.networking.int_to_ip(0) == "0.0.0.0"
    assert utils.networking.ip_to_int("0.0.0.0") == 0
    assert utils.networking.ip__str__(4, "0.0.0.0", 8888) == "/ipv4/0.0.0.0:8888"


def test_int_to_ip_range():
    """Test converting integer to IPv4 addresses in a range."""
    for i in range(10):
        assert utils.networking.int_to_ip(i) == f"0.0.0.{i}"
        assert utils.networking.ip_to_int(f"0.0.0.{i}") == i
        assert (
            utils.networking.ip__str__(4, f"0.0.0.{i}", 8888) == f"/ipv4/0.0.0.{i}:8888"
        )


def test_int_to_ip4_max():
    """Test converting integer to maximum IPv4 address."""
    assert utils.networking.int_to_ip(4294967295) == "255.255.255.255"
    assert utils.networking.ip_to_int("255.255.255.255") == 4294967295
    assert (
        utils.networking.ip__str__(4, "255.255.255.255", 8888)
        == "/ipv4/255.255.255.255:8888"
    )


# Test conversion functions for IPv6
def test_int_to_ip6_zero():
    """Test converting integer to IPv6 address for 0."""
    assert utils.networking.int_to_ip(4294967296) == "::1:0:0"
    assert utils.networking.ip_to_int("::1:0:0") == 4294967296
    assert utils.networking.ip__str__(6, "::1:0:0", 8888) == "/ipv6/::1:0:0:8888"


def test_int_to_ip6_range():
    """Test converting integer to IPv6 addresses in a range."""
    for i in range(10):
        assert utils.networking.int_to_ip(4294967296 + i) == f"::1:0:{i}"
        assert utils.networking.ip_to_int(f"::1:0:{i}") == 4294967296 + i
        assert (
            utils.networking.ip__str__(6, f"::1:0:{i}", 8888) == f"/ipv6/::1:0:{i}:8888"
        )


def test_int_to_ip6_max():
    """Test converting integer to maximum IPv6 address."""
    max_val = 340282366920938463463374607431768211455
    assert (
        utils.networking.int_to_ip(max_val) == "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"
    )
    assert (
        utils.networking.ip_to_int("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff") == max_val
    )
    assert (
        utils.networking.ip__str__(6, "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", 8888)
        == "/ipv6/ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff:8888"
    )


def test_int_to_ip6_overflow():
    """Test handling overflow when converting integer to IPv6 address."""
    overflow = 340282366920938463463374607431768211455 + 1
    with pytest.raises(Exception):
        utils.networking.int_to_ip(overflow)


def test_int_to_ip6_underflow():
    """Test handling underflow when converting integer to IPv6 address."""
    underflow = -1
    with pytest.raises(Exception):
        utils.networking.int_to_ip(underflow)


# Test getting external IP address
def test_get_external_ip(mocker):
    """Test getting the external IP address."""
    mocked_requests_get = mock.Mock(
        return_value=mock.Mock(
            **{
                "text": "192.168.1.1",
            },
        ),
    )

    mocker.patch.object(
        requests,
        "get",
        mocked_requests_get,
    )

    assert utils.networking.get_external_ip() == "192.168.1.1"

    mocked_requests_get.assert_called_once_with("https://checkip.amazonaws.com")


def test_get_external_ip_os_broken(mocker):
    """Test getting the external IP address when os.popen is broken."""
    mocked_requests_get = mock.Mock(
        return_value=mock.Mock(
            **{
                "text": "192.168.1.1",
            },
        ),
    )

    mocker.patch.object(
        requests,
        "get",
        mocked_requests_get,
    )

    class FakeReadline:
        def readline(self):
            return 1

    def mock_call():
        return FakeReadline()

    with mock.patch.object(os, "popen", new=mock_call):
        assert utils.networking.get_external_ip() == "192.168.1.1"

    mocked_requests_get.assert_called_once_with("https://checkip.amazonaws.com")


def test_get_external_ip_os_request_urllib_broken():
    """Test getting the external IP address when os.popen and requests.get/urllib.request are broken."""

    class FakeReadline:
        def readline(self):
            return 1

    def mock_call():
        return FakeReadline()

    class FakeResponse:
        def text(self):
            return 1

    def mock_call_two():
        return FakeResponse()

    class FakeRequest:
        def urlopen(self):
            return 1

    with mock.patch.object(os, "popen", new=mock_call):
        with mock.patch.object(requests, "get", new=mock_call_two):
            urllib.request = MagicMock(return_value=FakeRequest())
            with pytest.raises(Exception):
                assert utils.networking.get_external_ip()


# Test formatting WebSocket endpoint URL
@pytest.mark.parametrize(
    "url, expected",
    [
        ("wss://exampleendpoint:9944", "wss://exampleendpoint:9944"),
        ("ws://exampleendpoint:9944", "ws://exampleendpoint:9944"),
        (
            "exampleendpoint:9944",
            "ws://exampleendpoint:9944",
        ),  # should add ws:// not wss://
        (
            "ws://exampleendpoint",
            "ws://exampleendpoint",
        ),  # should not add port if not specified
        (
            "wss://exampleendpoint",
            "wss://exampleendpoint",
        ),  # should not add port if not specified
        (
            "exampleendpoint",
            "ws://exampleendpoint",
        ),  # should not add port if not specified
        (
            "exampleendpointwithws://:9944",
            "ws://exampleendpointwithws://:9944",
        ),  # should only care about the front
        (
            "exampleendpointwithwss://:9944",
            "ws://exampleendpointwithwss://:9944",
        ),  # should only care about the front
    ],
)
def test_format(url: str, expected: str):
    """Test formatting WebSocket endpoint URL."""
    assert utils.networking.get_formatted_ws_endpoint_url(url) == expected
