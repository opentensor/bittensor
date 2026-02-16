import os
import urllib
import pytest
import requests
import unittest.mock as mock
from bittensor import utils
from bittensor.utils.networking import (
    _validate_ip_response,
    ExternalIPNotFound,
    get_external_ip,
    get_formatted_ws_endpoint_url,
    int_to_ip,
    ip_to_int,
    ip_version,
    ip__str__,
)
from unittest.mock import MagicMock
import netaddr


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


# Tests for ip_version
class TestIpVersion:
    """Tests for the ip_version function."""

    def test_ipv4_version(self):
        """Test that IPv4 addresses return version 4."""
        assert ip_version("192.168.1.1") == 4
        assert ip_version("10.0.0.1") == 4
        assert ip_version("0.0.0.0") == 4
        assert ip_version("255.255.255.255") == 4

    def test_ipv6_version(self):
        """Test that IPv6 addresses return version 6."""
        assert ip_version("::1") == 6
        assert ip_version("fe80::1") == 6
        assert ip_version("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff") == 6

    def test_invalid_ip_raises(self):
        """Test that invalid IP strings raise an error."""
        with pytest.raises(netaddr.AddrFormatError):
            ip_version("not_an_ip")
        with pytest.raises(netaddr.AddrFormatError):
            ip_version("")
        with pytest.raises(netaddr.AddrFormatError):
            ip_version("999.999.999.999")


# Tests for ip_to_int with invalid input
class TestIpToIntValidation:
    """Tests for ip_to_int input validation."""

    def test_invalid_ip_string_raises(self):
        """Test that invalid IP strings raise netaddr.AddrFormatError."""
        with pytest.raises(netaddr.AddrFormatError):
            ip_to_int("not_an_ip")

    def test_empty_string_raises(self):
        """Test that empty string raises netaddr.AddrFormatError."""
        with pytest.raises(netaddr.AddrFormatError):
            ip_to_int("")

    def test_partial_ip_raises(self):
        """Test that partial IP addresses raise an error."""
        with pytest.raises(netaddr.AddrFormatError):
            ip_to_int("192.168.1")


# Tests for ip__str__ return type
class TestIpStrFormat:
    """Tests for the ip__str__ function."""

    def test_return_type_is_str(self):
        """Test that ip__str__ always returns a string."""
        result = ip__str__(4, "127.0.0.1", 8080)
        assert isinstance(result, str)

    def test_ipv6_format(self):
        """Test IPv6 formatted string."""
        result = ip__str__(6, "::1", 9944)
        assert result == "/ipv6/::1:9944"

    def test_port_zero(self):
        """Test with port 0."""
        result = ip__str__(4, "10.0.0.1", 0)
        assert result == "/ipv4/10.0.0.1:0"

    def test_high_port(self):
        """Test with high port number."""
        result = ip__str__(4, "10.0.0.1", 65535)
        assert result == "/ipv4/10.0.0.1:65535"


# Tests for _validate_ip_response
class TestValidateIpResponse:
    """Tests for the _validate_ip_response helper function."""

    def test_valid_ipv4(self):
        """Test validation with a valid IPv4 address."""
        assert _validate_ip_response("192.168.1.1") == "192.168.1.1"

    def test_valid_ipv4_with_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        assert _validate_ip_response("  10.0.0.1  ") == "10.0.0.1"
        assert _validate_ip_response("\n203.0.113.5\n") == "203.0.113.5"

    def test_valid_ipv6(self):
        """Test validation with a valid IPv6 address."""
        assert _validate_ip_response("::1") == "::1"
        assert _validate_ip_response("2001:db8::1") == "2001:db8::1"

    def test_invalid_ip_raises(self):
        """Test that invalid IP strings raise an error."""
        with pytest.raises(Exception):
            _validate_ip_response("not_an_ip")

    def test_empty_string_raises(self):
        """Test that empty string raises an error."""
        with pytest.raises(Exception):
            _validate_ip_response("")

    def test_html_response_raises(self):
        """Test that HTML responses (common failure mode) raise an error."""
        with pytest.raises(Exception):
            _validate_ip_response("<html>Error</html>")


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

    mocked_requests_get.assert_called_once_with(
        "https://checkip.amazonaws.com", timeout=5
    )


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

    mocked_requests_get.assert_called_once_with(
        "https://checkip.amazonaws.com", timeout=5
    )


def test_get_external_ip_all_broken_raises(mocker):
    """Test that ExternalIPNotFound is raised when all services return invalid data."""
    mocker.patch.object(
        requests,
        "get",
        side_effect=requests.exceptions.ConnectionError("no network"),
    )
    mocker.patch.object(
        os, "popen", side_effect=OSError("no curl")
    )
    mocker.patch(
        "bittensor.utils.networking.urllib_request.urlopen",
        side_effect=OSError("no urllib"),
    )
    with pytest.raises(ExternalIPNotFound):
        get_external_ip()


class TestGetExternalIpFallthrough:
    """Tests for get_external_ip fallthrough behavior when services fail."""

    def test_aws_connection_error_falls_through(self, mocker):
        """Test that a ConnectionError from AWS falls through to the next service.

        This was a bug where ExternalIPNotFound was caught instead of Exception,
        meaning real network errors would crash instead of falling through.
        """
        # requests.get is called for AWS (1st) and Wikipedia (2nd)
        mocker.patch.object(
            requests,
            "get",
            side_effect=[
                requests.exceptions.ConnectionError("AWS down"),
                mock.Mock(headers={"X-Client-IP": "203.0.113.5"}),
            ],
        )
        # Mock os.popen to fail (3 curl calls: ifconfig.me, ipinfo.io, dnsomatic)
        mocker.patch.object(
            os, "popen", side_effect=OSError("curl not found")
        )
        # Mock urllib to fail
        mocker.patch(
            "bittensor.utils.networking.urllib_request.urlopen",
            side_effect=OSError("network error"),
        )
        result = get_external_ip()
        assert result == "203.0.113.5"

    def test_all_services_fail_raises_external_ip_not_found(self, mocker):
        """Test that ExternalIPNotFound is raised when all services fail."""
        mocker.patch.object(
            requests,
            "get",
            side_effect=requests.exceptions.ConnectionError("no network"),
        )
        mocker.patch.object(
            os, "popen", side_effect=OSError("curl not found")
        )
        mocker.patch(
            "bittensor.utils.networking.urllib_request.urlopen",
            side_effect=OSError("network error"),
        )
        with pytest.raises(ExternalIPNotFound):
            get_external_ip()

    def test_aws_returns_invalid_ip_falls_through(self, mocker):
        """Test that an invalid IP response from AWS falls through."""
        mocker.patch.object(
            requests,
            "get",
            side_effect=[
                mock.Mock(text="<html>Error Page</html>"),  # AWS returns HTML error
                mock.Mock(headers={"X-Client-IP": "198.51.100.1"}),  # Wikipedia works
            ],
        )
        mocker.patch.object(
            os, "popen", side_effect=OSError("curl not found")
        )
        mocker.patch(
            "bittensor.utils.networking.urllib_request.urlopen",
            side_effect=OSError("network error"),
        )
        result = get_external_ip()
        assert result == "198.51.100.1"

    def test_timeout_error_falls_through(self, mocker):
        """Test that a timeout from the first service falls through."""
        mocker.patch.object(
            requests,
            "get",
            side_effect=[
                requests.exceptions.Timeout("timed out"),
                mock.Mock(headers={"X-Client-IP": "198.51.100.2"}),
            ],
        )
        mocker.patch.object(
            os, "popen", side_effect=OSError("curl not found")
        )
        mocker.patch(
            "bittensor.utils.networking.urllib_request.urlopen",
            side_effect=OSError("network error"),
        )
        result = get_external_ip()
        assert result == "198.51.100.2"


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


class TestGetFormattedWsEndpointUrlEdgeCases:
    """Edge case tests for get_formatted_ws_endpoint_url."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert get_formatted_ws_endpoint_url(None) is None

    def test_empty_string_prepends_ws(self):
        """Test that empty string gets ws:// prepended."""
        assert get_formatted_ws_endpoint_url("") == "ws://"

    def test_preserves_path(self):
        """Test that paths are preserved."""
        result = get_formatted_ws_endpoint_url("ws://host:9944/ws")
        assert result == "ws://host:9944/ws"

    def test_wss_not_modified(self):
        """Test that wss:// URLs are not modified."""
        result = get_formatted_ws_endpoint_url("wss://secure.host:443")
        assert result == "wss://secure.host:443"
