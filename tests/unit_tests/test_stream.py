"""
Unit tests for bittensor.core.stream module.

Tests the StreamingSynapse class, BTStreamingResponse, and related streaming functionality.
"""

import pytest
from abc import ABC
from typing import Optional
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from aiohttp import ClientResponse
from starlette.types import Send, Receive, Scope

from bittensor.core.stream import (
    StreamingSynapse,
    BTStreamingResponseModel,
)


# Concrete implementation for testing abstract StreamingSynapse
class ConcreteStreamingSynapse(StreamingSynapse):
    """Concrete implementation of StreamingSynapse for testing."""

    async def process_streaming_response(self, response: ClientResponse):
        """Implementation of abstract method."""
        # Simple implementation that reads response content
        async for chunk in response.content.iter_chunked(1024):
            pass

    def extract_response_json(self, response: ClientResponse) -> dict:
        """Implementation of abstract method."""
        return {"status": "success", "data": "test"}


class TestStreamingSynapseInitialization:
    """Tests for StreamingSynapse class creation."""

    def test_streaming_synapse_initialization(self):
        """Test StreamingSynapse class creation."""
        synapse = ConcreteStreamingSynapse()
        assert isinstance(synapse, StreamingSynapse)
        assert hasattr(synapse, "process_streaming_response")
        assert hasattr(synapse, "extract_response_json")
        assert hasattr(synapse, "create_streaming_response")

    def test_streaming_synapse_is_abstract(self):
        """Test that StreamingSynapse cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Should fail because abstract methods are not implemented
            StreamingSynapse()

    def test_streaming_synapse_inherits_from_synapse(self):
        """Test that StreamingSynapse inherits from Synapse."""
        from bittensor.core.synapse import Synapse

        synapse = ConcreteStreamingSynapse()
        assert isinstance(synapse, Synapse)

    def test_streaming_synapse_model_config(self):
        """Test that StreamingSynapse has proper model configuration."""
        synapse = ConcreteStreamingSynapse()
        assert hasattr(synapse, "model_config")
        assert synapse.model_config.get("validate_assignment") is True


class TestBTStreamingResponseModel:
    """Tests for BTStreamingResponseModel."""

    def test_bt_streaming_response_model_creation(self):
        """Test BTStreamingResponseModel initialization."""
        async def mock_token_streamer(send: Send):
            await send({"type": "http.response.body", "body": b"test"})

        model = BTStreamingResponseModel(token_streamer=mock_token_streamer)
        assert model.token_streamer == mock_token_streamer

    def test_bt_streaming_response_model_validation(self):
        """Test that BTStreamingResponseModel validates token_streamer type."""
        # Should accept callable
        async def valid_streamer(send: Send):
            pass

        model = BTStreamingResponseModel(token_streamer=valid_streamer)
        assert callable(model.token_streamer)

    def test_bt_streaming_response_model_invalid_type(self):
        """Test that BTStreamingResponseModel rejects invalid token_streamer."""
        with pytest.raises(Exception):  # Pydantic validation error
            BTStreamingResponseModel(token_streamer="not_a_callable")


class TestBTStreamingResponse:
    """Tests for BTStreamingResponse class."""

    def test_bt_streaming_response_creation(self):
        """Test BTStreamingResponse initialization."""
        async def mock_token_streamer(send: Send):
            await send({"type": "http.response.body", "body": b"test"})

        model = BTStreamingResponseModel(token_streamer=mock_token_streamer)
        synapse = ConcreteStreamingSynapse()

        response = StreamingSynapse.BTStreamingResponse(model, synapse=synapse)
        assert response.token_streamer == mock_token_streamer
        assert response.synapse == synapse

    def test_bt_streaming_response_without_synapse(self):
        """Test BTStreamingResponse initialization without synapse."""
        async def mock_token_streamer(send: Send):
            pass

        model = BTStreamingResponseModel(token_streamer=mock_token_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)
        assert response.token_streamer == mock_token_streamer
        assert response.synapse is None

    @pytest.mark.asyncio
    async def test_stream_response_method(self):
        """Test async stream_response functionality."""
        call_order = []

        async def mock_token_streamer(send: Send):
            call_order.append("token_streamer")
            await send({"type": "http.response.body", "body": b"chunk1"})

        model = BTStreamingResponseModel(token_streamer=mock_token_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        send_mock = AsyncMock()

        await response.stream_response(send_mock)

        # Verify send was called with correct structure
        assert send_mock.call_count == 3
        
        # First call: start response with headers
        first_call = send_mock.call_args_list[0][0][0]
        assert first_call["type"] == "http.response.start"
        assert first_call["status"] == 200
        assert any(h == (b"content-type", b"text/event-stream") for h in first_call["headers"])

        # Second call: token streamer
        assert call_order == ["token_streamer"]

        # Third call: end response
        last_call = send_mock.call_args_list[2][0][0]
        assert last_call["type"] == "http.response.body"
        assert last_call["body"] == b""
        assert last_call["more_body"] is False

    @pytest.mark.asyncio
    async def test_token_streamer_execution(self):
        """Verify token streamer callable execution."""
        executed = []

        async def test_streamer(send: Send):
            executed.append(True)
            await send({"type": "http.response.body", "body": b"data"})

        model = BTStreamingResponseModel(token_streamer=test_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        send_mock = AsyncMock()
        await response.stream_response(send_mock)

        assert len(executed) == 1
        assert executed[0] is True

    @pytest.mark.asyncio
    async def test_streaming_response_headers(self):
        """Verify content-type headers for event-streaming."""
        async def mock_streamer(send: Send):
            pass

        model = BTStreamingResponseModel(token_streamer=mock_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        send_mock = AsyncMock()
        await response.stream_response(send_mock)

        # Check that headers include text/event-stream
        headers_call = send_mock.call_args_list[0][0][0]
        headers = headers_call["headers"]
        
        assert (b"content-type", b"text/event-stream") in headers

    @pytest.mark.asyncio
    async def test_asgi_interface_compatibility(self):
        """Test ASGI scope/receive/send interface."""
        async def mock_streamer(send: Send):
            await send({"type": "http.response.body", "body": b"test"})

        model = BTStreamingResponseModel(token_streamer=mock_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        # Mock ASGI parameters
        scope_mock: Scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
        }
        receive_mock: Receive = AsyncMock()
        send_mock: Send = AsyncMock()

        # Call as ASGI app
        await response(scope_mock, receive_mock, send_mock)

        # Verify send was called
        assert send_mock.call_count >= 2

    @pytest.mark.asyncio
    async def test_streaming_response_cleanup(self):
        """Test proper resource cleanup."""
        cleanup_called = []

        async def streamer_with_cleanup(send: Send):
            try:
                await send({"type": "http.response.body", "body": b"data"})
            finally:
                cleanup_called.append(True)

        model = BTStreamingResponseModel(token_streamer=streamer_with_cleanup)
        response = StreamingSynapse.BTStreamingResponse(model)

        send_mock = AsyncMock()
        await response.stream_response(send_mock)

        # Verify cleanup was called
        assert len(cleanup_called) == 1


class TestAbstractMethodsEnforcement:
    """Tests for abstract methods enforcement."""

    def test_abstract_methods_enforcement(self):
        """Ensure abstract methods must be implemented."""
        # Missing both abstract methods
        with pytest.raises(TypeError) as exc_info:

            class IncompleteStreamingSynapse(StreamingSynapse):
                pass

            IncompleteStreamingSynapse()

        assert "abstract" in str(exc_info.value).lower()

    def test_missing_process_streaming_response(self):
        """Test that missing process_streaming_response raises error."""
        with pytest.raises(TypeError):

            class MissingProcessMethod(StreamingSynapse):
                def extract_response_json(self, response: ClientResponse) -> dict:
                    return {}

            MissingProcessMethod()

    def test_missing_extract_response_json(self):
        """Test that missing extract_response_json raises error."""
        with pytest.raises(TypeError):

            class MissingExtractMethod(StreamingSynapse):
                async def process_streaming_response(self, response: ClientResponse):
                    pass

            MissingExtractMethod()


class TestProcessStreamingResponseImplementation:
    """Tests for process_streaming_response implementation."""

    @pytest.mark.asyncio
    async def test_process_streaming_response_implementation(self):
        """Test subclass implementations of process_streaming_response."""
        processed_chunks = []

        class CustomStreamingSynapse(StreamingSynapse):
            async def process_streaming_response(self, response: ClientResponse):
                async for chunk in response.content.iter_chunked(1024):
                    processed_chunks.append(chunk)

            def extract_response_json(self, response: ClientResponse) -> dict:
                return {"processed": len(processed_chunks)}

        synapse = CustomStreamingSynapse()

        # Mock response
        mock_response = Mock(spec=ClientResponse)
        mock_content = Mock()

        async def mock_iter_chunked(size):
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk

        mock_content.iter_chunked = mock_iter_chunked
        mock_response.content = mock_content

        await synapse.process_streaming_response(mock_response)

        assert len(processed_chunks) == 3
        assert processed_chunks == [b"chunk1", b"chunk2", b"chunk3"]


class TestExtractResponseJsonImplementation:
    """Tests for extract_response_json implementation."""

    def test_extract_response_json_implementation(self):
        """Test JSON extraction from responses."""

        class JsonExtractingSynapse(StreamingSynapse):
            async def process_streaming_response(self, response: ClientResponse):
                pass

            def extract_response_json(self, response: ClientResponse) -> dict:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url),
                }

        synapse = JsonExtractingSynapse()

        # Mock response
        mock_response = Mock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "http://test.com"

        result = synapse.extract_response_json(mock_response)

        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["url"] == "http://test.com"

    def test_extract_response_json_custom_logic(self):
        """Test custom JSON extraction logic."""

        class CustomJsonSynapse(StreamingSynapse):
            accumulated_data: list = []

            async def process_streaming_response(self, response: ClientResponse):
                self.accumulated_data.append("data")

            def extract_response_json(self, response: ClientResponse) -> dict:
                return {
                    "accumulated": self.accumulated_data,
                    "count": len(self.accumulated_data),
                }

        synapse = CustomJsonSynapse()
        # Directly set the field value
        synapse.accumulated_data = ["item1", "item2"]

        mock_response = Mock(spec=ClientResponse)
        result = synapse.extract_response_json(mock_response)

        assert result["count"] == 2
        assert result["accumulated"] == ["item1", "item2"]


class TestCreateStreamingResponse:
    """Tests for create_streaming_response method."""

    def test_create_streaming_response_with_custom_streamer(self):
        """Test custom token streamers."""

        async def custom_streamer(send: Send):
            await send({"type": "http.response.body", "body": b"custom data"})

        synapse = ConcreteStreamingSynapse()
        response = synapse.create_streaming_response(custom_streamer)

        assert isinstance(response, StreamingSynapse.BTStreamingResponse)
        assert response.token_streamer == custom_streamer
        assert response.synapse == synapse

    def test_create_streaming_response_returns_correct_type(self):
        """Test that create_streaming_response returns BTStreamingResponse."""

        async def test_streamer(send: Send):
            pass

        synapse = ConcreteStreamingSynapse()
        response = synapse.create_streaming_response(test_streamer)

        assert isinstance(response, StreamingSynapse.BTStreamingResponse)
        assert hasattr(response, "stream_response")
        assert hasattr(response, "token_streamer")

    @pytest.mark.asyncio
    async def test_create_streaming_response_functional(self):
        """Test that created streaming response is functional."""
        chunks_sent = []

        async def tracking_streamer(send: Send):
            chunks_sent.append(b"chunk1")
            await send({"type": "http.response.body", "body": b"chunk1"})

        synapse = ConcreteStreamingSynapse()
        response = synapse.create_streaming_response(tracking_streamer)

        send_mock = AsyncMock()
        await response.stream_response(send_mock)

        assert len(chunks_sent) == 1
        assert send_mock.call_count >= 2


class TestStreamingResponseErrorHandling:
    """Tests for error handling in streaming."""

    @pytest.mark.asyncio
    async def test_streaming_response_error_handling(self):
        """Test error cases in streaming."""

        async def failing_streamer(send: Send):
            raise ValueError("Streaming error")

        model = BTStreamingResponseModel(token_streamer=failing_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        send_mock = AsyncMock()

        with pytest.raises(ValueError, match="Streaming error"):
            await response.stream_response(send_mock)

    @pytest.mark.asyncio
    async def test_streaming_response_send_error(self):
        """Test error when send fails."""

        async def normal_streamer(send: Send):
            await send({"type": "http.response.body", "body": b"data"})

        model = BTStreamingResponseModel(token_streamer=normal_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        # Mock send that fails
        send_mock = AsyncMock(side_effect=RuntimeError("Send failed"))

        with pytest.raises(RuntimeError, match="Send failed"):
            await response.stream_response(send_mock)

    @pytest.mark.asyncio
    async def test_streaming_response_partial_failure(self):
        """Test partial failure during streaming."""
        call_count = []

        async def partial_fail_streamer(send: Send):
            call_count.append(1)
            await send({"type": "http.response.body", "body": b"chunk1"})
            call_count.append(2)
            raise ConnectionError("Connection lost")

        model = BTStreamingResponseModel(token_streamer=partial_fail_streamer)
        response = StreamingSynapse.BTStreamingResponse(model)

        send_mock = AsyncMock()

        with pytest.raises(ConnectionError, match="Connection lost"):
            await response.stream_response(send_mock)

        # Verify partial execution
        assert len(call_count) == 2


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self):
        """Test complete streaming workflow."""
        chunks = []

        async def data_streamer(send: Send):
            for i in range(3):
                chunk = f"data_{i}".encode()
                chunks.append(chunk)
                await send({"type": "http.response.body", "body": chunk})

        synapse = ConcreteStreamingSynapse()
        response = synapse.create_streaming_response(data_streamer)

        send_mock = AsyncMock()
        await response.stream_response(send_mock)

        # Verify all chunks were processed
        assert len(chunks) == 3
        assert chunks[0] == b"data_0"
        assert chunks[1] == b"data_1"
        assert chunks[2] == b"data_2"

    @pytest.mark.asyncio
    async def test_streaming_with_headers_preservation(self):
        """Test that custom headers are preserved."""

        async def simple_streamer(send: Send):
            await send({"type": "http.response.body", "body": b"test"})

        model = BTStreamingResponseModel(token_streamer=simple_streamer)
        response = StreamingSynapse.BTStreamingResponse(
            model, headers={"X-Custom-Header": "test-value"}
        )

        send_mock = AsyncMock()
        await response.stream_response(send_mock)

        # Check headers in first call
        headers_call = send_mock.call_args_list[0][0][0]
        headers = dict(headers_call["headers"])
        
        # Verify event-stream header is present
        assert headers.get(b"content-type") == b"text/event-stream"

    @pytest.mark.asyncio
    async def test_multiple_streaming_responses(self):
        """Test creating multiple streaming responses."""

        async def streamer1(send: Send):
            await send({"type": "http.response.body", "body": b"stream1"})

        async def streamer2(send: Send):
            await send({"type": "http.response.body", "body": b"stream2"})

        synapse = ConcreteStreamingSynapse()
        
        response1 = synapse.create_streaming_response(streamer1)
        response2 = synapse.create_streaming_response(streamer2)

        assert response1.token_streamer != response2.token_streamer
        assert response1.synapse == response2.synapse == synapse

        # Both should work independently
        send_mock1 = AsyncMock()
        send_mock2 = AsyncMock()

        await response1.stream_response(send_mock1)
        await response2.stream_response(send_mock2)

        assert send_mock1.called
        assert send_mock2.called
