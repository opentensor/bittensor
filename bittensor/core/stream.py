from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Optional

from aiohttp import ClientResponse
from pydantic import ConfigDict, BaseModel
from starlette.responses import StreamingResponse as _StreamingResponse
from starlette.types import Send, Receive, Scope

from .synapse import Synapse


class BTStreamingResponseModel(BaseModel):
    """
    :func:`BTStreamingResponseModel` is a Pydantic model that encapsulates the token streamer callable for Pydantic
    validation.
    It is used within the :func:`StreamingSynapse` class to create a :func:`BTStreamingResponse` object, which is
    responsible for handling the streaming of tokens.

    The token streamer is a callable that takes a send function and returns an awaitable. It is responsible for generating
    the content of the streaming response, typically by processing tokens and sending them to the client.

    This model ensures that the token streamer conforms to the expected signature and provides a clear interface for
    passing the token streamer to the BTStreamingResponse class.

    Attributes:
        token_streamer: Callable[[Send], Awaitable[None]] The token streamer callable, which takes a send function
            (provided by the ASGI server) and returns an awaitable. It is responsible for generating the content of the
            streaming response.
    """

    token_streamer: Callable[[Send], Awaitable[None]]


class StreamingSynapse(Synapse, ABC):
    """
    The :func:`StreamingSynapse` class is designed to be subclassed for handling streaming responses in the Bittensor network.
    It provides abstract methods that must be implemented by the subclass to deserialize, process streaming responses,
    and extract JSON data. It also includes a method to create a streaming response object.
    """

    model_config = ConfigDict(validate_assignment=True)

    class BTStreamingResponse(_StreamingResponse):
        """
        :func:`BTStreamingResponse` is a specialized subclass of the Starlette StreamingResponse designed to handle the
        streaming of tokens within the Bittensor network. It is used internally by the StreamingSynapse class to manage
        the response streaming process, including sending headers and calling the token streamer provided by the subclass.

        This class is not intended to be directly instantiated or modified by developers subclassing StreamingSynapse.
        Instead, it is used by the :func:`create_streaming_response` method to create a response object based on the
        token streamer provided by the subclass.
        """

        def __init__(
            self,
            model: "BTStreamingResponseModel",
            *,
            synapse: "Optional[StreamingSynapse]" = None,
            **kwargs,
        ):
            """
            Initializes the BTStreamingResponse with the given token streamer model.

            Parameters:
                model: A BTStreamingResponseModel instance containing the token streamer callable, which is responsible
                    for generating the content of the response.
                synapse: The response Synapse to be used to update the response headers etc.
                **kwargs: Additional keyword arguments passed to the parent StreamingResponse class.
            """
            super().__init__(content=iter(()), **kwargs)
            self.token_streamer = model.token_streamer
            self.synapse = synapse

        async def stream_response(self, send: "Send"):
            """
            Asynchronously streams the response by sending headers and calling the token streamer.

            This method is responsible for initiating the response by sending the appropriate headers, including the
            content type for event-streaming. It then calls the token streamer to generate the content and sends the
            response body to the client.

            Parameters:
                send: A callable to send the response, provided by the ASGI server.
            """
            headers = [(b"content-type", b"text/event-stream")] + self.raw_headers

            await send(
                {"type": "http.response.start", "status": 200, "headers": headers}
            )

            await self.token_streamer(send)

            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def __call__(self, scope: "Scope", receive: "Receive", send: "Send"):
            """
            Asynchronously calls the :func:`stream_response method`, allowing the :func:`BTStreamingResponse` object to
            be used as an ASGI application.

            This method is part of the ASGI interface and is called by the ASGI server to handle the request and send
            the response. It delegates to the :func:`stream_response` method to perform the actual streaming process.

            Parameters:
                scope: The scope of the request, containing information about the client, server, and request itself.
                receive: A callable to receive the request, provided by the ASGI server.
                send: A callable to send the response, provided by the ASGI server.
            """
            await self.stream_response(send)

    @abstractmethod
    async def process_streaming_response(self, response: "ClientResponse"):
        """
        Abstract method that must be implemented by the subclass.
        This method should provide logic to handle the streaming response, such as parsing and accumulating data. It is
        called as the response is being streamed from the network, and should be implemented to handle the specific
        streaming data format and requirements of the subclass.

        Parameters:
            The response object to be processed, typically containing chunks of data.
        """
        ...

    @abstractmethod
    def extract_response_json(self, response: "ClientResponse") -> dict:
        """
        Abstract method that must be implemented by the subclass.
        This method should provide logic to extract JSON data from the response, including headers and content.
        It is called after the response has been processed and is responsible for retrieving structured data that can be
        used by the application.

        Parameters:
            The response object from which to extract JSON data.
        """

    def create_streaming_response(
        self, token_streamer: Callable[[Send], Awaitable[None]]
    ) -> "BTStreamingResponse":
        """
        Creates a streaming response using the provided token streamer.
        This method can be used by the subclass to create a response object that can be sent back to the client.
        The token streamer should be implemented to generate the content of the response according to the specific
        requirements of the subclass.

        Parameters:
            token_streamer: A callable that takes a send function and returns an awaitable. It's responsible for
                generating the content of the response.

        Returns:
            The streaming response object, ready to be sent to the client.
        """
        model_instance = BTStreamingResponseModel(token_streamer=token_streamer)

        return self.BTStreamingResponse(model_instance, synapse=self)
