from starlette.responses import StreamingResponse as _StreamingResponse
from starlette.types import Send
from functools import partial
from typing import Callable, Awaitable
from fastapi import FastAPI
import uvicorn
import asyncio

# Fake model and tokenizer
def tokenizer(text):
    for char in text:
        yield ord(char)

def model(ids):
    for id in ids:
        yield chr(id)


# Build custom streaming response class to encapsulate the logic for streaming tokens.
class StreamingResponse(_StreamingResponse):
    def __init__(self, token_streamer: Callable[[Send], Awaitable[None]], **kwargs) -> None:
        # Initialize with the token streaming function.
        super().__init__(content=iter(()), **kwargs)
        self.token_streamer = token_streamer

    async def stream_response(self, send: Send) -> None:
        # Start the HTTP response and stream tokens using the provided streaming function.
        await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
        await self.token_streamer(send)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def __call__(self, scope, receive, send):
        # Override the __call__ method to handle the streaming response.
        await self.stream_response(send)


# Implement custom callback to stream generated tokens.
async def generate_and_stream_tokens(prompt_text: str, send: Send):
    input_ids = tokenizer(prompt_text)
    for token in model(input_ids):
        print("tok:", token)
        # Send each token as a separate chunk in the response.
        await send({"type": "http.response.body", "body": (token + '\n').encode('utf-8'), "more_body": True})
        await asyncio.sleep(0.3)  # Simulate streaming by adding a delay between tokens.


# Implement stream tokens function
# This will be the model generating and streaming text
# Function to generate a streaming response based on the input prompt.
def return_streaming_response(prompt_text: str) -> StreamingResponse:
    # Prepare the streaming response by creating a custom StreamingResponse instance.
    token_streamer = partial(generate_and_stream_tokens, prompt_text)
    print(f"token_streamer: {token_streamer}")
    return StreamingResponse(token_streamer)

app = FastAPI()

# Endpoint to accept a text prompt and return a streaming response of tokens.
@app.get("/stream_tokens/")
async def stream_tokens(prompt_text: str):
    return return_streaming_response(prompt_text)