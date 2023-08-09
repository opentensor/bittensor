# client/dendrite.py
# Query code to request tokens from the server and print the streamed response.

import aiohttp
import asyncio

async def get_streamed_tokens(prompt_text: str):
    url = f"http://127.0.0.1:8000/stream_tokens/?prompt_text={prompt_text}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            async for chunk in response.content.iter_any():  # Read in chunks.
                print(chunk.decode('utf-8'))

prompt_text = "Your text here"
asyncio.run(get_streamed_tokens(prompt_text))