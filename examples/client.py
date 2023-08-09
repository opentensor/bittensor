import aiohttp
import asyncio

# async def fetch():
#     async with aiohttp.ClientSession() as session:
#         async with session.post("http://127.0.0.1:8000/streaming") as response:
#             print(response.headers.get('Content-Type'))
#             if response.headers.get('Content-Type').startswith('text/event-stream'):
#                 while True:
#                     chunk = await response.content.readany()
#                     if chunk:
#                         print(chunk.decode('utf-8'))  # Print each chunk
#                     else:
#                         print("No more data")
#                         break


async def fetch():
    async with aiohttp.ClientSession() as session:
        async with session.post("http://127.0.0.1:8000/streaming") as response:
            print(response.headers.get('Content-Type'))
            if response.headers.get('Content-Type').startswith('text/event-stream'):
                async for chunk in response.content.iter_any():
                    print(chunk.decode('utf-8'))

# async def fetch():
#     async with aiohttp.ClientSession() as session:
#         async with session.post("http://127.0.0.1:8000/streaming") as response:
#             print(response.headers.get('Content-Type'))  # Should print 'text/event-stream'
#             if response.headers.get('Content-Type') == 'text/event-stream':
#                 while not response.content.at_eof():
#                     chunk = await response.content.readany()
#                     print(chunk.decode('utf-8'))  # Should print each character

asyncio.run(fetch())
