from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

def generate_tokens():
    for char in "hello this is a test of streaming.":
        yield char + '\n'

@app.post("/streaming")
async def streaming():
    return StreamingResponse(generate_tokens(), media_type="text/event-stream")
