import inspect
from fastapi.responses import StreamingResponse


async def stream_generator(generator):
    if inspect.isasyncgen(generator):
        async for chunk in generator:
            yield f"data: {chunk.json()}\n\n"
    else:
        for chunk in generator:
            yield f"data: {chunk.json()}\n\n"

def sse_response(generator):
    return StreamingResponse(
        stream_generator(generator),
        media_type="text/event-stream"
    )