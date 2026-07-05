import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from builder_agent.web.events import sse_manager
from builder_agent.web.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture the running event loop for thread-safe event broadcasting
    sse_manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="Whetstone Dashboard", lifespan=lifespan)
app.include_router(router)


def start_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
