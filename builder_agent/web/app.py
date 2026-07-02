import asyncio

import uvicorn
from fastapi import FastAPI

from builder_agent.web.events import sse_manager
from builder_agent.web.routes import router

app = FastAPI(title="Whetstone Dashboard")
app.include_router(router)

@app.on_event("startup")
def startup_event():
    # Capture the running event loop for thread-safe event broadcasting
    sse_manager.loop = asyncio.get_event_loop()

def start_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
