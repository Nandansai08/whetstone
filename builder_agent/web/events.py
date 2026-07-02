import asyncio
from typing import Dict, Set


class SSEManager:
    def __init__(self):
        self.listeners: Dict[str, Set[asyncio.Queue]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, build_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        if build_id not in self.listeners:
            self.listeners[build_id] = set()
        self.listeners[build_id].add(queue)
        return queue

    def unsubscribe(self, build_id: str, queue: asyncio.Queue):
        if build_id in self.listeners:
            self.listeners[build_id].discard(queue)
            if not self.listeners[build_id]:
                del self.listeners[build_id]

    def broadcast(self, build_id: str, event_type: str, data: dict):
        if build_id in self.listeners:
            for queue in list(self.listeners[build_id]):
                if self.loop and self.loop.is_running():
                    self.loop.call_soon_threadsafe(
                        queue.put_nowait, {"type": event_type, "data": data}
                    )
                else:
                    try:
                        queue.put_nowait({"type": event_type, "data": data})
                    except Exception:
                        pass

sse_manager = SSEManager()
