import asyncio
import difflib
import json
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from builder_agent.web.events import sse_manager
from builder_agent.web.history import BuildHistory

router = APIRouter()

# Setup templates path
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

@router.get("/", response_class=HTMLResponse)
@router.get("/builds", response_class=HTMLResponse)
def index(request: Request):
    db = BuildHistory()
    builds = db.get_builds()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"builds": builds}
    )

@router.get("/build/{build_id}", response_class=HTMLResponse)
def build_detail(request: Request, build_id: str):
    db = BuildHistory()
    build = db.get_build(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    attempts = db.get_attempts(build_id)

    # Group attempts by subtask_id for charting and logs
    subtasks = {}
    for a in attempts:
        sid = a["subtask_id"]
        if sid not in subtasks:
            subtasks[sid] = []
        subtasks[sid].append(a)

    return templates.TemplateResponse(
        request=request,
        name="build.html",
        context={
            "build": build,
            "attempts": attempts,
            "subtasks": subtasks,
        }
    )

@router.get("/build/{build_id}/diff", response_class=HTMLResponse)
def get_diff(build_id: str, subtask: str, iter1: int, iter2: int):
    db = BuildHistory()
    attempts = db.get_attempts(build_id)
    sub_attempts = [a for a in attempts if a["subtask_id"] == subtask]

    code1 = ""
    code2 = ""
    for a in sub_attempts:
        if a["iteration"] == iter1:
            code1 = a["code"]
        if a["iteration"] == iter2:
            code2 = a["code"]

    diff_lines = list(difflib.unified_diff(
        code1.splitlines(keepends=True),
        code2.splitlines(keepends=True),
        fromfile=f"Iteration {iter1}",
        tofile=f"Iteration {iter2}"
    ))

    if not diff_lines:
        return HTMLResponse(
            content="<div class='text-gray-400 p-4'>No differences found.</div>"
        )

    html_lines = []
    for line in diff_lines:
        safe_line = (
            line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        if line.startswith("+") and not line.startswith("+++"):
            html_lines.append(
                f'<span class="text-emerald-400 font-mono block '
                f'bg-emerald-950/30 px-2">{safe_line}</span>'
            )
        elif line.startswith("-") and not line.startswith("---"):
            html_lines.append(
                f'<span class="text-rose-400 font-mono block '
                f'bg-rose-950/30 px-2">{safe_line}</span>'
            )
        elif line.startswith("@@"):
            html_lines.append(
                f'<span class="text-sky-400 font-mono block '
                f'bg-sky-950/30 px-2">{safe_line}</span>'
            )
        else:
            html_lines.append(
                f'<span class="text-slate-400 font-mono block '
                f'px-2">{safe_line}</span>'
            )

    diff_html = (
        "<pre class='bg-slate-950/80 border border-slate-800 p-4 rounded-lg "
        "text-xs leading-5 overflow-x-auto text-left'>"
        + "".join(html_lines)
        + "</pre>"
    )
    return HTMLResponse(content=diff_html)

@router.get("/build/{build_id}/stream")
async def get_stream(build_id: str):
    queue = sse_manager.subscribe(build_id)
    db = BuildHistory()

    async def event_generator():
        last_attempt_count = 0
        last_status = None
        try:
            while True:
                # 1. Fetch current status and attempts from shared database
                build = db.get_build(build_id)
                if not build:
                    break

                attempts = db.get_attempts(build_id)
                if len(attempts) > last_attempt_count:
                    for i in range(last_attempt_count, len(attempts)):
                        attempt = attempts[i]
                        yield f"event: attempt\ndata: {json.dumps(attempt)}\n\n"
                    last_attempt_count = len(attempts)

                if build["status"] != last_status:
                    yield f"event: status\ndata: {json.dumps(build)}\n\n"
                    last_status = build["status"]

                if build["status"] in ("passed", "failed"):
                    break

                # 2. Check for in-process queue updates
                while not queue.empty():
                    msg = queue.get_nowait()
                    yield f"event: {msg['type']}\ndata: {json.dumps(msg['data'])}\n\n"

                await asyncio.sleep(0.5)
        finally:
            sse_manager.unsubscribe(build_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
