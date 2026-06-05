"""FastAPI web backend for HepatoAI."""

from __future__ import annotations
import asyncio
import json
import uuid
from typing import AsyncGenerator

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tools import TOOLS, execute_tool
from prompts import get_system_prompt

app = FastAPI(title="HepatoAI")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# In-memory session store: session_id -> list of messages
sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str
    mode: str = "physician"


class NewSessionRequest(BaseModel):
    mode: str = "physician"


@app.post("/api/session")
def new_session(req: NewSessionRequest):
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return {"session_id": sid, "mode": req.mode}


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str):
    sessions[session_id] = []
    return {"ok": True}


async def stream_agent(message: str, session_id: str, mode: str) -> AsyncGenerator[str, None]:
    """Run the agentic loop and stream SSE events to the client."""
    client = anthropic.Anthropic()
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": message})

    def send(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    iterations = 0
    final_text = ""

    while iterations < 10:
        iterations += 1

        # Run API call in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=get_system_prompt(mode),
                tools=TOOLS,
                messages=history,
                thinking={"type": "disabled"},
            ),
        )

        # Emit usage info
        yield send("usage", {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        })

        text_parts = []
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                final_text += block.text
                yield send("text", {"content": block.text})
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # Append assistant message
        history.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        # Execute tools
        tool_results = []
        for tool_block in tool_use_blocks:
            yield send("tool_start", {
                "name": tool_block.name,
                "input": tool_block.input,
            })

            result = await loop.run_in_executor(
                None,
                lambda tb=tool_block: execute_tool(tb.name, tb.input),
            )

            yield send("tool_result", {
                "name": tool_block.name,
                "result": result,
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result, default=str),
            })

        history.append({"role": "user", "content": tool_results})

    yield send("done", {"final_text": final_text})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_agent(req.message, req.session_id, req.mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")
