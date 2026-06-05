"""FastAPI web backend for HepatoAI — powered by Groq."""

from __future__ import annotations
import asyncio
import json
import os
import uuid
from typing import AsyncGenerator

from groq import Groq
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tools import TOOLS, execute_tool
from prompts import PHYSICIAN_SYSTEM_PROMPT, PATIENT_SYSTEM_PROMPT

app = FastAPI(title="HepatoAI")

MODEL = "llama-3.3-70b-versatile"
sessions: dict[str, list[dict]] = {}


def _to_groq_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]

GROQ_TOOLS = _to_groq_tools(TOOLS)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    mode: str = "physician"
    lang: str = "zh"


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


def _system_prompt(mode: str, lang: str) -> str:
    base = PHYSICIAN_SYSTEM_PROMPT if mode == "physician" else PATIENT_SYSTEM_PROMPT
    if lang == "en":
        base += "\n\nIMPORTANT: The user is communicating in English. Please respond entirely in English."
    else:
        base += "\n\nIMPORTANT: 请用中文回复所有内容。"
    return base


async def stream_agent(message: str, session_id: str, mode: str, lang: str) -> AsyncGenerator[str, None]:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": message})

    def send(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    system_msg = {"role": "system", "content": _system_prompt(mode, lang)}
    loop = asyncio.get_event_loop()
    iterations = 0
    final_text = ""

    while iterations < 10:
        iterations += 1

        messages = [system_msg] + history

        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                messages=messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
            ),
        )

        msg = response.choices[0].message
        usage = response.usage

        yield send("usage", {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cache_read": 0,
            "cache_write": 0,
        })

        if msg.content:
            final_text += msg.content
            yield send("text", {"content": msg.content})

        # Append assistant message
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            **({"tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]} if msg.tool_calls else {}),
        })

        if not msg.tool_calls or response.choices[0].finish_reason == "stop":
            break

        # Execute tools
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            yield send("tool_start", {"name": tool_name, "input": tool_input})

            result = await loop.run_in_executor(
                None, lambda tn=tool_name, ti=tool_input: execute_tool(tn, ti)
            )

            yield send("tool_result", {"name": tool_name, "result": result})

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    yield send("done", {"final_text": final_text})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_agent(req.message, req.session_id, req.mode, req.lang),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")
