"""Main agent loop — powered by Groq (Llama 3.3 70B)."""

from __future__ import annotations
import json
import os
from typing import Any

from groq import Groq

from tools import TOOLS, execute_tool
from prompts import PHYSICIAN_SYSTEM_PROMPT, PATIENT_SYSTEM_PROMPT

MODEL = "llama-3.3-70b-versatile"

# Convert Anthropic-style tool definitions → OpenAI/Groq format
def _to_groq_tools(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result

GROQ_TOOLS = _to_groq_tools(TOOLS)


class LiverAgent:
    """Agentic loop for liver disease clinical decision support via Groq."""

    def __init__(self, mode: str = "physician", verbose: bool = False):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        self.mode = mode
        self.verbose = verbose
        self.conversation: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _system_content(self) -> str:
        return PHYSICIAN_SYSTEM_PROMPT if self.mode == "physician" else PATIENT_SYSTEM_PROMPT

    def _build_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system_content()}] + self.conversation

    def run_turn(
        self,
        user_message: str,
        on_text: Any = None,
        on_tool_start: Any = None,
        on_tool_end: Any = None,
    ) -> str:
        self.conversation.append({"role": "user", "content": user_message})

        final_text = ""
        iterations = 0

        while iterations < 10:
            iterations += 1

            response = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                messages=self._build_messages(),
                tools=GROQ_TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            usage = response.usage
            if usage:
                self.total_input_tokens += usage.prompt_tokens or 0
                self.total_output_tokens += usage.completion_tokens or 0

            if self.verbose:
                import sys
                print(f"[DEBUG] finish_reason={response.choices[0].finish_reason} "
                      f"in={usage.prompt_tokens if usage else '?'} "
                      f"out={usage.completion_tokens if usage else '?'}", file=sys.stderr)

            # Emit text
            if msg.content:
                final_text += msg.content
                if on_text:
                    on_text(msg.content)

            # Append assistant turn
            self.conversation.append({
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

            # No tool calls → done
            if not msg.tool_calls or response.choices[0].finish_reason == "stop":
                break

            # Execute tools
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                if on_tool_start:
                    on_tool_start(tool_name, tool_input)

                result = execute_tool(tool_name, tool_input)

                if on_tool_end:
                    on_tool_end(tool_name, result)

                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        return final_text

    def chat(self, message: str) -> str:
        return self.run_turn(message)

    def reset(self) -> None:
        self.conversation = []

    @property
    def usage_summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
