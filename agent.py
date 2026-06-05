"""Main agent loop for the liver disease diagnosis and treatment agent."""

from __future__ import annotations
import json
import sys
from typing import Any

import anthropic

from tools import TOOLS, execute_tool
from prompts import get_system_prompt

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096


class LiverAgent:
    """Agentic loop for liver disease clinical decision support."""

    def __init__(self, mode: str = "physician", verbose: bool = False):
        self.client = anthropic.Anthropic()
        self.mode = mode
        self.verbose = verbose
        self.conversation: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0

    def _system_prompt(self) -> list[dict]:
        return get_system_prompt(self.mode)

    def _build_messages(self) -> list[dict]:
        return self.conversation.copy()

    def _record_usage(self, usage: Any) -> None:
        self.total_input_tokens += getattr(usage, "input_tokens", 0)
        self.total_output_tokens += getattr(usage, "output_tokens", 0)
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def run_turn(
        self,
        user_message: str,
        on_text: Any = None,
        on_tool_start: Any = None,
        on_tool_end: Any = None,
    ) -> str:
        """Run a single conversation turn, handling the agentic tool-use loop.

        Returns the final assistant text response.
        """
        self.conversation.append({"role": "user", "content": user_message})

        final_text = ""
        iterations = 0
        max_iterations = 10  # safety cap

        while iterations < max_iterations:
            iterations += 1

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self._system_prompt(),
                tools=TOOLS,
                messages=self._build_messages(),
                thinking={"type": "disabled"},  # keep responses fast on Sonnet
            )

            self._record_usage(response.usage)

            if self.verbose:
                print(f"[DEBUG] Stop reason: {response.stop_reason}, "
                      f"Input tokens: {response.usage.input_tokens}, "
                      f"Output tokens: {response.usage.output_tokens}, "
                      f"Cache read: {getattr(response.usage, 'cache_read_input_tokens', 0)}, "
                      f"Cache write: {getattr(response.usage, 'cache_creation_input_tokens', 0)}",
                      file=sys.stderr)

            # Collect text and tool use blocks
            text_blocks = []
            tool_use_blocks = []
            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # Emit any text content
            for txt in text_blocks:
                final_text += txt
                if on_text:
                    on_text(txt)

            # Append assistant message to history
            self.conversation.append({
                "role": "assistant",
                "content": response.content,
            })

            # If no tool calls, we're done
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                break

            # Execute all tool calls
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input

                if on_tool_start:
                    on_tool_start(tool_name, tool_input)

                result = execute_tool(tool_name, tool_input)
                result_str = json.dumps(result, indent=2, default=str)

                if on_tool_end:
                    on_tool_end(tool_name, result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_str,
                })

            # Send tool results back
            self.conversation.append({
                "role": "user",
                "content": tool_results,
            })

        return final_text

    def chat(self, message: str) -> str:
        """Convenience method: run a turn and return the response."""
        return self.run_turn(message)

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation = []

    @property
    def usage_summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }
