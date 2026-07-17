"""Shared utilities for agents."""

from __future__ import annotations

import json
import logging
from typing import Any, List

from ...clients.wrapper import ChatCompletionClient
from ...clients.messages import SystemMessage, UserMessage, ImageObj

logger = logging.getLogger(__name__)


def compact_messages(
    messages: List[Any],
    tools: list,
    client: ChatCompletionClient,
    max_prompt_tokens: int,
    min_keep_rounds: int = 2,
) -> List[Any]:
    """Remove middle round triplets until token count is under max_prompt_tokens.

    Each round after the initial messages (system + first user) is a triplet:
    (assistant tool_call dict, tool result dict, UserMessage/user dict).
    Keeps the first two messages and the last ``min_keep_rounds`` rounds.
    Removes from the middle outward.

    Args:
        messages: The conversation message list (modified in place via filtering).
        tools: Tool definitions passed to the model.
        client: The ChatCompletionClient used for token counting.
        max_prompt_tokens: Target token budget. <=0 disables compaction.
        min_keep_rounds: Minimum recent rounds to always retain.
    """
    if max_prompt_tokens <= 0:
        return messages
    token_count = client.count_tokens(messages, tools)
    if token_count <= max_prompt_tokens:
        return messages

    round_starts = []
    idx = 2
    while idx + 2 < len(messages):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            round_starts.append(idx)
            idx += 3
        else:
            idx += 1

    removable = (
        round_starts[:-min_keep_rounds] if len(round_starts) > min_keep_rounds else []
    )
    if not removable:
        return messages

    mid = len(removable) // 2
    removal_order = sorted(range(len(removable)), key=lambda i: abs(i - mid))

    for ri in removal_order:
        start = removable[ri]
        if start >= len(messages) or messages[start] is None:
            continue
        messages[start] = None
        messages[start + 1] = None
        messages[start + 2] = None
        compact = [m for m in messages if m is not None]
        token_count = client.count_tokens(compact, tools)
        logger.info(f"Compacted messages: {token_count} tokens")
        if token_count <= max_prompt_tokens:
            return compact

    return [m for m in messages if m is not None]


def drop_old_screenshots(
    messages: List[Any],
    max_screenshots: int = 2,
) -> None:
    """Replace all but the latest ``max_screenshots`` image-bearing user messages
    with text-only summaries, in place.

    Looks for the ``screen_description`` field in the preceding assistant
    tool-call arguments to build a fallback label.
    """
    screenshot_indices = [
        j
        for j, m in enumerate(messages)
        if isinstance(m, UserMessage)
        and isinstance(m.content, list)
        and any(isinstance(c, ImageObj) for c in m.content)
    ]
    for j in screenshot_indices[:-max_screenshots]:
        screen_desc = ""
        if (
            j >= 2
            and isinstance(messages[j - 2], dict)
            and messages[j - 2].get("role") == "assistant"
        ):
            tool_calls = messages[j - 2].get("tool_calls", [])
            if tool_calls:
                prev_args = json.loads(tool_calls[0]["function"]["arguments"])
                if "screen_description" in prev_args:
                    screen_desc = (
                        f"[Previous screen: {prev_args['screen_description']}]"
                    )
        text_parts = [c for c in messages[j].content if isinstance(c, str)]
        combined = " ".join(text_parts) if text_parts else ""
        fallback = screen_desc or "[screenshot removed]"
        messages[j] = UserMessage(
            content=f"{fallback} {combined}".strip() if combined else fallback
        )


async def extract_from_page(
    markdown: str,
    question: str,
    client: ChatCompletionClient,
    max_chars: int = 20000,
) -> str:
    """Send page markdown to the model with a question, return concise answer."""
    if len(markdown) > max_chars:
        markdown = markdown[:max_chars] + "\n... [truncated]"
    extract_messages = [
        SystemMessage(
            content="You are given the markdown content of a web page. "
            "Answer the user's question based solely on this content. "
            "Be concise and extract only the relevant information."
        ),
        UserMessage(content=f"Page content:\n{markdown}\n\nQuestion: {question}"),
    ]
    result = await client.create(messages=extract_messages)
    return result.content.content or "Error: Extraction failed."
