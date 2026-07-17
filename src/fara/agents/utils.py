"""Utilities shared across agent implementations."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

CMD_OUTPUT_DIR = "cmd_outputs"
CMD_OUTPUT_ENV_DIR_LINUX = "/tmp"
CMD_OUTPUT_ENV_DIR_WINDOWS = r"C:\Users\Docker\AppData\Local\Temp"


def get_cmd_output_env_path(step: int, os_type: str = "linux") -> str:
    """Return the in-environment file path for a step's command output."""
    if os_type == "windows":
        return f"{CMD_OUTPUT_ENV_DIR_WINDOWS}\\cmd_output_step_{step}.txt"
    return f"{CMD_OUTPUT_ENV_DIR_LINUX}/cmd_output_step_{step}.txt"


def wrap_command_with_tee(command: str, step: int, os_type: str = "linux") -> str:
    """Wrap a shell command so its combined stdout+stderr is teed to a file.

    The original command's output is still returned; a copy is saved inside
    the environment for the agent to read.

    On Linux (Bash): uses `(cmd) 2>&1 | tee <path>`
    On Windows (PowerShell): uses `& { cmd } 2>&1 | Tee-Object -FilePath <path>`
    """
    env_path = get_cmd_output_env_path(step, os_type)
    if os_type == "windows":
        return f"& {{ {command} }} 2>&1 | Tee-Object -FilePath '{env_path}'"
    return f"({command}) 2>&1 | tee {env_path}"


def save_and_truncate_command_output(
    output: str,
    output_dir: Path | None,
    step: int,
    max_chars: int = 1000,
    os_type: str = "linux",
) -> str:
    """Save full output to host, return truncated version for LLM context.

    - If output <= max_chars (or max_chars <= 0): return output unchanged.
    - Otherwise: save full output to <output_dir>/cmd_outputs/step_<N>.txt
      and return first+last halves with a truncation notice + env file path.
    """
    if max_chars <= 0 or len(output) <= max_chars:
        return output

    if output_dir:
        host_dir = output_dir / CMD_OUTPUT_DIR
        os.makedirs(host_dir, exist_ok=True)
        host_path = host_dir / f"step_{step}.txt"
        host_path.write_text(output, encoding="utf-8")
        logger.debug("Saved full command output to %s", host_path)

    env_path = get_cmd_output_env_path(step, os_type)
    read_cmd = (
        f"Get-Content '{env_path}'" if os_type == "windows" else f"cat {env_path}"
    )
    half = max_chars // 2
    return (
        f"{output[:half]}\n"
        f"... [truncated {len(output) - max_chars} chars] ...\n"
        f"{output[-half:]}\n"
        f"Full output saved to: {env_path}\n"
        f"You can read it with: {read_cmd}"
    )


def truncate_observation(text: str, max_chars: int = 1000) -> str:
    """Truncate text keeping first and last halves with a marker in between.

    Duplicated in agento_train/data/convo_builder.py (FaraNextConvoBuilder._truncate_observation)
    because agento_train cannot import from agento_next. Keep both in sync.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n... [truncated {len(text) - max_chars} chars] ...\n"
        + text[-half:]
    )


def format_text_observation(action_name: str, text: str, max_chars: int = 1000) -> str:
    """Prepare text observations for the next model turn.

    ``run_command`` observations may already include a saved-file hint from
    ``save_and_truncate_command_output``. Preserve them as-is so that hint
    isn't lost to a second truncation pass. Other text observations still use
    the standard truncation helper.
    """
    if action_name == "run_command":
        return text
    return truncate_observation(text, max_chars)


def get_screenshot_dimensions(screenshot: Image.Image) -> tuple[int, int]:
    """Return validated screenshot dimensions for prompt construction."""
    width, height = screenshot.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid screenshot size: {width}x{height}")
    return int(width), int(height)


def build_screen_text(
    prefix_text: str,
    screenshot: Image.Image,
    url: str = "",
    page_context: str = "",
    context_summary: str = "",
) -> str:
    """Build the text block paired with a screenshot before a model call."""
    width, height = get_screenshot_dimensions(screenshot)
    parts = [
        prefix_text,
        f"Screenshot resolution: {width}x{height}",
    ]
    if url:
        parts.append(f"Current URL: {url}")
    if page_context:
        parts.append(page_context)

    text = "\n".join(parts)
    if context_summary:
        text = f"{text}\n\n{context_summary}"
    return text
