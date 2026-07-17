"""Fara prompt entry points.

``get_computer_use_system_prompt`` is re-exported from this repo's
``fara._prompts`` (which ships the FaraNext browser tool schema, the model
identities, the critical-points block, and the fn-call template). Only the
browser tool set is supported here.
"""

from ..._prompts import get_computer_use_system_prompt

__all__ = ["get_computer_use_system_prompt", "TOOL_SET_TO_MODE"]

TOOL_SET_TO_MODE = {
    "BROWSER_TOOLS_CORE": "fara_next_browser",
    "GPT54_BROWSER_TOOLS_CORE": "fara_next_browser",
    "BROWSER_TOOLS_WITH_READ_PAGE": "fara_next_browser",
    "GPT54_BROWSER_TOOLS_WITH_READ_PAGE": "fara_next_browser",
    "WINDOWS_TOOLS_CORE": "fara_next_windows_core",
    "WINDOWS_TOOLS_WITH_RUN_COMMAND": "fara_next_windows",
    "GPT54_WINDOWS_TOOLS_CORE": "fara_next_windows_core",
    "GPT54_WINDOWS_TOOLS_WITH_RUN_COMMAND": "fara_next_windows",
}
