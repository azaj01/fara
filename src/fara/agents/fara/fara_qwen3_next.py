"""FaraQwen3NextAgent - Extends FaraQwen3Agent with expanded tool sets."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

from ...clients.messages import FunctionCall, ImageObj, SystemMessage, UserMessage

from ...core.run_context import RunContext
from .fara_qwen3 import FaraQwen3Agent, FaraQwen3AgentConfig
from .fara_types import WebSurferEvent
from .prompts import get_computer_use_system_prompt, TOOL_SET_TO_MODE
from ..computer_agent.utils import extract_from_page
from ..utils import (
    format_text_observation,
    save_and_truncate_command_output,
    wrap_command_with_tee,
)


def extract_allowed_actions(system_prompt_text: str) -> frozenset[str]:
    """Extract the allowed action names from the tool schema enum in the system prompt."""
    match = re.search(r'"enum":\s*\[([^\]]+)\]', system_prompt_text)
    if not match:
        raise ValueError("Could not find action enum in system prompt")
    return frozenset(json.loads(f"[{match.group(1)}]"))


class FaraQwen3NextAgentConfig(FaraQwen3AgentConfig):
    """Configuration for FaraQwen3NextAgent."""

    name: str = "fara_next"
    max_observation_chars: int = 1000
    computer_use_mode: str = "fara_next_browser"
    identity: str | None = "fara_qwen3vl"
    critical_points: str | None = "fara-1.5"


class FaraQwen3NextAgent(FaraQwen3Agent):
    """Web/desktop automation agent with expanded tool sets (FaraNext).

    Extends FaraQwen3Agent with:
    - Dynamic tool mode selection (browser vs windows) via config.computer_use_mode
    - New GUI actions: double_click, right_click, triple_click, left_click_drag, hscroll
    - New non-GUI actions: read_page_answer_question, ask_user_question, run_command
    - Cursor position tracking for left_click_drag
    - press_enter defaults to False (model should use key(["Enter"]) explicitly)
    """

    _TEXT_OBSERVATION_ACTIONS: frozenset[str] = frozenset(
        {
            "run_command",
            "read_page_answer_question",
        }
    )

    _MODE_CANONICAL_IDENTITY: dict[str, tuple[str, ...]] = {
        "fara_next_browser": ("fara_qwen3vl", "fara_qwen35"),
        "fara_next_windows": ("fara_qwen3vl_windows", "fara_qwen35_windows"),
        "fara_next_windows_core": ("fara_qwen3vl_windows", "fara_qwen35_windows"),
    }

    def __init__(
        self,
        config: FaraQwen3NextAgentConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self.config: FaraQwen3NextAgentConfig
        self._allowed_actions: frozenset[str] = frozenset()
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0
        self._pending_observation: str = ""

    @classmethod
    def _get_config_class(cls) -> type[FaraQwen3NextAgentConfig]:
        return FaraQwen3NextAgentConfig

    async def initialize(self, run_context: RunContext) -> None:
        await super().initialize(run_context)
        valid_modes = set(TOOL_SET_TO_MODE.values())
        if self.config.computer_use_mode not in valid_modes:
            raise ValueError(
                f"Unknown computer_use_mode '{self.config.computer_use_mode}'. "
                f"Known values: {sorted(valid_modes)}"
            )
        canonical_identities = self._MODE_CANONICAL_IDENTITY[
            self.config.computer_use_mode
        ]
        if (
            self.config.identity is not None
            and self.config.identity not in canonical_identities
        ):
            raise ValueError(
                f"identity={self.config.identity!r} does not match any canonical "
                f"identity in {list(canonical_identities)!r} for "
                f"computer_use_mode={self.config.computer_use_mode!r}. "
                f"Use one of {list(canonical_identities)!r} or set identity=None "
                f"to auto-detect."
            )

    def _get_system_message(self, screenshot):
        system_prompt_info = get_computer_use_system_prompt(
            screenshot,
            self.mlm_processor_im_cfg,
            mode=self.config.computer_use_mode,
            include_input_text_key_args=self.config.include_input_text_key_args,
            fn_call_template=self.config.fn_call_template,
            randomize=False,
            display_size=self.DISPLAY_SIZE,
            identity=self.config.identity,
            critical_points=self.config.critical_points,
        )
        self._state.mlm_width, self._state.mlm_height = system_prompt_info["im_size"]
        scaled_screenshot = screenshot.resize(
            (self._state.mlm_width, self._state.mlm_height)
        )

        system_message = []
        full_text = ""
        for msg in system_prompt_info["conversation"]:
            tmp_content = ""
            for content in msg["content"]:
                tmp_content += content["text"]
            system_message.append(SystemMessage(content=tmp_content))
            full_text += tmp_content

        self._allowed_actions = extract_allowed_actions(full_text)

        return system_message, scaled_screenshot

    def _get_final_answer(self, thoughts: str, action_description: str) -> str:
        return action_description

    def _get_observation_prefix(self) -> str:
        obs = self._pending_observation
        self._pending_observation = ""
        return obs

    def maybe_remove_old_screenshots(self, history, includes_current=False):
        """Override to also strip observation text from old messages in windows mode."""
        result = super().maybe_remove_old_screenshots(history, includes_current)
        if "windows" not in self.config.computer_use_mode:
            return result

        for msg in result:
            if not isinstance(msg, UserMessage):
                continue
            if (msg.metadata or {}).get("is_original", False):
                continue
            if isinstance(msg.content, list):
                has_image = any(isinstance(c, ImageObj) for c in msg.content)
                if not has_image:
                    msg.content = [
                        self.USER_MESSAGE if isinstance(c, str) else c
                        for c in msg.content
                    ]

        return result

    async def _execute_action(
        self, env, function_call: list[FunctionCall]
    ) -> tuple[bool, str]:
        """Execute an action. Only actions in the current schema are allowed."""
        name = function_call[0].name
        args = function_call[0].arguments
        action_type = args.get("action", "")

        if action_type not in self._allowed_actions:
            if self.config.terminate_on_parse_error:
                self.logger.warning(
                    "terminate_on_parse_error=true: invalid action %r — "
                    "ending trajectory with thoughts as final answer",
                    action_type,
                )
                return True, args.get("thoughts", "") or args.get("answer", "")
            raise ValueError(
                f"Action '{action_type}' is not allowed in mode "
                f"'{self.config.computer_use_mode}'. Allowed: {sorted(self._allowed_actions)}"
            )

        url = (await env.get_page_context()).url
        self.logger.debug(
            WebSurferEvent(
                source="FaraQwen3NextAgent",
                url=url,
                action=name,
                arguments=args,
                message=f"{name}( {json.dumps(args)} )",
            )
        )

        if "coordinate" in args:
            args["coordinate"] = self._proc(args["coordinate"])

        is_stop_action, action_description = await self._dispatch_action(
            env, action_type, args
        )

        if action_type in self._TEXT_OBSERVATION_ACTIONS:
            self._pending_observation = format_text_observation(
                action_type,
                action_description,
                self.config.max_observation_chars,
            )

        if not is_stop_action:
            if hasattr(env, "wait_for_load"):
                await env.wait_for_load()

        self._state.num_actions += 1
        return is_stop_action, action_description

    async def _dispatch_action(
        self, env, action_type: str, args: dict
    ) -> tuple[bool, str]:
        """Dispatch action using ComputerEnvironment canonical methods."""

        # -- GUI actions --

        if action_type == "left_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.left_click(tgt_x, tgt_y)
            self._cursor_x, self._cursor_y = tgt_x, tgt_y
            return False, f"I clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "key":
            keys = args.get("keys", [])
            await env.key(keys)
            return False, f"I pressed the following keys: {keys}"

        elif action_type == "mouse_move":
            tgt_x, tgt_y = args["coordinate"]
            await env.mouse_move(tgt_x, tgt_y)
            self._cursor_x, self._cursor_y = tgt_x, tgt_y
            return False, f"I moved the cursor to ({tgt_x}, {tgt_y})."

        elif action_type == "type":
            text = str(args.get("text", args.get("text_value", "")))
            await env.type(text)
            return False, f"I typed '{text}'."

        elif action_type == "scroll":
            pixels = int(args.get("pixels", 0))
            pixels = int(pixels * self.config.viewport_height / self.DISPLAY_SIZE)
            await env.scroll(pixels)
            direction = "up" if pixels > 0 else "down"
            return False, f"I scrolled {direction}."

        elif action_type == "double_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.double_click(tgt_x, tgt_y)
            self._cursor_x, self._cursor_y = tgt_x, tgt_y
            return False, f"I double-clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "right_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.right_click(tgt_x, tgt_y)
            self._cursor_x, self._cursor_y = tgt_x, tgt_y
            return False, f"I right-clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "triple_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.triple_click(tgt_x, tgt_y)
            self._cursor_x, self._cursor_y = tgt_x, tgt_y
            return False, f"I triple-clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "left_click_drag":
            tgt_x, tgt_y = args["coordinate"]
            await env.left_click_drag(tgt_x, tgt_y)
            self._cursor_x, self._cursor_y = tgt_x, tgt_y
            return False, f"I dragged to ({tgt_x}, {tgt_y})."

        elif action_type == "hscroll":
            pixels = int(args.get("pixels", 0))
            pixels = int(pixels * self.config.viewport_width / self.DISPLAY_SIZE)
            await env.hscroll(pixels)
            return False, f"I scrolled horizontally by {pixels} pixels."

        # -- Browser-only actions --

        elif action_type == "visit_url":
            url = args.get("url", "")
            if url.startswith(("https://", "http://", "file://", "about:")):
                target = url
            elif " " in url:
                target = f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH"
            else:
                target = "https://" + url
            try:
                await env.goto_url(target)
            except Exception as e:
                msg = f"visit_url to {url} failed: {type(e).__name__}: {e}"
                self._pending_observation = msg
                return False, f"I tried to navigate to {url} but it failed: {e}"
            return False, f"I navigated to {url}."

        elif action_type == "history_back":
            await env.go_back()
            return False, "I clicked the browser back button."

        elif action_type == "web_search":
            query = args.get("query", "")
            await env.goto_url(
                f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH"
            )
            return False, f"I searched for '{query}'."

        elif action_type == "read_page_answer_question":
            question = str(args.get("question", ""))
            markdown = await env.get_page_markdown()
            answer = await extract_from_page(markdown, question, self._client)
            return False, f"I read the page to answer: {question}\nAnswer: {answer}"

        # -- Windows-only actions --

        elif action_type == "run_command":
            command = str(args.get("command", ""))
            step = self._state.current_step
            cmd = wrap_command_with_tee(command, step, self._os_type)
            result = await env.execute(cmd)
            if isinstance(result, dict):
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                result = f"{stdout}\n{stderr}".strip() if stderr else stdout
            else:
                result = str(result)
            result = save_and_truncate_command_output(
                result,
                self._output_dir,
                step,
                max_chars=self.config.max_observation_chars,
                os_type=self._os_type,
            )
            return False, f"I executed command: {command}\nOutput: {result}"

        # -- Shared non-GUI actions --

        elif action_type == "ask_user_question":
            question = str(args.get("question", ""))
            return True, f"I asked the user: {question}"

        elif action_type == "wait":
            duration = args.get("time", args.get("duration", 3.0))
            await env.wait(duration)
            return False, f"I waited {duration}s."

        elif action_type == "pause_and_memorize_fact":
            fact = str(args.get("fact", ""))
            self._state.facts.append(fact)
            return False, f"I memorized the following fact: {fact}"

        elif action_type == "terminate":
            answer = args.get("answer")
            if answer is None:
                raise ValueError("terminate action requires 'answer' argument")
            return True, answer

        else:
            raise ValueError(f"Unknown action: {action_type}")

    def _proc(self, coordinate):
        """Shorthand for proc_coords with standard display/viewport args."""
        return self.proc_coords(
            coordinate,
            self.DISPLAY_SIZE,
            self.DISPLAY_SIZE,
            self.config.viewport_width,
            self.config.viewport_height,
        )
