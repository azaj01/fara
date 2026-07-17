"""FaraQwen3Agent - Web automation agent using Qwen3 vision-language model."""

from __future__ import annotations

import copy
import io
import json
import ast
import logging
import shutil

import openai
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

from PIL import Image
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from ...clients.wrapper import ChatCompletionClient
from ...clients.create_utils import create_client_from_config
from ...clients.messages import (
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ImageObj,
    FunctionCall,
)

from ...core.agent import Agent, AgentConfig
from ...core.run_context import RunContext
from ...core.data_point import (
    Action as DPAction,
    AgentState,
    LLMConversation,
    LLMMessage as DPLLMMessage,
    SolverStatus,
    Outcome,
    ComputerObservation,
)
from ..captcha import wait_for_captcha
from ..coord_spaces import FARA_DISPLAY_SIZE
from .fara_types import WebSurferEvent
from .prompts import get_computer_use_system_prompt
from .utils import get_trimmed_url


class FaraQwen3AgentConfig(AgentConfig):
    """Configuration for FaraAgent.

    Notable fields:
      extra_create_args: Extra kwargs merged on top of ``temperature: 0`` and
        forwarded to ``chat.completions.create``. vLLM-only keys are routed
        via ``extra_body``.
      auto_user_reply: When True, ``ask_user_question`` does NOT halt
        the run — a fixed dummy user response is injected and the agent
        continues. For eval / no-user-simulator settings only. Default False
        preserves ``task_workflow.py``'s ``WAITING_FOR_USER`` semantics for
        data_gen.
      captcha_timeout_limit: Number of consecutive captcha-wait timeouts
        tolerated within a single run before the per-step captcha-wait
        gate is auto-disabled for the rest of the run. Set to ``0`` to
        skip the gate entirely from the start (appropriate without a
        Browserbase cloud session, which is what triggers the captcha
        solver). Defaults to ``2``.
      raise_on_captcha_timeout: When True, restore the legacy behavior
        of raising ``RuntimeError`` on the first captcha-wait timeout.
        When False (default), the agent logs a warning and moves past
        the timed-out captcha, falling back to ``captcha_timeout_limit``
        to auto-disable the gate after repeated timeouts.
    """

    name: str = "fara"
    client_config: dict[str, Any] | None = None
    start_page: str = "about:blank"
    max_rounds: int = 10
    max_n_images: int = 3
    fn_call_template: str = "fara-qwen3vl"
    identity: str | None = "fara_qwen3vl"
    critical_points: str | None = "fara-1.5"
    computer_use_mode: str = "aurora"
    save_screenshots: bool = False
    animate_actions: bool = False
    single_tab_mode: bool = True
    include_input_text_key_args: bool = True
    viewport_width: int = 1440
    viewport_height: int = 900
    min_pixels: int = 3136
    max_pixels: int = 12845056
    extra_create_args: dict[str, Any] | None = None
    auto_user_reply: bool = False
    captcha_timeout_limit: int = 2
    raise_on_captcha_timeout: bool = True
    terminate_on_parse_error: bool = False
    image_token_estimate: int = 1500
    # When >0, drop oldest screenshots until the estimated prompt is <= this cap.
    image_budget_token_cap: int = 0


@dataclass
class FaraQwen3AgentState:
    """Mutable state for FaraQwen3Agent that persists across resume calls."""

    chat_history: list[LLMMessage] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    num_actions: int = 0
    current_step: int = 0
    mlm_width: int = 0
    mlm_height: int = 0


class FaraQwen3Agent(Agent):
    """Web automation agent using vision-language models."""

    DEFAULT_START_PAGE = "https://www.bing.com/"
    DISPLAY_SIZE = FARA_DISPLAY_SIZE
    PATCH_SIZE = 16
    MERGE_SIZE = 2
    USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
    MAX_URL_LENGTH = 100

    def __init__(
        self, config: FaraQwen3AgentConfig | dict[str, Any] | None = None, **kwargs: Any
    ):
        super().__init__(config, **kwargs)
        self.config: FaraQwen3AgentConfig
        if type(self) is FaraQwen3Agent:
            assert (
                self.config.computer_use_mode == "aurora"
            ), f"FaraQwen3Agent is aurora-v2 era; got computer_use_mode={self.config.computer_use_mode!r}"
        self.logger = logging.getLogger(__name__)
        self._client: ChatCompletionClient | None = None
        self._state: FaraQwen3AgentState | None = None
        self._pending_observation: str = ""
        self._os_type: str = "linux"
        self._output_dir = None
        # Captcha-wait disable: after `config.captcha_timeout_limit`
        # consecutive captcha-wait timeouts in a single run, stop waiting
        # for captchas at all. A limit of 0 disables the gate up front.
        self._captcha_timeouts: int = 0
        self._captcha_disabled: bool = self.config.captcha_timeout_limit <= 0

    @property
    def mlm_processor_im_cfg(self) -> dict[str, int]:
        return {
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
            "patch_size": self.PATCH_SIZE,
            "merge_size": self.MERGE_SIZE,
        }

    @classmethod
    def _get_config_class(cls) -> type[FaraQwen3AgentConfig]:
        return FaraQwen3AgentConfig

    async def initialize(self, run_context: RunContext) -> None:
        """Initialize the agent."""
        await super().initialize(run_context)
        self._state = FaraQwen3AgentState(
            mlm_width=self.config.viewport_width,
            mlm_height=self.config.viewport_height,
        )

        if self.config.client_config is not None:
            self._client = create_client_from_config(self.config.client_config)
        elif self.config.client is not None:
            self._client = self.config.client
        else:
            raise ValueError("Either client or client_config must be provided")

    async def run(
        self, run_context: RunContext, input: Any = None
    ) -> Tuple[str, List, List]:
        """Run the agent on a task.

        Supports resuming after WAITING_FOR_USER: if ``_state.chat_history``
        is already populated the fresh-start init is skipped and the user
        response is appended to the conversation.
        """
        env = run_context.environment
        traj = run_context.solver_log
        self._os_type = getattr(env, "os_type", "linux")
        self._output_dir = run_context.output_dir

        pending_user_response = ""
        if self._state.chat_history:
            # Continuation after WAITING_FOR_USER
            traj.status = SolverStatus.RUNNING
            pending_user_response = traj.get_last_user_message() or ""
            start_step = self._state.current_step
        else:
            # Fresh start
            task_instruction = run_context.task.instruction
            scaled_screenshot = await self._get_scaled_screenshot(env)
            await self._save_screenshot(
                env, run_context.output_dir, "screenshot_0_pre.png"
            )
            self._log_initial_observations(run_context, task_instruction)

            self._state.chat_history.append(
                UserMessage(
                    content=[ImageObj.from_pil(scaled_screenshot), task_instruction],
                    metadata={"is_original": True},
                )
            )
            start_step = 0

        all_actions = []
        all_observations = []
        final_answer = "<no_answer>"

        for step in range(start_step + 1, self.config.max_rounds + 1):
            is_first_round = step == 1

            if not self._captcha_disabled:
                if not await wait_for_captcha(env):
                    if self.config.raise_on_captcha_timeout:
                        raise RuntimeError(
                            "Captcha timed out, unable to proceed with web surfing."
                        )
                    self._captcha_timeouts += 1
                    self.logger.warning(
                        "Captcha timed out at step %d (%d/%d); moving past "
                        "unsolved captcha.",
                        step,
                        self._captcha_timeouts,
                        self.config.captcha_timeout_limit,
                    )
                    if self._captcha_timeouts >= self.config.captcha_timeout_limit:
                        self.logger.warning(
                            "Captcha timed out %d times; disabling further "
                            "captcha waits for the rest of this run.",
                            self._captcha_timeouts,
                        )
                        self._captcha_disabled = True

            pre_screenshot_name = f"screenshot_{step}_pre.png"
            await self._save_screenshot(
                env, run_context.output_dir, pre_screenshot_name
            )
            current_url, page_context = await env.get_page_context()

            function_call, raw_response = await self._generate_model_call(
                env,
                is_first_round,
                scaled_screenshot if is_first_round else None,
                user_response=pending_user_response,
            )
            pending_user_response = ""
            all_actions.append(raw_response)

            action_args = function_call[0].arguments
            action_name = action_args.get("action", "unknown")
            thoughts = action_args.get("thoughts", "")

            self.logger.debug(
                f"\nThought #{step}: {thoughts}\nAction #{step}: executing tool '{action_name}' with arguments {json.dumps(action_args)}"
            )

            run_context.add_observation(
                ComputerObservation(
                    screenshot_path=pre_screenshot_name,
                    url=current_url,
                    page_info=page_context,
                )
            )
            dp_action = self._log_action(
                run_context, action_name, action_args, raw_response
            )

            is_stop_action, action_description = await self._execute_action(
                env, function_call
            )
            all_observations.append(action_description)

            self.logger.debug(f"Observation#{step}: {action_description}")

            post_screenshot_name = f"screenshot_{step}_post.png"
            if is_stop_action:
                if run_context.output_dir:
                    shutil.copyfile(
                        run_context.output_dir / pre_screenshot_name,
                        run_context.output_dir / post_screenshot_name,
                    )
                page_context_after = page_context
                url_after = current_url
            else:
                await self._save_screenshot(
                    env, run_context.output_dir, post_screenshot_name
                )
                url_after, page_context_after = await env.get_page_context()

            self._log_observation(
                run_context,
                action_description,
                post_screenshot_name,
                page_info=page_context_after,
                action_id=dp_action.id,
                url=url_after,
            )
            self._state.current_step = step
            run_context.checkpoint()

            if action_name == "ask_user_question":
                if self.config.auto_user_reply:
                    pending_user_response = (
                        "keep going so far as you don't make up information, "
                        "you have my approval"
                    )
                    run_context.checkpoint()
                    continue
                traj.status = SolverStatus.WAITING_FOR_USER
                run_context.checkpoint()
                return action_description, all_actions, all_observations

            if is_stop_action:
                final_answer = self._get_final_answer(thoughts, action_description)
                traj.outcome = Outcome(answer=final_answer)
                traj.status = SolverStatus.COMPLETE
                break

        run_context.checkpoint()
        if traj.status != SolverStatus.COMPLETE:
            final_answer = self._get_final_answer(thoughts, action_description)
            traj.outcome = Outcome(answer=final_answer)
            traj.status = SolverStatus.COMPLETE

        return final_answer, all_actions, all_observations

    async def close(self, run_context: RunContext) -> None:
        """Cleanup after the agent is done."""
        self._state = None
        self._client = None
        self._captcha_timeouts = 0
        self._captcha_disabled = self.config.captcha_timeout_limit <= 0
        await super().close(run_context)

    def _get_final_answer(self, thoughts: str, action_description: str) -> str:
        return thoughts

    def _get_observation_prefix(self) -> str:
        """Return text to prepend to the next user message. Override in subclasses."""
        obs = self._pending_observation
        self._pending_observation = ""
        return obs

    def _log_action(
        self,
        run_context: RunContext,
        name: str,
        args: dict,
        raw_response: str,
    ) -> DPAction:
        clean_args = {k: v for k, v in args.items() if k not in ("action", "thoughts")}
        thoughts = args.get("thoughts", "")
        action_summary = f"{name}({json.dumps(clean_args)})"
        nl_desc = f"{thoughts}\n{action_summary}" if thoughts else action_summary
        action = DPAction(
            action_name=name,
            content={"action": name, "arguments": dict(args)},
            action_nl_description=nl_desc,
            llm_conversation=LLMConversation(
                messages=[DPLLMMessage(raw_response=raw_response, reasoning=thoughts)]
            ),
            agent_state=AgentState(agent=self.name),
        )
        run_context.add_action(action)
        return action

    async def _save_screenshot(self, env, output_dir, filename: str) -> None:
        if output_dir:
            screenshot_bytes = await env.get_observation()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / filename).write_bytes(screenshot_bytes)

    async def _get_scaled_screenshot(self, env) -> Image.Image:
        """Get current screenshot and scale it for the model."""
        screenshot_bytes = await env.get_observation()
        screenshot = Image.open(io.BytesIO(screenshot_bytes))
        _, scaled_screenshot = self._get_system_message(screenshot)
        return scaled_screenshot

    def _get_system_message(
        self, screenshot: Image.Image
    ) -> Tuple[List[SystemMessage], Image.Image]:
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
        for msg in system_prompt_info["conversation"]:
            tmp_content = ""
            for content in msg["content"]:
                tmp_content += content["text"]
            system_message.append(SystemMessage(content=tmp_content))

        return system_message, scaled_screenshot

    def _parse_thoughts_and_action(self, message: str) -> Tuple[str, Dict[str, Any]]:
        try:
            tmp = message.split("<tool_call>\n")
            thoughts = tmp[0].strip()
            action_text = tmp[1].split("\n</tool_call>")[0]
            try:
                action = json.loads(action_text)
            except json.decoder.JSONDecodeError:
                self.logger.error(f"Invalid action text: {action_text}")
                action = ast.literal_eval(action_text)
            return thoughts, action
        except Exception as e:
            self.logger.error(
                f"Error parsing thoughts and action: {message}", exc_info=True
            )
            if self.config.terminate_on_parse_error:
                self.logger.warning(
                    "terminate_on_parse_error=true: ending trajectory with raw "
                    "model response as final answer"
                )
                return message.strip(), {
                    "name": "computer_use",
                    "arguments": {
                        "action": "terminate",
                        "answer": message.strip(),
                    },
                }
            raise e

    def convert_resized_coords_to_original(
        self, coords: List[float], rsz_w: int, rsz_h: int, og_w: int, og_h: int
    ) -> List[float]:
        scale_x = og_w / rsz_w
        scale_y = og_h / rsz_h
        return [coords[0] * scale_x, coords[1] * scale_y]

    def proc_coords(
        self,
        coords: List[float] | None,
        im_w: int,
        im_h: int,
        og_im_w: int | None = None,
        og_im_h: int | None = None,
    ) -> List[float] | None:
        if not coords:
            return coords
        if og_im_w is None:
            og_im_w = im_w
        if og_im_h is None:
            og_im_h = im_h
        tgt_x, tgt_y = coords
        return self.convert_resized_coords_to_original(
            [tgt_x, tgt_y], im_w, im_h, og_im_w, og_im_h
        )

    @retry(
        retry=retry_if_not_exception_type(openai.BadRequestError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=5.0, min=5.0, max=60),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    async def _make_model_call(
        self,
        history: List[LLMMessage],
        extra_create_args: Dict[str, Any] | None = None,
    ) -> str:
        """Make a model call using the client."""
        if self._client is None:
            raise ValueError(
                "No client configured for FaraAgent. Call initialize() first."
            )

        result = await self._client.create(
            messages=history,
            extra_create_args=extra_create_args or {},
        )
        return result.content.content

    def remove_screenshot_from_message(self, msg: LLMMessage) -> LLMMessage | None:
        """Remove the screenshot from the message content."""
        if isinstance(msg.content, list):
            new_content = [c for c in msg.content if not isinstance(c, ImageObj)]
            msg.content = new_content
            return msg
        elif isinstance(msg.content, ImageObj):
            return None
        return msg

    def maybe_remove_old_screenshots(
        self, history: List[LLMMessage], includes_current: bool = False
    ) -> List[LLMMessage]:
        """Remove old screenshots from the chat history."""
        if self.config.max_n_images <= 0:
            return history

        max_n_images = (
            self.config.max_n_images
            if includes_current
            else self.config.max_n_images - 1
        )
        new_history: List[LLMMessage] = []
        n_images = 0

        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            meta = (msg.metadata or {}) if isinstance(msg, UserMessage) else {}
            preserve_text = meta.get("is_original", False) or meta.get(
                "is_user_response", False
            )

            if i == 0 and n_images >= max_n_images:
                msg = self.remove_screenshot_from_message(msg)
                if msg is None:
                    continue

            if isinstance(msg.content, list):
                has_image = any(isinstance(c, ImageObj) for c in msg.content)
                if has_image:
                    if n_images < max_n_images:
                        new_history.append(msg)
                    elif preserve_text:
                        msg = self.remove_screenshot_from_message(msg)
                        if msg is not None:
                            raw = meta.get("user_response")
                            if raw is not None:
                                msg.content = [raw]
                            new_history.append(msg)
                    n_images += 1
                else:
                    new_history.append(msg)
            elif isinstance(msg.content, ImageObj):
                if n_images < max_n_images:
                    new_history.append(msg)
                n_images += 1
            else:
                new_history.append(msg)

        return new_history[::-1]

    def _estimate_prompt_tokens(self, history: List[LLMMessage]) -> int:
        total = 0
        for msg in history:
            items = msg.content if isinstance(msg.content, list) else [msg.content]
            for c in items:
                if isinstance(c, ImageObj):
                    total += self.config.image_token_estimate
                elif isinstance(c, str):
                    total += len(c) // 4 + 1
        return total

    @staticmethod
    def _msg_has_image(msg: LLMMessage) -> bool:
        c = msg.content
        if isinstance(c, list):
            return any(isinstance(x, ImageObj) for x in c)
        return isinstance(c, ImageObj)

    def _fit_images_to_budget(self, history: List[LLMMessage]) -> List[LLMMessage]:
        """Drop oldest screenshots (keeping the most recent) until the estimated
        prompt is within ``image_budget_token_cap``. Returns a new list; the
        original messages in ``chat_history`` are left unchanged."""
        cap = self.config.image_budget_token_cap
        if cap <= 0 or self._estimate_prompt_tokens(history) <= cap:
            return history
        out = list(history)
        img_idxs = [i for i, m in enumerate(out) if self._msg_has_image(m)]
        for i in img_idxs[:-1]:
            msg = copy.copy(out[i])
            msg.content = [x for x in msg.content if not isinstance(x, ImageObj)]
            out[i] = msg
            if self._estimate_prompt_tokens(out) <= cap:
                break
        return out

    async def _generate_model_call(
        self,
        env,
        is_first_round: bool,
        first_screenshot: Image.Image | None = None,
        user_response: str = "",
    ) -> Tuple[List[FunctionCall], str]:
        history = self.maybe_remove_old_screenshots(self._state.chat_history)

        screenshot_for_system = first_screenshot
        if not is_first_round:
            scaled_screenshot = await self._get_scaled_screenshot(env)
            screenshot_for_system = scaled_screenshot

            curr_url = (await env.get_page_context()).url
            if curr_url:
                trimmed_url = get_trimmed_url(curr_url, self.MAX_URL_LENGTH)
                url_prefix = f"Current URL: {trimmed_url}\n"
            else:
                url_prefix = ""
            observation_prefix = self._get_observation_prefix()
            if user_response:
                text_prompt = f"{url_prefix}{user_response}"
            elif observation_prefix:
                text_prompt = f"{url_prefix}{observation_prefix}\n{self.USER_MESSAGE}"
            else:
                text_prompt = f"{url_prefix}{self.USER_MESSAGE}"

            metadata = (
                {"is_user_response": True, "user_response": user_response}
                if user_response
                else None
            )
            curr_message = UserMessage(
                content=[ImageObj.from_pil(scaled_screenshot), text_prompt],
                metadata=metadata,
            )
            self._state.chat_history.append(curr_message)
            history.append(curr_message)

        system_message, _ = self._get_system_message(screenshot_for_system)
        history = system_message + history
        history = self._fit_images_to_budget(history)

        call_args: dict[str, Any] = {"temperature": 0}
        if self.config.extra_create_args:
            call_args.update(self.config.extra_create_args)
        message = await self._make_model_call(history, extra_create_args=call_args)

        self._state.chat_history.append(AssistantMessage(content=message))
        thoughts, action = self._parse_thoughts_and_action(message)
        action["arguments"]["thoughts"] = thoughts
        function_call = [FunctionCall(id="dummy", **action)]
        return function_call, message

    async def _execute_action(
        self, env, function_call: list[FunctionCall]
    ) -> tuple[bool, str]:
        """Execute an action on the environment."""
        name = function_call[0].name
        args = function_call[0].arguments
        action_description = ""

        self.logger.debug(
            WebSurferEvent(
                source="FaraAgent",
                url=(await env.get_page_context()).url,
                action=name,
                arguments=args,
                message=f"{name}( {json.dumps(args)} )",
            )
        )

        if "coordinate" in args:
            args["coordinate"] = self.proc_coords(
                args["coordinate"],
                self.DISPLAY_SIZE,
                self.DISPLAY_SIZE,
                self.config.viewport_width,
                self.config.viewport_height,
            )

        is_stop_action = False
        action_type = args["action"]

        if action_type == "visit_url":
            url = str(args["url"])
            action_description = f"I typed '{url}' into the browser address bar."
            if url.startswith(("https://", "http://", "file://", "about:")):
                await env.goto(url)
            elif " " in url:
                await env.goto(
                    f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH"
                )
            else:
                await env.goto("https://" + url)

        elif action_type == "history_back":
            action_description = "I clicked the browser back button."
            await env.back()

        elif action_type == "web_search":
            query = args.get("query")
            action_description = f"I typed '{query}' into the browser search bar."
            encoded_query = quote_plus(query)
            await env.goto(f"https://www.bing.com/search?q={encoded_query}&FORM=QBLH")

        elif action_type == "scroll":
            pixels = int(args.get("pixels", 0))
            if pixels > 0:
                action_description = "I scrolled up one page in the browser."
                await env.scroll_up()
            elif pixels < 0:
                action_description = "I scrolled down one page in the browser."
                await env.scroll_down()

        elif action_type in ("keypress", "key"):
            keys = args.get("keys", [])
            action_description = f"I pressed the following keys: {keys}"
            await env.keypress(keys)

        elif action_type in ("hover", "mouse_move"):
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                await env.hover(tgt_x, tgt_y)

        elif action_type in ("sleep", "wait"):
            duration = args.get("duration", args.get("time", 3.0))
            action_description = (
                "I am waiting a short period of time before taking further action."
            )
            await env.wait(duration)

        elif action_type in ("click", "left_click"):
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I clicked at coordinates ({tgt_x}, {tgt_y})."
                await env.click(tgt_x, tgt_y)

        elif action_type in ("input_text", "type"):
            text_value = args.get("text", args.get("text_value"))
            if text_value is None:
                raise ValueError(
                    "input_text/type action requires 'text' or 'text_value' argument"
                )
            text_value = str(text_value)
            action_description = f"I typed '{text_value}'."
            press_enter = args.get("press_enter", True)
            delete_existing = args.get("delete_existing_text", False)

            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                await env.type_text(
                    tgt_x,
                    tgt_y,
                    text_value,
                    press_enter=press_enter,
                    clear_first=delete_existing,
                )

        elif action_type == "pause_and_memorize_fact":
            fact = str(args.get("fact"))
            self._state.facts.append(fact)
            action_description = f"I memorized the following fact: {fact}"

        elif action_type in ("stop", "terminate"):
            action_description = args.get("thoughts", "Task terminated")
            is_stop_action = True

        else:
            raise ValueError(f"Unknown action: {action_type}")

        if not is_stop_action:
            await env.wait_for_load()

        self._state.num_actions += 1
        return is_stop_action, action_description
