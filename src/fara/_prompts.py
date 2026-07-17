"""Fara-1.5 browser toolset and system prompt.

Ported from agento_train's FaraNextBrowserComputerUse toolset and the
composable fara-qwen3vl fn-call template (identity + critical points).
"""

from typing import Union

from .qwen_helpers.base_tool import BaseTool
from .qwen_helpers.fncall_prompt import NousFnCallPrompt
from .qwen_helpers.utils import smart_resize

FARA_QWEN35_IDENTITY = """\
You are Fara, a computer use agent (CUA) specialized for web browsers. \
You are developed by Microsoft AI Frontiers. You assist users with \
completing and automating tasks that require the use of a web browser.

The model was trained in the timeframe of January - April 2026. You can \
effectively perform tasks even beyond this range by accessing the web \
browser and using the latest information on the live web. But your \
knowledge cutoff is limited to early 2026, so you may not be aware of \
events or developments that occurred after that time, without explicitly \
browsing and searching for latest information on the web.

This edition of the model was trained using SFT on top of \
Qwen3.5-9B, using a synthetic data mixture generated and \
developed by Microsoft AI Frontiers."""

FARA_QWEN3VL_IDENTITY = """\
You are Fara, a computer use agent (CUA) specialized for web browsers. \
You are developed by Microsoft AI Frontiers. You assist users with \
completing and automating tasks that require the use of a web browser.

The model was trained in the timeframe of January - March 2026. You can \
effectively perform tasks even beyond this range by accessing the web \
browser and using the latest information on the live web. But your \
knowledge cutoff is limited to early 2026, so you may not be aware of \
events or developments that occurred after that time, without explicitly \
browsing and searching for latest information on the web.

This edition of the model was trained using SFT on top of \
Qwen3-VL-8B-Instruct, using a synthetic data mixture generated and \
developed by Microsoft AI Frontiers."""

CRITICAL_POINTS_FARA_1_5 = """\
A critical point is a situation where we must pause and request information or confirmation from the user before \
proceeding. There are three types:

Case 1: Missing User Information — The task requires personal information that the user has not provided (e.g., email, \
phone number, address, payment details). Never fabricate or assume personal information. Fill in only what the user has \
explicitly provided, then pause and ask for any missing required fields. (e.g., form requires phone number but user \
only gave name and email -> fill name and email, then ask for phone number.) If the user has provided all required \
information, proceed without stopping.

Case 2: Underspecified Task — The task description is ambiguous or missing details needed to make a decision at the \
current step. Pause and ask for clarification. (e.g., user asks to book a flight but doesn't specify destination -> \
ask for destination.) If the user's instructions contain all information needed for the current decision, proceed \
without stopping.

Case 3: Irreversible Action — We are about to perform an action that cannot be undone (e.g., submitting a form, \
completing a purchase, sending a message, deleting data). If the user explicitly authorized the action (e.g., "submit \
the form", "complete the purchase", "you have my permission to submit") -> proceed without stopping. If the user did \
NOT explicitly authorize the action -> stop and ask for confirmation. (e.g., "fill out a form" with no mention of \
submitting -> fill the form, then ask before submitting; "fill out and submit a form" -> fill and submit without \
stopping.)

Only stop at a critical point if (1) required information is missing, (2) the task is ambiguous, OR (3) an irreversible \
action lacks explicit user authorization. If the user has provided all necessary information AND explicitly authorized \
the action, proceed without interruption."""

IDENTITY_REGISTRY = {
    "fara_qwen35": FARA_QWEN35_IDENTITY,
    "fara_qwen3vl": FARA_QWEN3VL_IDENTITY,
}

CRITICAL_POINTS_REGISTRY = {
    "fara-1.5": CRITICAL_POINTS_FARA_1_5,
}

FN_CALL_FORMAT = (
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "{tool_descs}\n"
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
    "</tool_call>"
)


def build_fara_fn_call_template(identity: str, critical_points: str) -> str:
    """Compose a Fara fn-call template from independent identity + critical points.

    Args:
        identity: Key in IDENTITY_REGISTRY or a raw string (>= 20 chars).
        critical_points: Key in CRITICAL_POINTS_REGISTRY or a raw string (>= 20 chars).

    Returns:
        Fully composed template string with {tool_descs} placeholder.
    """
    if identity in IDENTITY_REGISTRY:
        identity = IDENTITY_REGISTRY[identity]
    elif not isinstance(identity, str) or len(identity) < 20:
        raise ValueError(
            f"Unknown identity key: {identity!r}. "
            f"Available keys: {list(IDENTITY_REGISTRY.keys())}. "
            f"Or pass a raw string (>= 20 chars)."
        )

    if critical_points in CRITICAL_POINTS_REGISTRY:
        critical_points = CRITICAL_POINTS_REGISTRY[critical_points]
    elif not isinstance(critical_points, str) or len(critical_points) < 20:
        raise ValueError(
            f"Unknown critical_points key: {critical_points!r}. "
            f"Available keys: {list(CRITICAL_POINTS_REGISTRY.keys())}. "
            f"Or pass a raw string (>= 20 chars)."
        )

    return identity + "\n\n" + critical_points + "\n\n" + FN_CALL_FORMAT


class FaraBrowserComputerUse(BaseTool):
    """Schema-only computer_use tool definition for the Fara-1.5 browser action space."""

    name = "computer_use"

    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order. Includes "Enter", "Alt", "Shift", "Tab", "Control", "Backspace", "Delete", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageDown", "PageUp", "Shift", etc.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `double_click`: Double-click the left mouse button.
* `right_click`: Click the right mouse button.
* `triple_click`: Triple-click the left mouse button (e.g. to select a line of text).
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `visit_url`: Visit a specified URL.
* `history_back`: Go back to the previous page in the browser history.
* `web_search`: Perform a web search with a specified query.
* `read_page_answer_question`: Read the current page content and answer a question about it.
* `pause_and_memorize_fact`: Pause and memorize a fact for future reference.
* `ask_user_question`: Ask the user a clarifying question and wait for a response.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and provide the final answer.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "double_click",
                    "triple_click",
                    "scroll",
                    "hscroll",
                    "visit_url",
                    "history_back",
                    "web_search",
                    "read_page_answer_question",
                    "pause_and_memorize_fact",
                    "ask_user_question",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required by `action=left_click`, `action=double_click`, `action=right_click`, `action=triple_click`, `action=left_click_drag`, and `action=mouse_move`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.",
                "type": "number",
            },
            "url": {
                "description": "The URL to visit. Required only by `action=visit_url`.",
                "type": "string",
            },
            "query": {
                "description": "The query to search for. Required only by `action=web_search`.",
                "type": "string",
            },
            "fact": {
                "description": "The fact to remember for the future. Required only by `action=pause_and_memorize_fact`.",
                "type": "string",
            },
            "question": {
                "description": "The question to ask. Required by `action=read_page_answer_question` and `action=ask_user_question`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "answer": {
                "description": "The final answer for the task. Required only by `action=terminate`.",
                "type": "string",
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        raise NotImplementedError("FaraBrowserComputerUse is schema-only.")


def get_computer_use_system_prompt(
    image,
    processor_im_cfg,
    mode="fara_next_browser",
    include_input_text_key_args=False,
    fn_call_template="fara-qwen3vl",
    randomize=False,
    display_size=None,
    identity="fara_qwen35",
    critical_points="fara-1.5",
):
    """Build the Fara-1.5 system prompt for a screenshot.

    Args:
        image: PIL image of the current screenshot.
        processor_im_cfg: dict with patch_size, merge_size, min_pixels, max_pixels.
        mode: Tool-set mode. Only ``fara_next_browser`` is supported here (this
            repo ships the browser action space only).
        include_input_text_key_args, fn_call_template, randomize: accepted for
            signature compatibility with the training-time prompt builder;
            unused by the browser tool.
        display_size: Fixed coordinate-space size the model emits coordinates in;
            None uses the resized image dimensions.
        identity: Key in IDENTITY_REGISTRY or a raw identity string.
        critical_points: Key in CRITICAL_POINTS_REGISTRY or a raw string.
    """
    if mode != "fara_next_browser":
        raise NotImplementedError(
            f"computer_use_mode={mode!r} is not supported; this repo ships the "
            f"'fara_next_browser' action space only."
        )
    patch_size = processor_im_cfg["patch_size"]
    merge_size = processor_im_cfg["merge_size"]
    min_pixels = processor_im_cfg["min_pixels"]
    max_pixels = processor_im_cfg["max_pixels"]

    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    display_w = display_size if display_size is not None else resized_width
    display_h = display_size if display_size is not None else resized_height

    computer_use = FaraBrowserComputerUse(
        cfg={
            "display_width_px": display_w,
            "display_height_px": display_h,
        }
    )

    template = build_fara_fn_call_template(identity, critical_points)
    conversation = NousFnCallPrompt(template=template).preprocess_fncall_messages(
        messages=[],
        functions=[computer_use.function],
        lang=None,
    )

    return {
        "conversation": [msg.model_dump() for msg in conversation],
        "im_size": (resized_width, resized_height),
    }
