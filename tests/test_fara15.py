"""Tests for the Fara-1.5 agent (Fara15Agent).

Covers the two things the port must get exactly right: the system prompt
the model sees (identity + critical points + tool schema), and the
DataPoint trajectory format.
"""

import json
import logging

from PIL import Image

from fara import DataPoint, Fara15Agent, Task
from fara.agents.fara.fara15_agent import (
    Fara15AgentConfig,
    extract_allowed_actions,
)
from fara.agents.fara._prompts import (
    CRITICAL_POINTS_REGISTRY,
    IDENTITY_REGISTRY,
)
from fara.core.data_point import (
    Action,
    ComputerObservation,
    SolverStatus,
    ToolOutput,
)

BROWSER_ACTIONS = {
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
}


def _make_agent(identity="fara_qwen35"):
    agent = Fara15Agent(
        Fara15AgentConfig(
            client_config={"model": "m", "base_url": "u", "api_key": "k"},
            identity=identity,
        )
    )
    agent._state = type("st", (), {"mlm_width": 0, "mlm_height": 0})()
    return agent


def _system_text(agent):
    sysmsgs, _ = agent._get_system_message(Image.new("RGB", (1440, 900)))
    return "".join(m.content for m in sysmsgs)


def test_registries():
    assert set(IDENTITY_REGISTRY) == {"fara_qwen35", "fara_qwen3vl"}
    assert set(CRITICAL_POINTS_REGISTRY) == {"fara-1.5"}


def test_system_prompt_contents_and_action_space():
    text = _system_text(_make_agent())
    assert text.startswith("You are Fara, a computer use agent")
    assert "Qwen3.5-9B" in text  # identity fara_qwen35
    assert "A critical point is a situation" in text  # critical points 1.5
    assert "The screen's resolution is 1000x1000" in text  # fixed coord space
    assert extract_allowed_actions(text) == BROWSER_ACTIONS


def test_identity_variant():
    assert "Qwen3-VL-8B-Instruct" in _system_text(_make_agent("fara_qwen3vl"))


def test_system_prompt_byte_identical_to_training():
    """The generated prompt must match agento_train's exactly — the model was
    trained on that string. Rebuilds the expected prompt from the vendored
    identity/critical-points/tool schema and compares byte-for-byte."""
    from fara.agents.fara._prompts import (
        FN_CALL_FORMAT,
        FaraBrowserComputerUse,
        FARA_QWEN35_IDENTITY,
        CRITICAL_POINTS_FARA_1_5,
    )

    tool = FaraBrowserComputerUse(
        cfg={"display_width_px": 1000, "display_height_px": 1000}
    )
    tool_descs = json.dumps(
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        },
        ensure_ascii=False,
    )
    template = (
        FARA_QWEN35_IDENTITY + "\n\n" + CRITICAL_POINTS_FARA_1_5 + "\n\n" + FN_CALL_FORMAT
    )
    expected = template.format(tool_descs=tool_descs)
    assert _system_text(_make_agent()) == expected


def test_proc_coords_scales_display_space_to_viewport():
    agent = _make_agent()
    # DISPLAY_SIZE=1000 -> viewport 1440x900
    assert agent._proc([500, 500]) == [720.0, 450.0]
    assert agent._proc([1000, 1000]) == [1440.0, 900.0]


def test_parse_thoughts_and_action():
    agent = Fara15Agent.__new__(Fara15Agent)
    agent.logger = logging.getLogger("test")
    action_json = json.dumps(
        {
            "name": "computer_use",
            "arguments": {"action": "left_click", "coordinate": [500, 500]},
        }
    )
    message = f"I should click.\n<tool_call>\n{action_json}\n</tool_call>"
    thoughts, action = agent._parse_thoughts_and_action(message)
    assert thoughts == "I should click."
    assert action["arguments"]["action"] == "left_click"


def test_data_point_roundtrip(tmp_path):
    dp = DataPoint(task=Task(task_id="t1", instruction="find a hotel"))
    dp.solver_log.add_observation(
        ComputerObservation(screenshot_path="screenshot_1_pre.png", url="https://bing.com")
    )
    action = Action(
        action_name="left_click",
        content={"action": "left_click", "arguments": {"coordinate": [1, 2]}},
    )
    dp.solver_log.add_action(action)
    dp.solver_log.add_observation(
        ToolOutput(output="I clicked at coordinates (1, 2).", action_id=action.id)
    )
    dp.solver_log.status = SolverStatus.COMPLETE

    path = tmp_path / "data_point.json"
    dp.save(path)
    loaded = DataPoint.load(path)
    assert loaded.task.instruction == "find a hotel"
    assert loaded.solver_log.status == SolverStatus.COMPLETE
    steps = loaded.solver_log.steps()
    assert len(steps) == 1
    assert steps[0].action.action_name == "left_click"


def test_fara7b_still_importable():
    from fara import FaraAgent, FARA_ACTION_DEFINITIONS

    assert FaraAgent.__name__ == "FaraAgent"
    assert "terminate" in FARA_ACTION_DEFINITIONS
