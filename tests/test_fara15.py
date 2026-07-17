"""Tests for the Fara-1.5 agent port: prompts, action space, parsing, DataPoint."""

import json

import pytest
from PIL import Image

from fara import (
    FARA15_ACTION_DEFINITIONS,
    FARA_ACTION_DEFINITIONS,
    Fara15Agent,
    FaraAgent,
)
from fara._prompts import (
    CRITICAL_POINTS_REGISTRY,
    FaraBrowserComputerUse,
    IDENTITY_REGISTRY,
    get_computer_use_system_prompt,
)
from fara.data_point import (
    Action,
    ComputerObservation,
    DataPoint,
    SolverStatus,
    Task,
    ToolOutput,
)


def build_prompt(**kwargs):
    image = Image.new("RGB", (1440, 900))
    return get_computer_use_system_prompt(
        image, Fara15Agent.MLM_PROCESSOR_IM_CFG, display_size=1000, **kwargs
    )


def prompt_text(out):
    return "".join(c["text"] for msg in out["conversation"] for c in msg["content"])


def test_action_definitions_match_tool_schema():
    schema_actions = FaraBrowserComputerUse.parameters["properties"]["action"]["enum"]
    assert set(schema_actions) == set(FARA15_ACTION_DEFINITIONS)
    # Every non-action argument referenced by the definitions exists in the schema
    schema_args = set(FaraBrowserComputerUse.parameters["properties"]) - {"action"}
    for action, arg_names in FARA15_ACTION_DEFINITIONS.items():
        assert arg_names <= schema_args, action


def test_system_prompt_contents():
    out = build_prompt()
    text = prompt_text(out)
    assert len(out["conversation"]) == 1
    assert out["conversation"][0]["role"] == "system"
    assert text.startswith("You are Fara, a computer use agent")
    assert "Qwen3.5-9B" in text
    assert "A critical point is a situation" in text
    assert "The screen's resolution is 1000x1000" in text
    assert "<tool_call>" in text
    for action in FARA15_ACTION_DEFINITIONS:
        assert f"`{action}`" in text
    # 1440x900 resized for patch 16 / merge 2 (factor 32)
    assert out["im_size"] == (1440, 896)


def test_system_prompt_identity_variants():
    out = build_prompt(identity="fara_qwen3vl")
    assert "Qwen3-VL-8B-Instruct" in prompt_text(out)
    with pytest.raises(ValueError):
        build_prompt(identity="bogus")
    with pytest.raises(ValueError):
        build_prompt(critical_points="bogus")
    assert set(IDENTITY_REGISTRY) == {"fara_qwen35", "fara_qwen3vl"}
    assert set(CRITICAL_POINTS_REGISTRY) == {"fara-1.5"}


def test_fara15_action_space_is_superset_of_fara7b():
    # fara-1.0 space minus renamed/removed args is contained in fara-1.5
    assert set(FARA_ACTION_DEFINITIONS) - {"terminate"} <= set(
        FARA15_ACTION_DEFINITIONS
    )
    # terminate changed: status enum -> free-form answer
    assert FARA15_ACTION_DEFINITIONS["terminate"] == {"answer"}


def test_parse_thoughts_and_action():
    agent = Fara15Agent.__new__(Fara15Agent)
    import logging

    agent.logger = logging.getLogger("test")
    action_json = json.dumps(
        {
            "name": "computer_use",
            "arguments": {"action": "left_click", "coordinate": [500, 500]},
        }
    )
    message = f"I should click the button.\n<tool_call>\n{action_json}\n</tool_call>"
    thoughts, action = agent._parse_thoughts_and_action(message)
    assert thoughts == "I should click the button."
    assert action["arguments"]["action"] == "left_click"


def test_proc_coords_scales_from_display_space_to_viewport():
    agent = Fara15Agent.__new__(Fara15Agent)
    agent.viewport_width = 1440
    agent.viewport_height = 900
    assert agent.proc_coords([500, 500]) == [720.0, 450.0]
    assert agent.proc_coords([1000, 1000]) == [1440.0, 900.0]
    assert agent.proc_coords(None) is None


def test_data_point_roundtrip(tmp_path):
    dp = DataPoint(task=Task(task_id="t1", instruction="find a hotel"))
    dp.solver_log.add_observation(
        ComputerObservation(screenshot_path="screenshot0.png", url="https://bing.com")
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
    assert steps[0].observations_post.main[0].output == "I clicked at coordinates (1, 2)."


def test_fara7b_agent_still_importable():
    assert FaraAgent.__name__ == "FaraAgent"
    assert "terminate" in FARA_ACTION_DEFINITIONS
    from fara.fara_7b import FaraAgent as Fara7B

    assert Fara7B is FaraAgent
