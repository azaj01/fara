"""Agent base class - replaces autogen dependency."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from .data_point import ComputerObservation, ToolOutput, UserMessage as UserMessageObs
from .run_context import RunContext


class AgentConfig(BaseModel):
    """Base configuration for agents."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str = "agent"
    client: Any = None
    metadata: dict[str, Any] = {}


class Agent(ABC):
    """
    Base class for all agents.

    Replaces autogen dependency with a simple interface:
    - initialize: setup the agent
    - run: execute the agent's logic
    - close: cleanup

    Agents are responsible for logging their actions to run_context.data_point.solver_log
    """

    def __init__(
        self, config: AgentConfig | dict[str, Any] | None = None, **kwargs: Any
    ):
        if config is None:
            config = self._get_config_class().model_validate(kwargs)
        elif isinstance(config, dict):
            config = self._get_config_class().model_validate(config)
        self.config = config
        self._initialized = False

    @classmethod
    def _get_config_class(cls) -> type[AgentConfig]:
        """Override in subclasses to return the specific config class."""
        return AgentConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def client(self) -> Any:
        return self.config.client

    async def initialize(self, run_context: RunContext) -> None:
        """
        Initialize the agent for a run.

        Override this to setup any resources needed.
        Called before run().

        Args:
            run_context: The shared run context
        """
        self._initialized = True

    @abstractmethod
    async def run(
        self,
        run_context: RunContext,
        input: Any = None,
    ) -> Any:
        """
        Execute the agent's main logic.

        Args:
            run_context: Shared context with environment and data point
            input: Optional input (e.g., Task for orchestrator)

        Returns:
            Agent-specific output
        """
        ...

    async def close(self, run_context: RunContext) -> None:
        """
        Cleanup after the agent is done.

        Override this to release any resources.
        Called after run() completes or on error.

        Args:
            run_context: The shared run context
        """
        self._initialized = False

    def _log_observation(
        self,
        run_context: RunContext,
        output: str,
        screenshot: str,
        page_info: str,
        error: str = "",
        action_id: str | None = None,
        url: str = "",
    ) -> None:
        """Log post-action observations (ToolOutput, ComputerObservation) to the solver log."""
        run_context.add_observation(
            ToolOutput(output=output, error=error, action_id=action_id)
        )
        if screenshot or page_info:
            run_context.add_observation(
                ComputerObservation(
                    screenshot_path=screenshot,
                    url=url,
                    page_info=page_info,
                    action_id=action_id,
                )
            )

    def _log_initial_observations(
        self,
        run_context: RunContext,
        task: str,
        screenshot_path: str = "screenshot_0_pre.png",
    ) -> None:
        """Log the initial task message and screenshot at the start of a fresh run."""
        run_context.add_observation(UserMessageObs(content=task))
        run_context.add_observation(
            ComputerObservation(screenshot_path=screenshot_path)
        )
        if run_context.output_dir:
            post_path = screenshot_path.replace("_pre.", "_post.")
            shutil.copyfile(
                run_context.output_dir / screenshot_path,
                run_context.output_dir / post_path,
            )

    def save_state(self, run_context: RunContext) -> None:
        """
        Save agent-specific state for resuming.

        Override to save any state needed to resume mid-run.

        Args:
            run_context: The shared run context
        """
        pass

    def load_state(self, run_context: RunContext) -> None:
        """
        Load agent-specific state when resuming.

        Override to restore state from a previous run.

        Args:
            run_context: The shared run context
        """
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
