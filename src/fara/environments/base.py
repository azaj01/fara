"""Environment base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


class EnvironmentConfig(BaseModel):
    """Base configuration for environments."""

    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any] = {}


class Environment(ABC):
    """Base class for all environments."""

    def __init__(
        self, config: EnvironmentConfig | dict[str, Any] | None = None, **kwargs: Any
    ):
        if config is None:
            config = self._get_config_class().model_validate(kwargs)
        elif isinstance(config, dict):
            config = self._get_config_class().model_validate(config)
        self.config = config
        self._initialized = False

    @classmethod
    def _get_config_class(cls) -> type[EnvironmentConfig]:
        """Override in subclasses to return the specific config class."""
        return EnvironmentConfig

    @abstractmethod
    async def initialize(self, **kwargs) -> None:
        """Initialize the environment."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the environment and cleanup resources."""
        ...

    @abstractmethod
    async def get_observation(self) -> Any:
        """Get current observation from the environment."""
        ...

    async def is_alive(self) -> bool:
        """Check if the environment is still responsive."""
        raise NotImplementedError

    async def reinitialize(self) -> None:
        """Reset the environment by closing and re-initializing."""
        raise NotImplementedError

    async def save_state(self) -> Dict[str, Any]:
        """Save environment state for resuming."""
        return {}

    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load environment state when resuming."""
        pass

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def is_initialized(self) -> bool:
        return self._initialized
