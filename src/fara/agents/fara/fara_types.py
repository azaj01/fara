"""Fara-specific types. Most types are imported from aztool.clients."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WebSurferEvent:
    """Event for logging web surfing actions."""

    source: str
    message: str
    url: str
    action: str | None = None
    arguments: Dict[str, Any] | None = None
