"""Client factory. Builds a ChatCompletionClient from an endpoint config dict."""

from typing import Any, Dict

from .wrapper import ChatCompletionClient


def create_client_from_config(config: Dict[str, Any]) -> ChatCompletionClient:
    """Create a ChatCompletionClient from a ``{model, base_url, api_key}`` dict."""
    return ChatCompletionClient(
        model=config.get("model", "Fara1.5-9B"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
    )
