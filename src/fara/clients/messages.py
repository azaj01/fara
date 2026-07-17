import io
import base64
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict, Optional
from PIL import Image

ToolSchema = Dict[str, Any]
Tool = ToolSchema


@dataclass
class LLMMessage:
    content: str | List[Dict[str, Any]]
    source: str = "user"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMessage(LLMMessage):
    source: str = "system"


@dataclass
class UserMessage(LLMMessage):
    source: str = "user"


@dataclass
class AssistantMessage(LLMMessage):
    source: str = "assistant"


@dataclass
class ImageObj:
    """Image wrapper for handling screenshots and images"""

    image: Image.Image

    @classmethod
    def from_pil(cls, image: Image.Image) -> "ImageObj":
        return cls(image=image)

    def to_base64(self) -> str:
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def resize(self, size: Tuple[int, int]) -> Image.Image:
        """Resize the image"""
        return self.image.resize(size)


@dataclass
class RequestUsage:
    """Usage statistics for a model request"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    num_calls: int = 1

    def __add__(self, other: "RequestUsage") -> "RequestUsage":
        return RequestUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            num_calls=self.num_calls + other.num_calls,
        )

    def get_cost(self, prompt_price: float, completion_price: float) -> float:
        """
        Calculate cost based on token usage and pricing.
        Args:
            prompt_price: Price per prompt token
            completion_price: Price per completion token
        """
        return (self.prompt_tokens * prompt_price) + (
            self.completion_tokens * completion_price
        )


@dataclass
class ModelResponse:
    """Response from model call"""

    content: str
    usage: RequestUsage = field(default_factory=RequestUsage)


@dataclass
class CreateResult:
    """Result from a chat completion request.

    Attributes:
        content: The message object from the API response.
            For OpenAI/Azure: access text via content.content, tool calls via content.tool_calls.
            For Anthropic: access text via content.content[0].text.
        usage: Token usage statistics for the request.
        finish_reason: The reason the model stopped generating ('stop', 'tool_calls', etc.).
        cached: Whether this result was returned from cache.
    """

    content: Any
    usage: RequestUsage = field(default_factory=RequestUsage)
    finish_reason: Optional[str] = None
    cached: bool = False


@dataclass
class FunctionCall:
    """Represents a function call with arguments"""

    id: str
    name: str
    arguments: Dict[str, Any]


def message_to_openai_format(message: LLMMessage) -> Dict[str, Any]:
    """Convert our LLMMessage to OpenAI API format"""
    role = (
        "system"
        if isinstance(message, SystemMessage)
        else "assistant"
        if isinstance(message, AssistantMessage)
        else "user"
    )

    # Handle multimodal content (text + images)
    if isinstance(message.content, list):
        content_parts = []
        for item in message.content:
            if isinstance(item, ImageObj):
                # Convert image to base64 data URL
                base64_image = item.to_base64()
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )
            elif isinstance(item, str):
                content_parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                content_parts.append(item)
        return {"role": role, "content": content_parts}
    else:
        return {"role": role, "content": message.content}
