"""Base classes for pixel-level GUI environments.

Defines the canonical action interface from action_space.md:
- ComputerEnvironment: desktop-level GUI actions (click, type, scroll, etc.)
- BrowserEnvironment: adds browser navigation (goto_url, go_back, refresh)
"""

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum
from typing import NamedTuple

from pydantic import model_validator

from .base import Environment, EnvironmentConfig
from .screen_resolutions import sample_random_screen_resolution


class OSType(StrEnum):
    """Operating system type for computer environments."""

    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"


class PageContext(NamedTuple):
    """URL and viewport info returned by get_page_context."""

    url: str
    page_info: str


class ComputerEnvironmentConfig(EnvironmentConfig):
    """Configuration for computer (GUI) environments."""

    viewport_width: int = 1920
    viewport_height: int = 1080
    randomize_screen_res: bool = False

    @model_validator(mode="after")
    def _apply_randomize_screen_res(self) -> "ComputerEnvironmentConfig":
        if self.randomize_screen_res:
            w, h = sample_random_screen_resolution()
            self.viewport_width = w
            self.viewport_height = h
        return self


class ComputerEnvironment(Environment):
    """Base class for pixel-level GUI environments (desktop or browser).

    Subclasses must implement all core GUI actions. Extended actions
    raise NotImplementedError by default and can be overridden.
    """

    @property
    @abstractmethod
    def os_type(self) -> OSType:
        """The OS this environment runs on. Concrete subclasses must set this."""
        ...

    async def save_state(self) -> dict:
        """Persist the resolved viewport so prepro gets the actual (possibly
        randomized) dims rather than the raw seed values."""
        cfg: ComputerEnvironmentConfig = self.config  # type: ignore[assignment]
        return {
            "viewport_width": cfg.viewport_width,
            "viewport_height": cfg.viewport_height,
        }

    # --- Core GUI actions (required) ---

    @abstractmethod
    async def left_click(self, x: int, y: int) -> None: ...

    @abstractmethod
    async def right_click(self, x: int, y: int) -> None: ...

    @abstractmethod
    async def double_click(self, x: int, y: int) -> None: ...

    @abstractmethod
    async def mouse_move(self, x: int, y: int) -> None: ...

    @abstractmethod
    async def left_click_drag(self, end_x: int, end_y: int) -> None:
        """Drag from current cursor position to (end_x, end_y)."""
        ...

    async def drag_from_to(
        self, start_x: int, start_y: int, end_x: int, end_y: int
    ) -> None:
        """Move to (start_x, start_y), then drag to (end_x, end_y)."""
        await self.mouse_move(start_x, start_y)
        await self.left_click_drag(end_x, end_y)

    @abstractmethod
    async def key(self, keys: list[str]) -> None: ...

    @abstractmethod
    async def type(self, text: str) -> None: ...

    @abstractmethod
    async def scroll(self, pixels: int) -> None:
        """Scroll vertically. Positive=up, negative=down."""
        ...

    @abstractmethod
    async def wait(self, duration: float) -> None: ...

    @abstractmethod
    async def get_screenshot(self) -> bytes: ...

    async def get_screenshot_with_retry(
        self, retries: int = 3, delay: float = 5
    ) -> bytes:
        """Get screenshot with retries for transient environment errors."""
        import asyncio

        for attempt in range(retries):
            try:
                return await self.get_screenshot()
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"RETRYABLE_ERROR: screenshot failed after {retries} attempts: {e}"
                    ) from e
                await asyncio.sleep(delay)

    # --- Extended actions (optional) ---

    async def middle_click(self, x: int, y: int) -> None:
        raise NotImplementedError

    async def triple_click(self, x: int, y: int) -> None:
        raise NotImplementedError

    async def left_mouse_down(self, x: int, y: int) -> None:
        raise NotImplementedError

    async def left_mouse_up(self, x: int, y: int) -> None:
        raise NotImplementedError

    async def hold_key(self, key: str, duration: float) -> None:
        raise NotImplementedError

    async def cursor_position(self) -> tuple[int, int]:
        raise NotImplementedError

    async def hscroll(self, pixels: int) -> None:
        """Horizontal scroll. Positive=right, negative=left."""
        raise NotImplementedError

    async def screenshot_region(self, x0: int, y0: int, x1: int, y1: int) -> bytes:
        raise NotImplementedError

    async def get_page_context(self) -> PageContext:
        """Return page URL and text summary of page state.

        Returns empty PageContext by default. Browser environments override this.
        """
        return PageContext(url="", page_info="")


class BrowserEnvironmentConfig(ComputerEnvironmentConfig):
    """Configuration for browser-based environments."""

    start_page: str = "about:blank"
    extra_sleep_time: float = 3.0


class BrowserEnvironment(ComputerEnvironment):
    """ComputerEnvironment extended with browser navigation actions."""

    @abstractmethod
    async def goto_url(self, url: str) -> None: ...

    @abstractmethod
    async def go_back(self) -> None: ...

    async def refresh(self) -> None:
        raise NotImplementedError("refresh not supported by this environment")
