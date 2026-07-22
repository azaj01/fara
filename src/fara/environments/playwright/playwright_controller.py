import asyncio
import base64
import os
import logging
import functools
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Awaitable,
    cast,
)

from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError, TargetClosedError
from playwright.async_api import Download, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from ._types import (
    InteractiveRegion,
    VisualViewport,
    interactiveregion_from_dict,
    visualviewport_from_dict,
)
from .webpage_text_utils import WebpageTextUtilsPlaywright

# Some of the Code for clicking coordinates and keypresses adapted from https://github.com/openai/openai-cua-sample-app/blob/main/computers/base_playwright.py
# Copyright 2025 OpenAI - MIT License
from ...agents.key_mapping import CUA_KEY_TO_PLAYWRIGHT_KEY

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def handle_target_closed(max_retries: int = 2, timeout_secs: int = 30):
    """
    Decorator to handle TargetClosedError and tunnel connection errors by attempting to recover the page.

    Args:
        max_retries: Maximum number of retry attempts
        timeout_secs: Timeout for page operations during recovery
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = args[0].logger
            page = None
            if len(args) >= 2 and hasattr(
                args[1], "url"
            ):
                page = args[1]

            retries = 0
            last_error = None

            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except (TargetClosedError, PlaywrightError) as e:
                    is_tunnel_error = "net::ERR_TUNNEL_CONNECTION_FAILED" in str(e)
                    is_target_closed = isinstance(
                        e, TargetClosedError
                    ) or "Target page, context or browser has been closed" in str(e)

                    if not (is_tunnel_error or is_target_closed):
                        raise e

                    last_error = e
                    retries += 1

                    if retries > max_retries:
                        raise e

                    if page is None:
                        raise e

                    error_type = (
                        "tunnel connection" if is_tunnel_error else "target closed"
                    )
                    logger.warning(
                        f"{error_type} error in {func.__name__}, attempting recovery (retry {retries}/{max_retries})"
                    )

                    try:
                        await _recover_page(page, timeout_secs, logger)
                        await asyncio.sleep(0.5)
                    except Exception as recovery_error:
                        logger.error(f"Page recovery failed: {recovery_error}")
                        raise e from recovery_error

            raise last_error

        return wrapper

    return decorator


async def _recover_page(page: Page, timeout_secs: int = 30, logger=None) -> None:
    """
    Attempt to recover a closed page by reloading it.

    Args:
        page: The Playwright page object to recover
        timeout_secs: Timeout for recovery operations
    """
    logger = logger or logging.getLogger("playwright_controller")
    try:
        await page.evaluate("1", timeout=1000)
        return
    except Exception:
        pass

    browser = page.context.browser if page.context else None
    if browser is not None and not browser.is_connected():
        raise Exception(
            "playwright_controller._recover_page(): browser disconnected, recovery impossible"
        )

    try:
        await page.evaluate("window.stop()", timeout=2000)
    except Exception:
        pass

    try:
        await page.reload(timeout=timeout_secs * 1000, wait_until="commit")
        logger.info("playwright_controller._recover_page(): Page recovery successful")
    except Exception as e:
        logger.error(f"playwright_controller._recover_page(): Page reload failed: {e}")

        try:
            current_url = page.url
            if current_url and current_url != "about:blank":
                await page.goto(
                    current_url,
                    timeout=timeout_secs * 1000,
                    wait_until="commit",
                )
                logger.info(
                    "playwright_controller._recover_page(): Page recovery via goto successful"
                )
            else:
                raise Exception(
                    "playwright_controller._recover_page(): No valid URL to navigate to"
                )
        except Exception as goto_error:
            raise Exception(
                f"playwright_controller._recover_page(): All recovery methods failed. Reload error: {e}, Goto error: {goto_error}"
            )


def handle_target_closed_with_context(max_retries: int = 2, timeout_secs: int = 30):
    """
    Enhanced decorator that can also handle browser context recreation.
    Use this for critical operations where you have access to the browser context.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = args[0].logger
            page = None
            if len(args) >= 2 and hasattr(args[1], "url"):
                page = args[1]

            retries = 0
            last_error = None

            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except (TargetClosedError, PlaywrightError) as e:
                    is_tunnel_error = "net::ERR_TUNNEL_CONNECTION_FAILED" in str(e)
                    is_target_closed = isinstance(
                        e, TargetClosedError
                    ) or "Target page, context or browser has been closed" in str(e)

                    if not (is_tunnel_error or is_target_closed):
                        raise e

                    last_error = e
                    retries += 1

                    if retries > max_retries:
                        raise e

                    if page is None:
                        raise e

                    error_type = (
                        "tunnel connection" if is_tunnel_error else "target closed"
                    )
                    logger.warning(
                        f"playwright_controller.handle_target_closed_with_context(): {error_type} error in {func.__name__}, attempting enhanced recovery (retry {retries}/{max_retries})"
                    )

                    try:
                        context = page.context
                        browser = context.browser

                        if browser and not browser.is_connected():
                            logger.error(
                                "playwright_controller.handle_target_closed_with_context(): Browser connection lost - cannot recover automatically"
                            )
                            raise e

                        await _recover_page(page, timeout_secs)
                        await asyncio.sleep(0.5)

                    except Exception as recovery_error:
                        logger.error(
                            f"playwright_controller.handle_target_closed_with_context(): Enhanced page recovery failed: {recovery_error}"
                        )
                        raise e from recovery_error

            raise last_error

        return wrapper

    return decorator


class PlaywrightController:
    def __init__(
        self,
        animate_actions: bool = False,
        downloads_folder: Optional[str] = None,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        _download_handler: Optional[Callable[[Download], None]] = None,
        to_resize_viewport: bool = True,
        single_tab_mode: bool = False,
        sleep_after_action: int = 10,
        timeout_load: int = 1,
        logger=None,
    ) -> None:
        """
        A controller for Playwright to interact with web pages.
        animate_actions: If True, actions will be animated.
        downloads_folder: The folder to save downloads to.
        viewport_width: The width of the viewport.
        viewport_height: The height of the viewport.
        _download_handler: A handler for downloads.
        to_resize_viewport: If True, the viewport will be resized.
        single_tab_mode (bool): If True, forces navigation to happen in the same tab rather than opening new tabs/windows.

        """
        self.animate_actions = animate_actions
        self.downloads_folder = downloads_folder
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._download_handler = _download_handler
        self.to_resize_viewport = to_resize_viewport
        self.single_tab_mode = single_tab_mode
        self._sleep_after_action = sleep_after_action
        self._timeout_load = timeout_load
        self.logger = logger or logging.getLogger("playwright_controller")


        self._page_script: str = ""
        self.last_cursor_position: Tuple[float, float] = (0.0, 0.0)
        self._text_utils = WebpageTextUtilsPlaywright()

        with open(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"),
            "rt",
        ) as fh:
            self._page_script = fh.read()

    async def sleep(self, page: Page, duration: Union[int, float]) -> None:
        await asyncio.sleep(duration)

    @handle_target_closed()
    async def get_interactive_rects(self, page: Page) -> Dict[str, InteractiveRegion]:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = cast(
            Dict[str, Dict[str, Any]],
            await page.evaluate("MultimodalWebSurfer.getInteractiveRects();"),
        )

        assert isinstance(result, dict)
        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            assert isinstance(k, str)
            typed_results[k] = interactiveregion_from_dict(result[k])

        return typed_results

    @handle_target_closed()
    async def get_visual_viewport(self, page: Page) -> VisualViewport:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            self.logger.warning("Failed to inject page script for visual viewport")
            return visualviewport_from_dict({})
        return visualviewport_from_dict(
            await page.evaluate("MultimodalWebSurfer.getVisualViewport();")
        )

    @handle_target_closed()
    async def get_focused_rect_id(self, page: Page) -> str:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = await page.evaluate("MultimodalWebSurfer.getFocusedElementId();")
        return str(result)

    @handle_target_closed()
    async def get_page_metadata(self, page: Page) -> Dict[str, Any]:
        assert page is not None
        attempts = 3
        while attempts > 0:
            try:
                try:
                    await self._ensure_page_ready(page)
                    await page.evaluate(self._page_script)
                except Exception:
                    pass
                result = await page.evaluate("MultimodalWebSurfer.getPageMetadata();")
                assert isinstance(result, dict)
                return cast(Dict[str, Any], result)
            except Exception as e:
                self.logger.error(
                    f"Error: playwrite_controller.get_page_metadata: {str(e)},\nattempting again..."
                )
                attempts -= 1
                await asyncio.sleep(0.5)

    @handle_target_closed()
    async def on_new_page(self, page: Page) -> None:
        assert page is not None
        await page.bring_to_front()
        page.on("download", self._download_handler)
        if self.to_resize_viewport and self.viewport_width and self.viewport_height:
            await page.set_viewport_size(
                {"width": self.viewport_width, "height": self.viewport_height}
            )

        await self.sleep(page, 0.2)
        await page.add_init_script(
            path=os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "page_script.js"
            )
        )
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=60000)
        except PlaywrightTimeoutError:
            self.logger.error("WARNING: Page load timeout, page might not be loaded")
            await page.evaluate("window.stop()")

    @handle_target_closed()
    async def _ensure_page_ready(self, page: Page) -> None:
        assert page is not None
        await self.on_new_page(page)

    @handle_target_closed()
    async def get_screenshot(self, page: Page, path: str | None = None) -> bytes:
        """
        Capture a screenshot of the current page.

        Args:
            page (Page): The Playwright page object.
            path (str, optional): The file path to save the screenshot. If None, the screenshot will be returned as bytes. Default: None
        """
        await self._ensure_page_ready(page)
        try:
            screenshot = await page.screenshot(path=path, timeout=15000)
            return screenshot
        except Exception:
            await page.evaluate("window.stop()")
            screenshot = await page.screenshot(path=path, timeout=15000)
            return screenshot

    @handle_target_closed()
    async def back(self, page: Page) -> None:
        await self._ensure_page_ready(page)
        await page.go_back()

    @handle_target_closed()
    async def visit_page(self, page: Page, url: str) -> Tuple[bool, bool]:
        await self._ensure_page_ready(page)
        reset_prior_metadata_hash = False
        reset_last_download = False
        try:
            await page.goto(url, wait_until="commit")
            reset_prior_metadata_hash = True
        except Exception as e_outer:
            if self.downloads_folder and "net::ERR_ABORTED" in str(e_outer):
                async with page.expect_download() as download_info:
                    try:
                        await page.goto(url, wait_until="commit")
                    except Exception as e_inner:
                        if "net::ERR_ABORTED" in str(e_inner):
                            pass
                        else:
                            raise e_inner
                    download = await download_info.value
                    fname = os.path.join(
                        self.downloads_folder, download.suggested_filename
                    )
                    await download.save_as(fname)
                    message = f"<body style=\"margin: 20px;\"><h1>Successfully downloaded '{download.suggested_filename}' to local path:<br><br>{fname}</h1></body>"
                    await page.goto(
                        "data:text/html;base64,"
                        + base64.b64encode(message.encode("utf-8")).decode("utf-8"),
                        wait_until="commit",
                    )
                    reset_last_download = True
            else:
                raise e_outer
        return reset_prior_metadata_hash, reset_last_download

    @handle_target_closed()
    async def page_down(
        self, page: Page, amount: int = 400, full_page: bool = False
    ) -> None:
        await self._ensure_page_ready(page)
        if full_page:
            await page.mouse.wheel(0, self.viewport_height - 50)
        else:
            await page.mouse.wheel(0, amount)

    @handle_target_closed()
    async def page_up(
        self, page: Page, amount: int = 400, full_page: bool = False
    ) -> None:
        await self._ensure_page_ready(page)
        if full_page:
            await page.mouse.wheel(0, -self.viewport_height + 50)
        else:
            await page.mouse.wheel(0, -amount)

    async def gradual_cursor_animation(
        self, page: Page, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> None:
        steps = 20
        for step in range(steps):
            x = start_x + (end_x - start_x) * (step / steps)
            y = start_y + (end_y - start_y) * (step / steps)
            await page.evaluate(f"""
                (function() {{
                    let cursor = document.getElementById('red-cursor');
                    cursor.style.left = '{x}px';
                    cursor.style.top = '{y}px';
                }})();
            """)
            await asyncio.sleep(0.05)

        self.last_cursor_position = (end_x, end_y)

    async def add_cursor_box(self, page: Page, identifier: str) -> None:
        await page.evaluate(f"""
            (function() {{
                let elm = document.querySelector("[__elementId='{identifier}']");
                if (elm) {{
                    elm.style.transition = 'border 0.3s ease-in-out';
                    elm.style.border = '2px solid red';
                }}
            }})();
        """)
        await asyncio.sleep(0.3)

        await page.evaluate("""
            (function() {
                let cursor = document.createElement('div');
                cursor.id = 'red-cursor';
                cursor.style.width = '10px';
                cursor.style.height = '10px';
                cursor.style.backgroundColor = 'red';
                cursor.style.position = 'absolute';
                cursor.style.borderRadius = '50%';
                cursor.style.zIndex = '10000';
                document.body.appendChild(cursor);
            })();
        """)

    async def remove_cursor_box(self, page: Page, identifier: str) -> None:
        await page.evaluate(f"""
            (function() {{
                let elm = document.querySelector("[__elementId='{identifier}']");
                if (elm) {{
                    elm.style.border = '';
                }}
                let cursor = document.getElementById('red-cursor');
                if (cursor) {{
                    cursor.remove();
                }}
            }})();
        """)

    async def _click_with_popup_capture(
        self, page: Page, action: Callable[[], Awaitable[None]]
    ) -> Page | None:
        """Run a click-like action and capture any popup it opens.

        Returns the new page so the caller can promote it to the active
        page (see ``Environment._swap_to_new_page``). The env decides what
        to do with the old page (close it in ``single_tab_mode``, keep it
        otherwise). Returns ``None`` if no popup fires within the timeout.
        """
        new_page: Page | None = None
        try:
            async with page.expect_event("popup", timeout=1000) as page_info:
                await action()
                new_page = await page_info.value
                assert isinstance(new_page, Page)
                await self.on_new_page(new_page)
        except TimeoutError:
            pass
        return new_page

    @handle_target_closed()
    async def click_coords(self, page: Page, x: float, y: float) -> Page | None:
        await self._ensure_page_ready(page)

        if self.animate_actions:
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        return await self._click_with_popup_capture(
            page, lambda: page.mouse.click(x, y, delay=10)
        )

    @handle_target_closed()
    async def click_id(self, page: Page, identifier: str) -> Page | None:
        """
        Returns new page if a new page is opened, otherwise None.
        """
        new_page: Page | None = None
        await self._ensure_page_ready(page)
        selector = f"[__elementId='{identifier}']"
        try:
            await page.wait_for_selector(
                selector, state="visible", timeout=self._timeout_load * 1000
            )
            target = page.locator(selector)
            await target.scroll_into_view_if_needed()
        except TimeoutError:
            raise ValueError(
                f"Element with identifier {identifier} not found or not visible"
            )

        box = await target.bounding_box()
        if not box:
            raise ValueError(
                f"Element with identifier {identifier} is not visible on the page."
            )
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        if self.single_tab_mode:
            await target.evaluate("""
                el => {
                    // Remove target attribute from clicked element and all _blank links/forms
                    el.removeAttribute('target');
                    document.querySelectorAll('a[target=_blank], form[target=_blank]')
                        .forEach(e => e.removeAttribute('target'));
                }
            """)

        download = None
        download_future: asyncio.Task[Download] | None = None

        if self.downloads_folder:
            try:
                download_future = asyncio.create_task(
                    page.wait_for_event(
                        "download", timeout=500
                    )
                )
            except Exception as e:
                self.logger.error(f"Failed to set up download listener: {e}")
                download_future = None

        async def perform_click() -> Optional[Page]:
            nonlocal download
            try:
                if self.single_tab_mode:
                    await page.mouse.move(center_x, center_y, steps=1)
                    await page.mouse.click(center_x, center_y)
                    return None
                else:
                    new_page_promise: asyncio.Task[Page] = asyncio.create_task(
                        page.context.wait_for_event(
                            "page", timeout=self._timeout_load * 1000
                        )
                    )

                    await page.mouse.move(center_x, center_y, steps=1)
                    await page.mouse.click(center_x, center_y, delay=10)

                    try:
                        new_page = await new_page_promise
                        await self.on_new_page(new_page)
                        return new_page
                    except TimeoutError:
                        return None
            except Exception as e:
                raise e

        if self.animate_actions:
            await self.add_cursor_box(page, identifier)
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(
                page, start_x, start_y, center_x, center_y
            )

        new_page = await perform_click()

        if download_future:
            try:
                if not download:
                    try:
                        download = await asyncio.wait_for(
                            download_future, timeout=self._timeout_load * 1000
                        )
                    except asyncio.TimeoutError:
                        pass

                if download:
                    self.logger.info(
                        f"Downloading {download.suggested_filename} to {self.downloads_folder}"
                    )
                    assert self.downloads_folder is not None
                    fname = os.path.join(
                        self.downloads_folder, download.suggested_filename
                    )
                    await download.save_as(fname)
            except Exception as e:
                self.logger.error(f"Error handling download: {e}")
            finally:
                if not download_future.done():
                    download_future.cancel()

        if self.animate_actions:
            await self.remove_cursor_box(page, identifier)

        if new_page:
            await new_page.wait_for_load_state("domcontentloaded")
            if self._sleep_after_action > 0:
                await new_page.wait_for_timeout(self._sleep_after_action * 1000)
        else:
            await page.wait_for_load_state("domcontentloaded")
            if self._sleep_after_action > 0:
                await page.wait_for_timeout(self._sleep_after_action * 1000)

        return new_page

    @handle_target_closed()
    async def select_option(self, page: Page, identifier: str) -> Optional[Page]:
        """
        Select an option element with the given identifier. If the element has a visible size,
        it will be clicked normally. Otherwise, it will be selected programmatically.

        Args:
            page (Page): The Playwright page object.
            identifier (str): The element identifier of the option to select.

        Returns:
            Optional[Page]: The Playwright page object of the newly focused tab.
        """
        await self._ensure_page_ready(page)
        new_page: Optional[Page] = None
        try:
            await page.wait_for_selector(
                f"[__elementId='{identifier}']", state="attached"
            )

            try:
                target = page.locator(f"[__elementId='{identifier}']").first
                box = await target.bounding_box()

                if box and box["width"] > 0 and box["height"] > 0:
                    return await self.click_id(page, identifier)

            except PlaywrightError as e:
                if "strict mode violation" in str(e):
                    elements = await page.locator(f"[__elementId='{identifier}']").all()
                    for element in elements:
                        try:
                            if await element.is_visible():
                                await element.click()
                                return new_page
                        except PlaywrightError:
                            continue

            option_element = await page.evaluate(
                """
                (identifier) => {
                    const elements = document.querySelectorAll(`[__elementId='${identifier}']`);
                    for (const el of elements) {
                        if (el.tagName.toLowerCase() === 'option') {
                            return true;
                        }
                    }
                    return false;
                }
                """,
                identifier,
            )

            if option_element:
                await page.evaluate(
                    """
                    (identifier) => {
                        const option = Array.from(document.querySelectorAll(`[__elementId='${identifier}']`))
                            .find(el => el.tagName.toLowerCase() === 'option');
                        if (!option) throw new Error('Option not found');
                        const select = option.closest('select');
                        if (select) {
                            option.selected = true;
                            select.dispatchEvent(new Event('change', { bubbles: true }));
                            select.blur();
                        }
                    }
                    """,
                    identifier,
                )
            else:
                await page.evaluate(
                    """
                    (identifier) => {
                        const element = document.querySelector(`[__elementId='${identifier}']`);
                        if (!element) throw new Error('Element not found');

                        // Dispatch multiple events to ensure the selection is registered
                        const events = ['mousedown', 'mouseup', 'click', 'change'];
                        events.forEach(eventType => {
                            element.dispatchEvent(new Event(eventType, { bubbles: true }));
                        });

                        // If element has aria-selected, set it
                        if (element.hasAttribute('aria-selected')) {
                            element.setAttribute('aria-selected', 'true');
                        }

                        // If element has a data-value, try to set it on the parent
                        const value = element.getAttribute('data-value');
                        if (value) {
                            const parent = element.closest('[role="listbox"], [role="combobox"]');
                            if (parent) {
                                parent.setAttribute('data-value', value);
                            }
                        }
                    }
                    """,
                    identifier,
                )

            if self._sleep_after_action > 0:
                await page.wait_for_timeout(self._sleep_after_action * 1000)

        except PlaywrightTimeoutError:
            raise ValueError(
                f"No option found with identifier '{identifier}' within "
                f"{self._timeout_load} seconds."
            ) from None
        return new_page

    @handle_target_closed()
    async def hover_coords(self, page: Page, x: float, y: float) -> None:
        """
        Hovers the mouse over the specified coordinates.

        Args:
            page (Page): The Playwright page object.
            x (float): The x coordinate to hover over.
            y (float): The y coordinate to hover over.
        """
        await self._ensure_page_ready(page)

        if self.animate_actions:
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.move(x, y)

    @handle_target_closed()
    async def hover_id(self, page: Page, identifier: str) -> None:
        """
        Hovers the mouse over the target with the given id.
        """
        await self._ensure_page_ready(page)
        target = page.locator(f"[__elementId='{identifier}']")

        try:
            await target.wait_for(timeout=5000)
        except TimeoutError:
            raise RuntimeError(
                f"Tool use response is invalid: no such element to hover: {identifier}"
            )

        await target.scroll_into_view_if_needed()
        await asyncio.sleep(0.3)

        box = cast(Dict[str, Union[int, float]], await target.bounding_box())

        if self.animate_actions:
            await self.add_cursor_box(page, identifier)
            start_x, start_y = self.last_cursor_position
            end_x, end_y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            await self.gradual_cursor_animation(page, start_x, start_y, end_x, end_y)
            await asyncio.sleep(0.1)
            await page.mouse.move(
                box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            )

            await self.remove_cursor_box(page, identifier)
        else:
            await page.mouse.move(
                box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            )

    async def _clear_field(self, page: Page) -> None:
        """Select all text in the focused field and delete it."""
        await page.keyboard.press("ControlOrMeta+A")
        await page.keyboard.press("Backspace")

    @handle_target_closed()
    async def fill_coords(
        self,
        page: Page,
        x: float,
        y: float,
        value: str,
        press_enter: bool = True,
        delete_existing_text: bool = False,
    ) -> Page | None:
        await self._ensure_page_ready(page)
        new_page: Page | None = None

        if self.animate_actions:
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.click(x, y)

        if delete_existing_text:
            await self._clear_field(page)

        if len(value) < 100:
            delay_typing_speed = 100
        else:
            delay_typing_speed = 10

        try:
            await page.keyboard.type(value, delay=delay_typing_speed)
        except PlaywrightError:
            await self._clear_field(page)
            await page.keyboard.type(value, delay=delay_typing_speed)

        if press_enter:
            await page.keyboard.press("Enter")

        try:
            new_page = await page.wait_for_event("popup", timeout=2000)
            assert isinstance(new_page, Page)
            await self.on_new_page(new_page)
        except (TimeoutError, asyncio.TimeoutError):
            pass

        return new_page

    @handle_target_closed()
    async def fill_id(
        self,
        page: Page,
        identifier: str,
        value: str,
        press_enter: bool = True,
        delete_existing_text: bool = False,
    ) -> None:
        """
        Fill the element with the given identifier with the specified value.
        Works with text inputs, textareas, and comboboxes.
        Args:
            page (Page): The Playwright page object.
            identifier (str): The element identifier.
            value (str): The value to fill.
            press_enter (bool): Whether to press enter after filling.
            delete_existing_text (bool): Whether to delete existing text before filling.
        """
        await self._ensure_page_ready(page)
        await page.wait_for_selector(f"[__elementId='{identifier}']", state="visible")
        target = page.locator(f"[__elementId='{identifier}']")
        await target.scroll_into_view_if_needed()

        try:
            await target.wait_for(timeout=5000)
        except TimeoutError:
            raise RuntimeError(
                f"Tool use response is invalid: No such element to fill input_text into: {identifier}"
            ) from None

        box = cast(Dict[str, Union[int, float]], await target.bounding_box())

        if self.single_tab_mode:
            await target.evaluate("""
                el => el.removeAttribute('target')
                // Remove 'target' on all <a> tags
                for (const a of document.querySelectorAll('a[target=_blank]')) {
                    a.removeAttribute('target');
                }
                // Remove 'target' on all <form> tags
                for (const frm of document.querySelectorAll('form[target=_blank]')) {
                    frm.removeAttribute('target');
                }
            """)

        page = await self.fill_coords(
            page,
            float(box["x"] + box["width"] / 2),
            float(box["y"] + box["height"] / 2),
            value,
            press_enter,
            delete_existing_text,
        )

    @handle_target_closed()
    async def scroll_id(self, page: Page, identifier: str, direction: str) -> None:
        await self._ensure_page_ready(page)
        await page.evaluate(
            f"""
        (function() {{
            let elm = document.querySelector("[__elementId='{identifier}']");
            if (elm) {{
                if ("{direction}" == "up") {{
                    elm.scrollTop = Math.max(0, elm.scrollTop - elm.clientHeight);
                }} else if ("{direction}" == "down") {{
                    elm.scrollTop = Math.min(elm.scrollHeight - elm.clientHeight, elm.scrollTop + elm.clientHeight);
                }} else if ("{direction}" == "left") {{
                    elm.scrollLeft = Math.max(0, elm.scrollLeft - elm.clientWidth);
                }} else if ("{direction}" == "right") {{
                    elm.scrollLeft = Math.min(elm.scrollWidth - elm.clientWidth, elm.scrollLeft + elm.clientWidth);
                }}
            }}
        }})();
    """
        )

    async def keypress(self, page: Page, keys: list[str], key_mapping=None) -> None:
        """
        Press specified keys in sequence.

        Args:
            page (Page): The Playwright page object
            keys (List[str]): List of keys to press
        """
        await self._ensure_page_ready(page)
        key_mapping = key_mapping or CUA_KEY_TO_PLAYWRIGHT_KEY
        mapped_keys = [key_mapping.get(key.lower(), key) for key in keys]
        pressed: list[str] = []
        try:
            for key in mapped_keys:
                await page.keyboard.down(key)
                pressed.append(key)
                if self.animate_actions:
                    await asyncio.sleep(0.05)
        except Exception as e:
            msg = str(e)
            if "Unknown key" in msg:
                bad = next(
                    (k for k in keys if key_mapping.get(k.lower(), k) not in pressed),
                    keys[-1],
                )
                raise ValueError(
                    f"Unknown key: {bad!r}. keypress only accepts single characters "
                    f"or named keys (Enter, Tab, ArrowDown, Control, ...). "
                    f"Use type() for text input."
                ) from None
            raise RuntimeError(
                f"I tried to keypress(keys={keys}), but I got an error: {e}"
            ) from None
        finally:
            for key in reversed(pressed):
                try:
                    await page.keyboard.up(key)
                    if self.animate_actions:
                        await asyncio.sleep(0.05)
                except Exception:
                    continue

    @handle_target_closed()
    async def cua_click(
        self, page: Page, x: int, y: int, button: str = "left"
    ) -> Page | None:
        match button:
            case "back":
                await self.back(page)
                return None
            case "forward":
                await self.cua_forward(page)
                return None
            case "wheel":
                await page.mouse.wheel(x, y)
                return None
            case _:
                await self._ensure_page_ready(page)
                button_type = "right" if button == "right" else "left"
                return await self._click_with_popup_capture(
                    page, lambda: page.mouse.click(x, y, button=button_type)
                )

    @handle_target_closed()
    async def cua_double_click(self, page: Page, x: int, y: int) -> Page | None:
        await self._ensure_page_ready(page)
        return await self._click_with_popup_capture(
            page, lambda: page.mouse.dblclick(x, y)
        )

    @handle_target_closed()
    async def cua_scroll(
        self, page: Page, x: int, y: int, scroll_x: int, scroll_y: int
    ) -> None:
        await page.mouse.move(x, y)
        await page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

    @handle_target_closed()
    async def cua_type(self, page: Page, text: str) -> None:
        await page.keyboard.type(text)

    async def cua_wait(self, page: Page, ms: int = 1000) -> None:
        await asyncio.sleep(ms / 1000.0)

    @handle_target_closed()
    async def cua_move(self, page: Page, x: int, y: int) -> None:
        await page.mouse.move(x, y)

    @handle_target_closed()
    async def cua_drag(self, page: Page, path: List[Dict[str, int]]) -> None:
        if not path:
            return
        await page.mouse.move(path[0]["x"], path[0]["y"])
        await page.mouse.down()
        for point in path[1:]:
            await page.mouse.move(point["x"], point["y"])
        await page.mouse.up()

    @handle_target_closed()
    async def cua_forward(self, page: Page) -> None:
        await self._ensure_page_ready(page)
        await page.go_forward()

    @handle_target_closed()
    async def get_webpage_text(self, page: Page, n_lines: int = 100) -> str:
        """
        page: playwright page object
        n_lines: number of lines to return from the page innertext
        return: text in the first n_lines of the page
        """
        await self._ensure_page_ready(page)
        try:
            text_in_viewport = await page.evaluate("""() => {
                return document.body.innerText;
            }""")
            text_in_viewport = "\n".join(text_in_viewport.split("\n")[:n_lines])
            text_in_viewport = "\n".join(
                [line for line in text_in_viewport.split("\n") if line.strip()]
            )
            assert isinstance(text_in_viewport, str)
            return text_in_viewport
        except Exception:
            return ""

    async def get_page_markdown(self, page: Page) -> str:
        await self._ensure_page_ready(page)
        return await self._text_utils.get_page_markdown(page)

    @handle_target_closed()
    async def get_page_url(self, page: Page) -> str:
        """Get the current page URL."""
        await self._ensure_page_ready(page)
        return page.url

    @handle_target_closed()
    async def get_page_title(self, page: Page) -> str:
        """Get the current page title."""
        await self._ensure_page_ready(page)
        return await page.title()

    @handle_target_closed()
    async def wait_for_load_state(
        self, page: Page, state: str = "domcontentloaded", timeout: Optional[int] = None
    ) -> None:
        """Wait for the page to reach a specific load state."""
        await page.wait_for_load_state(state, timeout=timeout)
