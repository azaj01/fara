"""Shared captcha-waiting utility for browser-based agents."""

import asyncio
import logging

CAPTCHA_TIMEOUT = 60

logger = logging.getLogger(__name__)


async def wait_for_captcha(env, timeout: int = CAPTCHA_TIMEOUT) -> bool:
    """Wait for captcha to be solved if environment supports it."""
    if not hasattr(env, "wait_for_captcha_resolution"):
        return True

    if hasattr(env, "_captcha_event") and env._captcha_event.is_set():
        return True

    logger.info(f"Waiting {timeout}s for captcha to finish...")
    try:
        await asyncio.wait_for(
            env.wait_for_captcha_resolution(),
            timeout=timeout,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Captcha timeout after {timeout} seconds!")
        return False
