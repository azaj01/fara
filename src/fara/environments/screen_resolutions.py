"""Popular desktop screen resolutions used for `randomize_screen_res`.

`weight` is the sampling probability (in %) used by
``sample_random_screen_resolution``; the 12 resolutions below are chosen
and weighted to span Windows (native + DPI-scaled), macOS (MBP "Looks
Like" modes), Linux-flagship 16:10, Surface 3:2, and ultrawide layouts.
Weights do not need to sum to exactly 100 — ``random.choices`` normalizes
them. The `aspect`, `scenario`, and `sources` fields are documentation
only.
"""

from __future__ import annotations

import random
from typing import NamedTuple


class ScreenResolution(NamedTuple):
    width: int
    height: int
    aspect: str
    scenario: str
    weight: float
    sources: tuple[str, ...]


POPULAR_SCREEN_RESOLUTIONS: list[ScreenResolution] = [
    ScreenResolution(
        width=1920,
        height=1080,
        aspect="16:9",
        scenario="Windows laptop/desktop dominant baseline — Full HD at 100% scaling",
        weight=32.0,
        sources=(
            "https://gs.statcounter.com/screen-resolution-stats/desktop",
            "https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam",
        ),
    ),
    ScreenResolution(
        width=1536,
        height=864,
        aspect="16:9",
        scenario=(
            "Windows 1920x1080 @125% scaling as seen by DPI-Unaware pyautogui — "
            "the default 'scaled FHD laptop' view"
        ),
        weight=14.0,
        sources=(
            "https://gs.statcounter.com/screen-resolution-stats/desktop",
            "https://learn.microsoft.com/en-us/windows/win32/hidpi/high-dpi-desktop-application-development-on-windows",
        ),
    ),
    ScreenResolution(
        width=1366,
        height=768,
        aspect="16:9",
        scenario=(
            "Budget Windows laptops native; sub-$300 Chromebooks (K-12 fleet); "
            "Surface Laptop SE"
        ),
        weight=8.0,
        sources=(
            "https://gs.statcounter.com/screen-resolution-stats/desktop",
            "https://www.aboutchromebooks.com/chromebook-hardware-statistics/",
            "https://www.starryhope.com/chromebooks/chromebook-comparison-chart/",
        ),
    ),
    ScreenResolution(
        width=2560,
        height=1440,
        aspect="16:9",
        scenario=(
            "QHD desktop monitor standard for Windows; also macOS Studio Display "
            "default 'Looks Like' and 4K Windows panels @ 150%"
        ),
        weight=9.0,
        sources=(
            "https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam",
            "https://www.apple.com/studio-display/specs/",
            "https://gs.statcounter.com/screen-resolution-stats/desktop",
        ),
    ),
    ScreenResolution(
        width=1280,
        height=720,
        aspect="16:9",
        scenario=(
            "Windows 1920x1080 @150% scaling as seen by DPI-Unaware pyautogui; "
            "also small/netbook panels"
        ),
        weight=5.0,
        sources=(
            "https://gs.statcounter.com/screen-resolution-stats/desktop",
            "https://github.com/asweigart/pyautogui/issues/33",
        ),
    ),
    ScreenResolution(
        width=3840,
        height=2160,
        aspect="16:9",
        scenario=(
            "4K monitor and 4K laptop @ 100% scaling — Windows enthusiasts, "
            "some Linux dev workstations"
        ),
        weight=5.0,
        sources=(
            "https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam",
            "https://tech-docs.system76.com/models/oryp10/README.html",
        ),
    ),
    ScreenResolution(
        width=1504,
        height=1000,
        aspect="3:2",
        scenario=(
            'Surface Laptop 13.5" (2256x1504 @150% default, DPI-Unaware view); '
            'Surface Book 3 13.5" (3000x2000 @200% = 1500x1000)'
        ),
        weight=4.0,
        sources=(
            "https://surfacetip.com/surface-display-comparison/",
            "https://glossary.surfacetip.com/definition/2256x1504/",
        ),
    ),
    ScreenResolution(
        width=3440,
        height=1440,
        aspect="21:9",
        scenario=(
            "Ultrawide gaming/productivity monitor — almost exclusively Windows "
            "desktop enthusiasts; fundamentally different layout regime "
            "(side-by-side apps are the default)"
        ),
        weight=3.0,
        sources=(
            "https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam",
        ),
    ),
    ScreenResolution(
        width=1512,
        height=982,
        aspect="~3:2",
        scenario=(
            "macOS 14\" MacBook Pro M1 Pro -> M5 Max default 'Looks Like' "
            "(pixel-doubled from 3024x1964 native) — the developer-laptop default"
        ),
        weight=7.0,
        sources=(
            "https://www.apple.com/macbook-pro/specs/",
            "https://9to5mac.com/2021/10/19/new-macbook-pro-screen-resolution-options/",
            "https://github.com/asweigart/pyautogui/issues/699",
        ),
    ),
    ScreenResolution(
        width=1728,
        height=1117,
        aspect="~3:2",
        scenario=(
            "macOS 16\" MacBook Pro M1 Pro -> M5 Max default 'Looks Like' "
            "(pixel-doubled from 3456x2234 native)"
        ),
        weight=5.0,
        sources=(
            "https://www.apple.com/macbook-pro/specs/",
            "https://9to5mac.com/2021/10/19/new-macbook-pro-screen-resolution-options/",
        ),
    ),
    ScreenResolution(
        width=1440,
        height=900,
        aspect="16:10",
        scenario=(
            'Legacy 13" MBP/MBA (Intel/M1 era, 2560x1600 panel @2x); also older '
            "Windows WXGA+ business monitors"
        ),
        weight=3.0,
        sources=(
            "https://gs.statcounter.com/screen-resolution-stats/desktop",
            "https://support.apple.com/en-us/111883",
        ),
    ),
    ScreenResolution(
        width=1920,
        height=1200,
        aspect="16:10",
        scenario=(
            "Linux laptop flagship: ThinkPad X1C/T14/P14s base, Dell XPS 13 "
            "Developer Edition base, System76 Lemur Pro/Pangolin/Oryx Pro, "
            "Framework 13, Lenovo/HP Chromebook Plus 14 (2025-26)"
        ),
        weight=5.0,
        sources=(
            "https://psref.lenovo.com/syspool/Sys/PDF/ThinkPad/ThinkPad_X1_Carbon_Gen_12/ThinkPad_X1_Carbon_Gen_12_Spec.pdf",
            "https://www.dell.com/support/manuals/en-us/xps-13-9340-laptop/xps-13-9340_owners_manual/display",
            "https://system76.com/laptops/lemur-pro",
            "https://news.lenovo.com/pressroom/press-releases/next-generation-hybrid-ai-chromebook-plus-14-10/",
        ),
    ),
]


def sample_random_screen_resolution(
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Sample a (width, height) from POPULAR_SCREEN_RESOLUTIONS weighted by `weight`."""
    r = rng or random
    resolutions = [(s.width, s.height) for s in POPULAR_SCREEN_RESOLUTIONS]
    weights = [s.weight for s in POPULAR_SCREEN_RESOLUTIONS]
    return r.choices(resolutions, weights=weights, k=1)[0]
