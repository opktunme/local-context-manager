from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOW_TITLE = "LCM Context Window"


def find_browser() -> Path | None:
    candidates = [
        Path(os.environ.get("PROGRAMFILES(X86)", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", ""))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", ""))
        / "Google/Chrome/Application/chrome.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    resolved = shutil.which("msedge") or shutil.which("chrome")
    return Path(resolved) if resolved else None


def find_overlay_window() -> int | None:
    user32 = ctypes.windll.user32
    matches: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if not length:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if WINDOW_TITLE in buffer.value:
            matches.append(hwnd)
        return True

    user32.EnumWindows(callback_type(visit), 0)
    return matches[-1] if matches else None


def make_topmost(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetWindowPos(
        hwnd,
        -1,  # HWND_TOPMOST
        0,
        0,
        264,
        264,
        0x0002 | 0x0040,  # SWP_NOMOVE | SWP_SHOWWINDOW
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the LCM always-on-top context window")
    parser.add_argument("--url", default="http://127.0.0.1:7331/overlay")
    args = parser.parse_args()
    if os.name != "nt":
        raise SystemExit("The always-on-top launcher currently supports Windows only.")
    browser = find_browser()
    if not browser:
        raise SystemExit("Microsoft Edge or Google Chrome was not found.")

    profile = ROOT / "data" / "overlay-browser"
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [
            str(browser),
            f"--app={args.url}",
            "--window-size=264,264",
            "--no-first-run",
            "--disable-extensions",
            f"--user-data-dir={profile}",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        hwnd = find_overlay_window()
        if hwnd:
            make_topmost(hwnd)
            return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
