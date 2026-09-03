from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"


def npm_command() -> str:
    command = "npm.cmd" if os.name == "nt" else "npm"
    resolved = shutil.which(command)
    if not resolved:
        raise SystemExit("Node.js/npm is required to build or develop the dashboard.")
    return resolved


def build_frontend() -> None:
    if not (FRONTEND / "node_modules").exists():
        subprocess.run([npm_command(), "install"], cwd=FRONTEND, check=True)
    subprocess.run([npm_command(), "run", "build"], cwd=FRONTEND, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Local Context Manager")
    parser.add_argument(
        "--dev", action="store_true", help="Run Vite with hot reload beside the proxy"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=7331, help="Proxy/dashboard port (default: 7331)"
    )
    parser.add_argument("--skip-build", action="store_true", help="Use the existing frontend build")
    arguments = parser.parse_args()

    os.environ["LCM_HOST"] = arguments.host
    os.environ["LCM_PORT"] = str(arguments.port)
    os.environ.setdefault("LCM_DATA_DIR", str(ROOT / "data"))

    frontend_process: subprocess.Popen[bytes] | None = None
    if arguments.dev:
        if not (FRONTEND / "node_modules").exists():
            subprocess.run([npm_command(), "install"], cwd=FRONTEND, check=True)
        frontend_process = subprocess.Popen([npm_command(), "run", "dev"], cwd=FRONTEND)
        print("Dashboard development URL: http://127.0.0.1:5173")
    elif not arguments.skip_build and not (FRONTEND / "dist" / "index.html").exists():
        build_frontend()

    print(f"LCM dashboard and proxy: http://{arguments.host}:{arguments.port}")
    print(f"OpenAI-compatible base URL: http://{arguments.host}:{arguments.port}/v1")
    try:
        uvicorn.run(
            "lcm.main:app",
            host=arguments.host,
            port=arguments.port,
            app_dir=str(BACKEND),
            reload=arguments.dev,
            reload_dirs=[str(BACKEND)] if arguments.dev else None,
        )
    finally:
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)


if __name__ == "__main__":
    main()
