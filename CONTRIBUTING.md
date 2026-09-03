# Contributing

Thank you for helping improve Local Context Manager. Keep changes focused, local-first, and honest about what has been measured.

## Development setup

Requirements: Python 3.11+, Node.js 20+, and npm.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cd frontend
npm ci
cd ..
.\.venv\Scripts\python.exe run.py --dev
```

Native LongCat workbench development additionally requires a built XTLLM checkout and converted model assets. Configure those paths in `.env`; never commit model files or local paths.

## Before opening a pull request

```powershell
.\.venv\Scripts\python.exe -m ruff check backend tests run.py
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run lint
npm run build
```

Please include tests for behavior changes. UI changes should preserve keyboard access, clear error states, and the local-only privacy model.

## Pull requests

- Explain the user-facing problem and the chosen solution.
- Keep unrelated refactors out of the change.
- Document new settings and update `.env.example` when configuration changes.
- Do not include databases, prompts, model weights, generated bundles, credentials, or benchmark claims without reproducible evidence.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
