# Local Context Manager

> See, control, and optimize exactly what is sent to your local LLM.

[![CI](https://github.com/opktunme/local-context-manager/actions/workflows/ci.yml/badge.svg)](https://github.com/opktunme/local-context-manager/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-22c55e.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/)
[![Node 20+](https://img.shields.io/badge/node-20%2B-339933.svg)](https://nodejs.org/)

Local Context Manager (LCM) is a local middleware proxy, context debugger, and optional local chat workbench. Existing frontends can keep talking to its OpenAI-compatible endpoint; the bundled workbench can also run the installed LongCat profiles directly through XTLLM while applying the same persistent context policy.

```text
Frontend / coding agent  →  LCM :7331  →  Local inference server
                                │
                         browser dashboard
```

The name and small `LCM` wordmark are intentionally isolated so the project can be renamed later.

## Why this exists

Context has a direct cost on personal hardware:

- KV-cache consumes RAM or VRAM.
- Long prompts increase prefill work and time to first token.
- Irrelevant history competes for model attention.
- Small local models have less room to recover from noisy context.
- A model may support 128K while the practical hardware budget is only 16K or 20K.

LCM turns the hidden prompt into inspectable context objects. Removing an object creates a persistent policy overlay, so it stays excluded if a frontend resends the unchanged full conversation on its next turn.

## MVP workflow

1. Start LM Studio, Ollama, llama.cpp, vLLM, MLX-LM, XTLLM/xLLM, or another local OpenAI-compatible server.
2. Start LCM.
3. Choose the backend in the first-run screen.
4. Change the frontend base URL from (for example) `http://localhost:1234/v1` to `http://localhost:7331/v1`.
5. Chat normally. LCM captures context attribution and timing while preserving SSE streaming.
6. Remove, pin, inspect, summarize, restore, or undo context from the dashboard.
7. Open **Requests** to verify the exact incoming and forwarded JSON.

## Quick start

Requirements: Python 3.11+ and Node.js 20+.

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm ci
cd ..
.\.venv\Scripts\python.exe run.py
```

Then open [http://127.0.0.1:7331](http://127.0.0.1:7331).

The default route is the Codex-style workbench. The complete context dashboard remains available at [http://127.0.0.1:7331/context](http://127.0.0.1:7331/context), and the compact window is at `/overlay`. Without an XTLLM installation, the proxy and context dashboard still work normally and the native LongCat profiles are shown as unavailable.

### Optional XTLLM LongCat workbench

The bundled native workbench supports the LongCat profiles produced by [XTLLM](https://github.com/opktunme/XTLLM). XTLLM binaries and converted model weights are intentionally not included in this repository. Copy `.env.example` to `.env`, then set the two paths for your installation:

```dotenv
LCM_XTLLM_ROOT=D:/models/xtllm-runtime
LCM_XTLLM_BINARY_ROOT=D:/src/XTLLM
```

`LCM_XTLLM_ROOT` must contain the converted `runtime/longcat-flash-lite-*` directories. `LCM_XTLLM_BINARY_ROOT` must contain the XTLLM `build/xtllm-longcat-flash-lite*.exe` binaries. When both are present, LCM exposes:

- **LongCat Exact** — Q4 experts with the Q8 shared path; measured at about 9.66 GiB peak VRAM.
- **LongCat Full** — full 12-route hybrid with quantized dense/output paths; measured at about 10.38 GiB peak VRAM.

The resource panel uses one shared 70 GiB host-RAM envelope for inference/expert cache plus a separately adjustable context-window allocation, and an independent 11.5 GiB VRAM ceiling. These defaults describe the reference test machine and can be adjusted in code for other hardware. The installed LongCat Flash Lite Sparse config supports 983,040 tokens (the model's advertised “up to 1M” window). XTLLM sizes LongCat context dynamically instead of compiling a 256-token ceiling. Its current expanded FP32 KV cache is device-local, so active capacity is calculated from the selected VRAM budget and profile; the larger model/host-RAM plan remains visible separately. Structured multi-turn transcripts use XTLLM's role-preserving binary protocol rather than flattening messages into text labels. Each run streams concise UI-only stages (context inspection, resource preparation, model load, decode, completion), so progress visibility does not consume model context. Chat history, settings, metrics, context actions, and run results persist in the same local SQLite database.

When stored conversation history is larger than the VRAM-safe runtime window, the workbench automatically sends the largest recent slice that fits while preserving the full conversation in SQLite. The activity trail reports how many older messages were omitted from that native request.

After installation, `start.cmd` is the shortest production-style start command. For hot-reload development, run:

```powershell
.\dev.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend && npm install && cd ..
.venv/bin/python run.py
```

Development:

```bash
chmod +x dev
./dev
```

`python run.py` creates the production dashboard bundle when necessary, then serves the dashboard and proxy from one process. `python run.py --dev` starts the FastAPI and Vite development servers together.

### Configuration

All settings are optional and use the `LCM_` prefix. Values can be placed in a local `.env` file or exported by the shell.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LCM_HOST` | `127.0.0.1` | Dashboard and proxy bind address |
| `LCM_PORT` | `7331` | Dashboard and proxy port |
| `LCM_DATA_DIR` | `<repo>/data` | SQLite database and temporary local state |
| `LCM_XTLLM_ROOT` | sibling `xtllm-runtime` directory | Converted LongCat runtime assets |
| `LCM_XTLLM_BINARY_ROOT` | sibling `XTLLM` directory | Built XTLLM checkout |
| `LCM_LONGCAT_TIMEOUT_SECONDS` | `900` | Maximum native inference time per response |

Do not commit `.env`; it is ignored because it may reveal machine-specific paths.

## Live context window

The dashboard's **Live overlay** button launches a glance-size, always-on-top Windows window (roughly 2.5 inches square at standard display scaling). Its 400 colored cells fill from the bottom as context grows; each color maps to a context category, and the dashed line marks the configured working budget. The display refreshes from a metadata-only local endpoint several times per second, so raw prompt content is not copied into the overlay feed.

The browser view is also available directly at [http://127.0.0.1:7331/overlay](http://127.0.0.1:7331/overlay). The Windows launcher uses a dedicated browser profile under the project's local `data` directory.

## Backend setup

| Runtime | Default URL | MVP status | Notes |
| --- | --- | --- | --- |
| LM Studio | `http://127.0.0.1:1234` | Supported | Chat Completions, models, SSE; OpenAI-compatible route is used |
| Ollama | `http://127.0.0.1:11434` | Supported | OpenAI-compatible chat/models plus optional native discovery |
| llama.cpp | `http://127.0.0.1:8080` | Supported | Chat Completions, models, SSE |
| vLLM | `http://127.0.0.1:8000` | Supported | Chat Completions, models, SSE |
| MLX / MLX-LM | `http://127.0.0.1:8080` | Supported | OpenAI-style chat server; optional metadata varies by server |
| XTLLM / xLLM | `http://127.0.0.1:9977` | Supported through OpenAI-compatible adapter | Advanced runtime cache controls intentionally deferred |
| Other OpenAI-compatible | user supplied | Supported | Requires `/v1/chat/completions`; `/v1/models` is used for connection tests |

The adapter assumptions follow current official runtime documentation: [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), [LM Studio OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat), [vLLM online serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/), [MLX-LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md), and [xLLM online service](https://xllm.readthedocs.io/en/latest/getting_started/online_service/).

### LM Studio example

Start the local server in LM Studio's Developer tab (normally port 1234). In LCM choose **LM Studio**, test `http://127.0.0.1:1234`, and save. Configure the existing client with:

```text
Base URL: http://127.0.0.1:7331/v1
API key:  any non-empty placeholder, if the client requires one
```

### Ollama example

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

Choose **Ollama** and test `http://127.0.0.1:11434`.

### llama.cpp example

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

Choose **llama.cpp** and test `http://127.0.0.1:8080`.

## API compatibility

Implemented proxy routes:

- `POST /v1/chat/completions`
- `GET /v1/models`

Chat Completions supports normal JSON and byte-relayed SSE streaming. System, developer, user, assistant, tool-call, and tool-result messages are retained. Multimodal content arrays are attributed without trying to decode attachments. Unknown top-level request fields and unknown fields inside retained messages are forwarded rather than stripped.

Optional explicit session header:

```http
X-LCM-Session-ID: my-project-thread-42
```

LCM hashes the external identifier before using it as an internal key. When the header is absent, it compares normalized fingerprints, shared prefixes, model/backend identity, and recent activity.

## Dashboard

The dark, desktop-first working surface includes:

- Model/backend identity and context/budget meter.
- Token attribution by system, conversation, code/files, tools, RAG/memory, images, and summaries.
- Chronological, largest-first, type, and recommendation sorting.
- Full-content inspector with source, fingerprint, dependencies, inclusion reason, context share, and estimated saving.
- Zero-compute recommendations with an explanation for each recommendation.
- Accumulation timeline and largest-consumers view.
- Before/after removal review, linked-object warnings, undo, restore, pin, and reset.
- Editable zero-compute summary drafts and optional local-LLM summary generation through the configured backend.
- Incoming versus forwarded JSON and a readable per-message diff.
- Measured request duration and time to first response byte.
- KV-cache estimate only when architecture assumptions are configured.
- Local audit history and a no-model-required demo session.

## Safety rules

By default LCM will not:

- remove the current user message;
- remove system or developer instructions;
- alter pinned context;
- split linked tool-call and tool-result objects;
- call a cloud model or accept a public backend URL;
- label an estimated token count as exact;
- claim unmeasured speed improvement;
- modify the frontend's stored conversation.

Real-time cleanup is off by default and runs locally before each forwarded request when enabled. The dashboard exposes three policies:

- **Conservative** removes later exact duplicates only.
- **Balanced** also removes high-confidence waste such as oversized stale logs and tool output.
- **Aggressive** additionally creates zero-compute summaries for older substantial turns and prunes stale retrieved context.

Pinned objects, system/developer instructions, and the current user request remain protected in every mode. Every automatic action is written to the local audit log and can be undone.

## Context and token model

Each OpenAI message becomes a context object with a stable SHA-256-derived fingerprint and an occurrence suffix. LCM retains the raw message for lossless reconstruction. It records type, role, source, order, token attribution, status, preview, relationships, recommendation, and inclusion reason.

The bundled tokenizer is an offline fallback estimator and every count it produces is displayed with `~` and an **estimated** label. The abstraction is ready for backend tokenization, locally installed Hugging Face tokenizers, and GGUF metadata in a later iteration. Whole-request prompt token counts reported by the backend are stored as measured metrics; they are not misapplied to individual items.

## Storage and privacy

All state is in `data/lcm.sqlite3` by default:

- sessions and inferred lineage;
- context fingerprints and statuses;
- pins, removals, summaries, and undo snapshots;
- backend configuration;
- bounded request snapshots and performance metrics;
- audit records.
- workbench chats, per-chat LongCat limits, and measured generation metrics.

There is no account, telemetry, CDN, external font, analytics package, update checker, or cloud API. Local inference clients ignore environment HTTP proxies (`trust_env=False`). Public backend hosts are rejected; loopback, private IP, `localhost`, and `.local` hosts are allowed.

See [architecture](docs/architecture.md) and [privacy/security](docs/privacy-security.md).

## Contributing

Bug reports and focused pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [changelog](CHANGELOG.md) before contributing.

## Development and tests

Install development dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run all verification:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check backend tests run.py
cd frontend
npm run lint
npm run build
```

The suite uses local mock OpenAI-compatible and Ollama-style servers. A real model is not required. Covered behavior includes pass-through, SSE relay, transformations, session inference, fingerprint stability, persistent removal, restoration, pin safety, duplicates, estimated token labelling, budgets, tool dependency integrity, summaries/undo, backend failure, malformed input, multimodal structures, large contexts, and the local-only URL guard.

## Project structure

```text
backend/lcm/
  adapters/          runtime adapters and local URL policy
  context/           fingerprints, normalization, recommendations, overlay engine
  metrics/           KV-cache calculation
  proxy/             JSON and SSE relay
  workbench.py       native XTLLM LongCat execution and progress bridge
  storage/           SQLite schema and audit state
  tokenizers/        tokenizer abstraction and labelled fallback
frontend/src/        React workbench, context dashboard, and compact overlay
tests/               unit and mock-server integration tests
docs/                architecture and security notes
run.py               one-command cross-platform launcher
```

## Intentionally deferred

The MVP provides extension points but does not pretend these are complete:

- automatic Hugging Face/GGUF tokenizer discovery;
- optional local embedding model and semantic clustering;
- backend-specific live GPU/CPU telemetry;
- prompt-cache and backend KV-cache introspection/control;
- Responses API and Anthropic Messages proxy routes;
- coding-agent and MCP-specific context object parsers;
- model-specific calibrated speed predictions;
- persistent XTLLM worker processes (the installed binaries currently start one isolated Vulkan run per reply).

The next iteration should add an exact local tokenizer provider layer, enrich LM Studio/Ollama/llama.cpp model metadata, then add the Responses API while reusing the same policy overlay.

## License

[MIT](LICENSE). Contributions are welcome.
