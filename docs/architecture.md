# Architecture

```text
Any OpenAI-compatible frontend
          │ full chat request / SSE
          ▼
┌──────────────────────────────────────────┐
│ Local Context Manager :7331              │
│                                          │
│  session inference → context normalize   │
│         ↓                 ↓              │
│  policy overlay ← SQLite audit state     │
│         ↓                                │
│  payload rebuild → request inspector     │
└──────────────────────┬───────────────────┘
                       │ transformed request / SSE
                       ▼
      LM Studio · Ollama · llama.cpp · vLLM
      MLX / MLX-LM · XTLLM / xLLM · generic
```

## Request lifecycle

1. LCM validates the OpenAI Chat Completions envelope while retaining unknown fields.
2. Every message receives a stable content fingerprint and occurrence suffix.
3. `X-LCM-Session-ID` selects an explicit policy overlay. Without it, shared-prefix matching chooses a recent compatible session.
4. SQLite restores statuses such as `removed`, `pinned`, and `summarized` even when a frontend resends its original full history.
5. Linked tool calls and results are handled as a dependency group.
6. A new payload is reconstructed. That payload—not the original—is sent to the backend.
7. Streaming response bytes are relayed as they arrive. LCM records start, first-byte, and end times.

## Backend adapters

Most current local runtimes expose `/v1/chat/completions` and `/v1/models`, so they inherit the generic OpenAI-compatible adapter. Specialized adapters add discovery or metrics without making those capabilities mandatory. Unsupported optional metadata is displayed as unavailable, never fabricated.

## Storage

SQLite uses WAL mode and foreign keys. Only a bounded number of full request snapshots are retained per session (30 by default). Context items are keyed by fingerprint plus occurrence; policies and audit records are compact overlays. No external database is required.

## KV-cache estimate

When model architecture fields are supplied, LCM uses:

```text
bytes = tokens × layers × 2 (K + V) × KV heads × head dimension
        × bytes per KV element × batch size
```

The dashboard exposes every assumption and labels the result as an estimate.
