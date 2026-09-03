# Local Context Manager frontend

React and TypeScript interface for the workbench, full context dashboard, and compact context-window overlay. For complete application setup, see the repository [README](../README.md).

Run frontend commands from this directory:

```bash
npm ci
npm run dev
npm run lint
npm run build
```

During development, Vite proxies `/api`, `/v1`, and `/healthz` to the FastAPI service at `http://127.0.0.1:7331`.
