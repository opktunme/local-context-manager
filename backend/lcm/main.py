from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import create_api_router
from .config import PROJECT_ROOT, settings
from .context import ContextEngine
from .proxy import ProxyService
from .storage import Database

database = Database(settings.database_path, settings.request_retention)
engine = ContextEngine(database)
proxy = ProxyService(database, engine, settings)

app = FastAPI(
    title="Local Context Manager",
    description="Local-only context runtime and debugger for OpenAI-compatible LLM servers.",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-LCM-Session-ID",
        "X-LCM-Request-ID",
        "X-LCM-Incoming-Tokens",
        "X-LCM-Forwarded-Tokens",
    ],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self' http://127.0.0.1:* http://localhost:*"
    )
    return response


app.include_router(create_api_router(database, engine, proxy, settings))

dist_dir = PROJECT_ROOT / "frontend" / "dist"
assets_dir = dist_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/{path:path}", include_in_schema=False)
async def dashboard(path: str):
    index = dist_dir / "index.html"
    candidate = dist_dir / path
    if candidate.is_file() and dist_dir in candidate.resolve().parents:
        return FileResponse(candidate)
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        status_code=200,
        content={
            "service": "Local Context Manager",
            "message": "Frontend is not built yet. Run `python run.py --dev` or build the frontend.",
            "api_docs": "/api/docs",
        },
    )
