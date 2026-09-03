from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings. Every default is local and can work offline."""

    model_config = SettingsConfigDict(
        env_prefix="LCM_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 7331
    data_dir: Path = Path(os.getenv("LCM_DATA_DIR", PROJECT_ROOT / "data"))
    database_name: str = "lcm.sqlite3"
    request_retention: int = 30
    allow_private_lan: bool = True
    xtllm_root: Path = PROJECT_ROOT.parent / "xtllm-runtime"
    xtllm_binary_root: Path = PROJECT_ROOT.parent / "XTLLM"
    longcat_timeout_seconds: int = 900

    @property
    def database_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / self.database_name


settings = Settings()
