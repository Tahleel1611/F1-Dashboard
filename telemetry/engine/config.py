"""Runtime configuration for telemetry ingestion and visualization."""

from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class F1RuntimeConfig:
    data_mode: str
    fastf1_cache_dir: Path
    database_url: str | None

    @property
    def is_live_mode(self) -> bool:
        return self.data_mode.upper() == "LIVE"


@lru_cache(maxsize=1)
def get_runtime_config() -> F1RuntimeConfig:
    data_mode = os.getenv("F1_DATA_MODE", "LIVE").strip().upper() or "LIVE"
    if data_mode not in {"LIVE", "OFFLINE"}:
        data_mode = "LIVE"

    cache_dir = Path(os.getenv("FASTF1_CACHE_DIR", ".fastf1_cache")).expanduser().resolve()
    database_url = os.getenv("DATABASE_URL")
    return F1RuntimeConfig(data_mode=data_mode, fastf1_cache_dir=cache_dir, database_url=database_url)
