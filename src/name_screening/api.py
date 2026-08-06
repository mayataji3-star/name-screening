from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig
from .data_io import load_watchlist
from .models import ScreenRequest, ScreenResponse
from .screener import NameScreener

config = AppConfig()
watchlist_df = load_watchlist(
    str(config.watchlist_path),
    target_size=config.watchlist_target_size,
    opensanctions_path=str(config.opensanctions_path),
    alias_map_path=str(config.alias_map_path),
    use_mock_data=config.use_mock_data,
    entity_scope=config.entity_scope,
    max_records=config.opensanctions_max_records,
)
screener = NameScreener.from_config(config)
screener.ensure_index(watchlist_df)

app = FastAPI(title="Bilingual Name Screening MVP")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index/rebuild")
def rebuild_index() -> dict[str, str]:
    global watchlist_df
    watchlist_df = load_watchlist(
        str(config.watchlist_path),
        target_size=config.watchlist_target_size,
        opensanctions_path=str(config.opensanctions_path),
        alias_map_path=str(config.alias_map_path),
        use_mock_data=config.use_mock_data,
        entity_scope=config.entity_scope,
        max_records=config.opensanctions_max_records,
    )
    screener.rebuild_index(watchlist_df)
    return {"status": "rebuilt", "records": str(len(watchlist_df))}


@app.post("/screen", response_model=ScreenResponse)
def screen(request: ScreenRequest) -> ScreenResponse:
    return screener.screen(request)


@app.post("/screen/batch", response_model=list[ScreenResponse])
def screen_batch(requests: list[ScreenRequest]) -> list[ScreenResponse]:
    return screener.screen_batch(requests)
