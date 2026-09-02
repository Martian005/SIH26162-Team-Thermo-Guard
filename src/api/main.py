"""FastAPI service backing the map UI.

Layers are served as GeoJSON. The registry is small enough to ship whole; raw
detections are always bbox- and date-filtered with a hard cap, because 2.6 M
points will hang a browser if handed over in one response.

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PROCESSED  # noqa: E402

app = FastAPI(title="Thermal Anomaly Classification API", version="0.1.0")

STATIC = Path(__file__).parent / "static"
MAX_POINTS = 30_000


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        raise HTTPException(503, f"{name} not built yet -- run the pipeline first")
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def registry() -> pd.DataFrame:
    df = _read("registry.parquet")
    expl_path = PROCESSED / "explanations.parquet"
    if expl_path.exists():
        df = df.merge(pd.read_parquet(expl_path)[["cell_id", "top_reasons"]],
                      on="cell_id", how="left")
    return df


@lru_cache(maxsize=1)
def predictions() -> pd.DataFrame:
    return _read("predictions.parquet")


@lru_cache(maxsize=1)
def alerts() -> pd.DataFrame:
    return _read("alerts.parquet")


def to_geojson(df: pd.DataFrame, lat: str, lon: str) -> dict:
    """Frame -> GeoJSON FeatureCollection, NaNs dropped so JSON stays valid."""
    props = [c for c in df.columns if c not in (lat, lon)]
    feats = []
    for row in df.itertuples(index=False):
        d = dict(zip(df.columns, row))
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(d[lon]), float(d[lat])]},
            "properties": {k: (None if pd.isna(d[k]) else
                               (d[k].item() if hasattr(d[k], "item") else d[k]))
                           for k in props},
        })
    return {"type": "FeatureCollection", "features": feats}


@app.get("/api/stats")
def stats() -> dict:
    out: dict = {}
    try:
        reg = registry()
        out["registry_total"] = len(reg)
        out["registry_by_class"] = reg["pred_class"].value_counts().to_dict()
    except HTTPException:
        out["registry_total"] = 0
    try:
        pred = predictions()
        out["detections_total"] = len(pred)
        out["detections_by_class"] = pred["pred_class"].value_counts().to_dict()
        out["date_min"] = str(pred["acq_date"].min())
        out["date_max"] = str(pred["acq_date"].max())
    except HTTPException:
        out["detections_total"] = 0
    try:
        out["alerts_total"] = len(alerts())
    except HTTPException:
        out["alerts_total"] = 0
    return out


@app.get("/api/registry")
def get_registry(cls: str | None = None) -> dict:
    df = registry()
    if cls:
        df = df[df["pred_class"] == cls]
    return to_geojson(df, "lat", "lon")


@app.get("/api/alerts")
def get_alerts(limit: int = Query(500, le=5000)) -> dict:
    df = alerts().head(limit)
    return to_geojson(df, "latitude", "longitude")


@app.get("/api/detections")
def get_detections(
    west: float = 68.0, south: float = 6.0, east: float = 98.0, north: float = 38.0,
    start: str | None = None, end: str | None = None,
    cls: str | None = None,
    limit: int = Query(MAX_POINTS, le=MAX_POINTS),
) -> dict:
    df = predictions()
    m = (
        df["longitude"].between(west, east)
        & df["latitude"].between(south, north)
    )
    if start:
        m &= df["acq_date"] >= start
    if end:
        m &= df["acq_date"] <= end
    if cls:
        m &= df["pred_class"] == cls
    sub = df[m]
    truncated = len(sub) > limit
    if truncated:
        # Sample rather than head() so the map stays spatially representative.
        sub = sub.sample(limit, random_state=0)
    gj = to_geojson(sub, "latitude", "longitude")
    gj["truncated"] = truncated
    gj["matched"] = int(m.sum())
    return gj


@app.get("/api/cell/{cell_id}")
def get_cell(cell_id: str) -> dict:
    reg = registry()
    row = reg[reg["cell_id"] == cell_id]
    if row.empty:
        raise HTTPException(404, "cell not in registry")
    rec = row.iloc[0].to_dict()
    hist = predictions()
    h = hist[hist["cell_id"] == cell_id]
    rec["history"] = (
        h.groupby("acq_date")["frp"].max().reset_index()
        .rename(columns={"frp": "frp_max"}).to_dict("records")
    )
    return {k: (None if pd.isna(v) else v) for k, v in rec.items()
            if k != "history"} | {"history": rec["history"]}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(404, "UI not built")
    return FileResponse(page)
