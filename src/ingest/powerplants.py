"""Fetch the WRI Global Power Plant Database and keep Indian thermal plants.

Thermal (coal / gas / oil) plants emit a steady, predictable heat signature, so
they are both a weak-label source for the `thermal_power` class and a context
feature ("how far is this detection from the nearest coal plant?").

Usage:
    python -m src.ingest.powerplants
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import INDIA_BBOX, INTERIM, RAW  # noqa: E402

SOURCES = [
    "https://raw.githubusercontent.com/wri/global-power-plant-database/master/"
    "output_database/global_power_plant_database.csv",
    "https://datasets.wri.org/dataset/540dcf46-f287-47ac-985d-269b04bea4c6/"
    "resource/c240ed2e-1190-4d7e-b1da-c66b72e08858/download/globalpowerplantdatabasev130.zip",
]

# fuels that produce a persistent thermal signature detectable from orbit
THERMAL_FUELS = {"Coal", "Gas", "Oil", "Petcoke", "Waste", "Biomass"}


def download() -> pd.DataFrame:
    """Try each mirror until one yields a parseable CSV."""
    dest = RAW / "global_power_plants.csv"
    if dest.exists() and dest.stat().st_size > 0:
        return pd.read_csv(dest, low_memory=False)

    last_err = None
    for url in SOURCES:
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            if url.endswith(".csv"):
                dest.write_bytes(resp.content)
                return pd.read_csv(dest, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - mirror may be down or moved
            last_err = exc
            print(f"  [warn] {url.split('/')[2]}: {exc}")

    raise SystemExit(f"All power plant mirrors failed. Last error: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch WRI power plant database")
    ap.add_argument("--out", default=str(INTERIM / "power_plants.parquet"))
    args = ap.parse_args()

    df = download()
    print(f"global rows: {len(df):,}")

    w, s, e, n = INDIA_BBOX
    df = df[
        df["latitude"].between(s, n)
        & df["longitude"].between(w, e)
    ].copy()
    print(f"in study bbox: {len(df):,}")

    keep = [
        "name", "country", "capacity_mw", "latitude", "longitude",
        "primary_fuel", "other_fuel1", "commissioning_year", "owner",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df["is_thermal"] = df["primary_fuel"].isin(THERMAL_FUELS)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    out_path = Path(args.out)
    gdf.to_parquet(out_path, index=False)

    print(f"\nwrote {len(gdf):,} plants -> {out_path}")
    print(f"thermal plants: {int(gdf['is_thermal'].sum()):,}")
    print("\nby primary fuel:")
    print(gdf["primary_fuel"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
