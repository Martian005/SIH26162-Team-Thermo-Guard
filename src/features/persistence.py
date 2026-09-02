"""Separate persistent thermal sources from transient fire events.

This is the single most discriminative signal in the whole system. A gas flare
burns in the same 500 m cell on hundreds of nights a year; a wildfire burns for
days, moves, and never returns to exactly the same pixel. Crop-residue burning
is intense but collapses into a two-month seasonal window.

Two outputs:

  cells.parquet   one row per ~500 m grid cell, with recurrence, seasonality
                  and radiometric statistics. Cells above the persistence
                  thresholds become the Persistent Thermal Source Registry,
                  which the anomaly engine later baselines against.

  events.parquet  spatiotemporal DBSCAN clusters over the *transient* detections
                  only, giving each fire event a duration, footprint and
                  centroid drift.

Usage:
    python -m src.features.persistence
    python -m src.features.persistence --min-active-days 20 --min-months 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import GRID_DEG, INTERIM, PROCESSED  # noqa: E402

KM_PER_DEG_LAT = 111.32


def _row_entropy(counts: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy per row. 1.0 = perfectly uniform.

    Vectorised over the whole matrix: a per-row `.apply` over hundreds of
    thousands of cells is unusably slow.
    """
    counts = counts.astype(float)
    total = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    log_p = np.where(p > 0, np.log(p, out=np.zeros_like(p), where=p > 0), 0.0)
    h = -(p * log_p).sum(axis=1)
    h_max = np.log(counts.shape[1])
    return h / h_max if h_max > 0 else np.zeros(len(counts))


def add_grid(df: pd.DataFrame, grid_deg: float = GRID_DEG) -> pd.DataFrame:
    """Snap each detection to a fixed grid cell (~500 m at Indian latitudes)."""
    df = df.copy()
    df["cell_lat"] = (df["latitude"] / grid_deg).round().astype(int)
    df["cell_lon"] = (df["longitude"] / grid_deg).round().astype(int)
    df["cell_id"] = (
        df["cell_lat"].astype(str) + "_" + df["cell_lon"].astype(str)
    )
    return df


def cell_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detections into per-cell recurrence and radiometric stats."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["acq_date"])
    df["month"] = df["date"].dt.month
    df["is_night"] = df["daynight"].eq("N")

    total_span_days = (df["date"].max() - df["date"].min()).days + 1

    g = df.groupby("cell_id")

    stats = pd.DataFrame(
        {
            "n_detections": g.size(),
            "n_active_days": g["date"].nunique(),
            "first_seen": g["date"].min(),
            "last_seen": g["date"].max(),
            "n_months": g["month"].nunique(),
            "centroid_lat": g["latitude"].mean(),
            "centroid_lon": g["longitude"].mean(),
            "spread_lat_deg": g["latitude"].std().fillna(0.0),
            "spread_lon_deg": g["longitude"].std().fillna(0.0),
            "frp_mean": g["frp"].mean(),
            "frp_median": g["frp"].median(),
            "frp_std": g["frp"].std().fillna(0.0),
            "frp_max": g["frp"].max(),
            "frp_p95": g["frp"].quantile(0.95),
            "ti4_mean": g["bright_ti4"].mean(),
            "ti4_std": g["bright_ti4"].std().fillna(0.0),
            "ti5_mean": g["bright_ti5"].mean(),
            "n_night": g["is_night"].sum(),
        }
    )

    stats["ti4_ti5_diff"] = stats["ti4_mean"] - stats["ti5_mean"]
    stats["night_ratio"] = stats["n_night"] / stats["n_detections"]
    stats["lifetime_days"] = (
        stats["last_seen"] - stats["first_seen"]
    ).dt.days + 1
    stats["duty_cycle"] = stats["n_active_days"] / stats["lifetime_days"]
    stats["coverage_of_record"] = stats["n_active_days"] / total_span_days

    # Radiometric stability: flares hold a near-constant FRP, wildfires do not.
    stats["frp_cv"] = (stats["frp_std"] / stats["frp_mean"]).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    # Spatial spread in metres -- a fixed installation barely moves.
    lat_rad = np.radians(stats["centroid_lat"])
    stats["spread_m"] = np.hypot(
        stats["spread_lat_deg"] * KM_PER_DEG_LAT * 1000,
        stats["spread_lon_deg"] * KM_PER_DEG_LAT * 1000 * np.cos(lat_rad),
    )

    # Seasonality: month-of-year entropy. Industrial sources burn year round
    # (entropy near 1); stubble burning spikes in two months (entropy near 0).
    month_counts = (
        df.groupby(["cell_id", "month"]).size().unstack(fill_value=0).reindex(
            columns=range(1, 13), fill_value=0
        )
    )
    month_counts = month_counts.reindex(stats.index)
    stats["month_entropy"] = _row_entropy(month_counts.to_numpy())
    stats["peak_month"] = month_counts.idxmax(axis=1)
    stats["peak_month_share"] = (
        month_counts.max(axis=1) / month_counts.sum(axis=1)
    )

    # FIRMS `type` exists only on archive products. Fractions per type are both
    # cheaper than a modal aggregation and more informative for the classifier.
    if "type" in df.columns:
        for code, label in ((0, "veg"), (2, "static"), (3, "offshore")):
            stats[f"firms_type{code}_frac"] = (
                df.assign(_m=df["type"].eq(code)).groupby("cell_id")["_m"].mean()
            )

    return stats.reset_index()


def mark_persistent(
    cells: pd.DataFrame, min_active_days: int, min_months: int, max_spread_m: float
) -> pd.DataFrame:
    """Flag cells that behave like fixed installations rather than fires."""
    cells = cells.copy()
    cells["is_persistent"] = (
        (cells["n_active_days"] >= min_active_days)
        & (cells["n_months"] >= min_months)
        & (cells["spread_m"] <= max_spread_m)
    )
    return cells


def _cluster_chunk(df: pd.DataFrame, eps_km: float, eps_days: float) -> np.ndarray:
    """Run one spatiotemporal DBSCAN and return raw labels.

    Space and time are put on a common scale so one `eps` covers both: distance
    is expressed in km, and time is rescaled so `eps_days` equals `eps_km`.
    """
    lat0 = float(df["latitude"].mean())
    x_km = df["longitude"].to_numpy() * KM_PER_DEG_LAT * np.cos(np.radians(lat0))
    y_km = df["latitude"].to_numpy() * KM_PER_DEG_LAT
    day = pd.to_datetime(df["acq_date"]).astype("int64").to_numpy() / (86400 * 1e9)
    t_km = (day - day.min()) * (eps_km / eps_days)

    coords = np.column_stack([x_km, y_km, t_km])
    return DBSCAN(eps=eps_km, min_samples=3, algorithm="ball_tree").fit_predict(coords)


def cluster_events(df: pd.DataFrame, eps_km: float, eps_days: float) -> pd.DataFrame:
    """Spatiotemporal DBSCAN over transient detections to recover fire events.

    DBSCAN is superlinear, so a single pass over millions of points is not
    viable. Detections are clustered one calendar month at a time, which keeps
    each pass small. Events straddling a month boundary get split; nearly all
    fire events are far shorter than a month, so the cost is acceptable.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["_ym"] = pd.to_datetime(df["acq_date"]).dt.to_period("M")

    pieces, offset = [], 0
    for _, chunk in tqdm(df.groupby("_ym", sort=True), desc="events/month"):
        if len(chunk) < 3:
            continue
        labels = _cluster_chunk(chunk, eps_km, eps_days)
        chunk = chunk.assign(event_id=np.where(labels >= 0, labels + offset, -1))
        pieces.append(chunk)
        if (labels >= 0).any():
            offset += int(labels.max()) + 1

    if not pieces:
        return pd.DataFrame()

    df = pd.concat(pieces, ignore_index=True)
    clustered = df[df["event_id"] >= 0]
    if clustered.empty:
        return pd.DataFrame()

    # Every aggregation below must be a built-in reducer. A lambda here runs
    # per group in Python and stalls for minutes across hundreds of thousands
    # of events.
    clustered = clustered.assign(_night=clustered["daynight"].eq("N"))
    events = clustered.groupby("event_id").agg(
        n_detections=("latitude", "size"),
        start=("acq_date", "min"),
        end=("acq_date", "max"),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        lat_min=("latitude", "min"),
        lat_max=("latitude", "max"),
        lon_min=("longitude", "min"),
        lon_max=("longitude", "max"),
        frp_sum=("frp", "sum"),
        frp_max=("frp", "max"),
        frp_mean=("frp", "mean"),
        ti4_mean=("bright_ti4", "mean"),
        ti5_mean=("bright_ti5", "mean"),
        night_frac=("_night", "mean"),
    ).reset_index()

    events["lat_span_deg"] = events["lat_max"] - events["lat_min"]
    events["lon_span_deg"] = events["lon_max"] - events["lon_min"]
    events["duration_days"] = (
        pd.to_datetime(events["end"]) - pd.to_datetime(events["start"])
    ).dt.days + 1
    events["extent_km"] = np.hypot(
        events["lat_span_deg"] * KM_PER_DEG_LAT,
        events["lon_span_deg"] * KM_PER_DEG_LAT
        * np.cos(np.radians(events["centroid_lat"])),
    )
    # A spreading front is the wildfire signature; flares score near zero.
    events["spread_rate_km_per_day"] = events["extent_km"] / events["duration_days"]

    return events


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistence clustering + event detection")
    ap.add_argument("--detections", default=str(INTERIM / "firms_detections.parquet"))
    ap.add_argument("--min-active-days", type=int, default=15)
    ap.add_argument("--min-months", type=int, default=4)
    ap.add_argument("--max-spread-m", type=float, default=1500.0)
    ap.add_argument("--eps-km", type=float, default=1.0)
    ap.add_argument("--eps-days", type=float, default=2.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.detections)
    print(f"loaded {len(df):,} detections")

    df = add_grid(df)
    print(f"{df['cell_id'].nunique():,} distinct ~500 m cells")

    cells = cell_statistics(df)
    cells = mark_persistent(
        cells, args.min_active_days, args.min_months, args.max_spread_m
    )

    n_persist = int(cells["is_persistent"].sum())
    print(f"\npersistent cells: {n_persist:,} of {len(cells):,} "
          f"({100 * n_persist / len(cells):.2f}%)")

    persistent_ids = set(cells.loc[cells["is_persistent"], "cell_id"])
    transient = df[~df["cell_id"].isin(persistent_ids)]
    print(f"transient detections: {len(transient):,} of {len(df):,}")

    events = cluster_events(transient, args.eps_km, args.eps_days)
    print(f"fire events clustered: {len(events):,}")

    cells.to_parquet(PROCESSED / "cells.parquet", index=False)
    events.to_parquet(PROCESSED / "events.parquet", index=False)
    df.to_parquet(INTERIM / "firms_gridded.parquet", index=False)

    print("\n--- persistent cell profile (median) ---")
    cols = ["n_active_days", "n_months", "duty_cycle", "night_ratio",
            "month_entropy", "frp_mean", "frp_cv", "spread_m"]
    print(cells.groupby("is_persistent")[cols].median().T.to_string())

    if not events.empty:
        print("\n--- event profile (median) ---")
        print(events[["n_detections", "duration_days", "extent_km",
                      "spread_rate_km_per_day", "night_frac", "frp_max"]]
              .median().to_string())


if __name__ == "__main__":
    main()
