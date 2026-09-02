"""Flag candidate accidental industrial fires against the persistent-source baseline.

Classification alone is academic. The operational question a disaster-management
desk actually asks is: *is anything burning today that should not be?*

The Persistent Thermal Source Registry answers it. Every routine flare, kiln and
boiler in the country already has a measured FRP distribution. Against that
baseline two things are genuinely anomalous:

  ESCALATION   heat at a known industrial site far above its own normal range.
               A refinery flare running at 40 MW when its two-year 95th
               percentile is 6 MW is not a flare any more.

  NEW SOURCE   heat inside an industrial footprint that has no detection history
               at all. Nothing routine starts burning for the first time.

Both are scored by severity so an operator sees the worst first. Note this
deliberately baselines each site against *itself* -- a steel plant and a small
foundry have very different normal ranges, and a single national FRP threshold
would drown in false positives from the former while missing the latter.

Usage:
    python -m src.model.anomaly
    python -m src.model.anomaly --since 2026-03-01 --z-threshold 3.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import INTERIM, PROCESSED  # noqa: E402

MIN_BASELINE_DAYS = 10  # below this the baseline is too thin to trust


def build_registry(cells: pd.DataFrame) -> pd.DataFrame:
    """Persistent sources with a usable FRP baseline."""
    reg = cells[cells["is_persistent"] & (cells["n_active_days"] >= MIN_BASELINE_DAYS)]
    keep = [
        "cell_id", "centroid_lat", "centroid_lon", "n_detections", "n_active_days",
        "n_months", "night_ratio", "frp_mean", "frp_std", "frp_p95", "frp_max",
        "first_seen", "last_seen",
    ]
    return reg[[c for c in keep if c in reg.columns]].copy()


def score_escalations(
    det: pd.DataFrame, registry: pd.DataFrame, z_threshold: float
) -> pd.DataFrame:
    """Detections at known sources burning far above that source's own normal."""
    merged = det.merge(registry, on="cell_id", how="inner", suffixes=("", "_base"))
    if merged.empty:
        return merged

    # A near-zero sigma would make every wobble a 100-sigma event; floor it.
    sigma = merged["frp_std"].fillna(0).clip(lower=0.5)
    merged["frp_z"] = (merged["frp"] - merged["frp_mean"]) / sigma
    merged["frp_ratio_p95"] = merged["frp"] / merged["frp_p95"].clip(lower=0.1)

    hit = merged[
        (merged["frp_z"] >= z_threshold) & (merged["frp_ratio_p95"] >= 2.0)
    ].copy()
    hit["alert_type"] = "escalation_at_known_source"
    hit["severity"] = np.clip(hit["frp_z"] / 10.0, 0, 1) * 0.6 + np.clip(
        hit["frp_ratio_p95"] / 20.0, 0, 1
    ) * 0.4
    return hit


def score_new_sources(
    det: pd.DataFrame, registry: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    """Heat inside an industrial footprint with no history at that cell."""
    known = set(registry["cell_id"])
    fresh = det[~det["cell_id"].isin(known)].copy()
    if fresh.empty or context is None:
        return pd.DataFrame()

    ctx_cols = [c for c in ["cell_id", "inside_industrial", "dist_any_industry_km",
                            "dist_refinery_km", "dist_thermal_plant_km"]
                if c in context.columns]
    fresh = fresh.merge(context[ctx_cols], on="cell_id", how="left")

    inside = fresh.get("inside_industrial", pd.Series(False, index=fresh.index))
    near = fresh.get("dist_any_industry_km", pd.Series(999.0, index=fresh.index)) < 1.0

    hit = fresh[inside.fillna(False).astype(bool) | near.fillna(False)].copy()
    if hit.empty:
        return hit

    hit["alert_type"] = "new_source_at_industrial_site"
    # Severity driven by raw intensity, since there is no baseline to compare to.
    hit["severity"] = np.clip(hit["frp"] / 50.0, 0, 1)
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect anomalous industrial thermal events")
    ap.add_argument("--detections", default=str(INTERIM / "firms_gridded.parquet"))
    ap.add_argument("--cells", default=str(PROCESSED / "cells.parquet"))
    ap.add_argument("--context", default=str(INTERIM / "context.parquet"))
    ap.add_argument("--since", default="2026-01-01",
                    help="treat detections on/after this date as the live window")
    ap.add_argument("--z-threshold", type=float, default=3.0)
    ap.add_argument("--out", default=str(PROCESSED / "alerts.parquet"))
    args = ap.parse_args()

    cells = pd.read_parquet(args.cells)
    registry = build_registry(cells)
    print(f"registry: {len(registry):,} persistent sources with a usable baseline")

    det = pd.read_parquet(args.detections)
    live = det[det["acq_date"] >= args.since].copy()
    print(f"live window {args.since}+: {len(live):,} detections")

    ctx_path = Path(args.context)
    context = pd.read_parquet(ctx_path) if ctx_path.exists() else None
    if context is None:
        print("  [warn] context.parquet missing -- new-source rule disabled")

    esc = score_escalations(live, registry, args.z_threshold)
    print(f"escalations at known sources : {len(esc):,}")

    new = score_new_sources(live, registry, context)
    print(f"new sources at industrial sites: {len(new):,}")

    cols = ["cell_id", "latitude", "longitude", "acq_date", "acq_time", "frp",
            "bright_ti4", "daynight", "alert_type", "severity"]
    parts = [d[[c for c in cols if c in d.columns]] for d in (esc, new) if len(d)]
    if not parts:
        print("no alerts in window")
        return

    alerts = pd.concat(parts, ignore_index=True).sort_values(
        "severity", ascending=False
    )
    out_path = Path(args.out)
    alerts.to_parquet(out_path, index=False)

    print(f"\nwrote {len(alerts):,} alerts -> {out_path}")
    print("\nby type:")
    print(alerts["alert_type"].value_counts().to_string())
    print("\ntop 15 by severity:")
    print(alerts.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
