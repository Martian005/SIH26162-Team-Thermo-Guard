"""Run the whole pipeline end to end.

Each stage is skipped if its output already exists, so this is safe to re-run
after a failure. Pass --force to rebuild everything.

Usage:
    python run_pipeline.py
    python run_pipeline.py --from context
    python run_pipeline.py --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from config import INTERIM, MODELS, PROCESSED, ROOT

PYTHON = sys.executable

# name, module, args, output that marks it done
STAGES = [
    ("firms",      "src.ingest.firms",       [], INTERIM / "firms_detections.parquet"),
    ("plants",     "src.ingest.powerplants", [], INTERIM / "power_plants.parquet"),
    ("osm",        "src.ingest.osm",         [], INTERIM / "osm_infrastructure.parquet"),
    ("persistence","src.features.persistence",[], PROCESSED / "cells.parquet"),
    ("landcover",  "src.features.landcover", [], INTERIM / "landcover.parquet"),
    ("context",    "src.features.context",   [], INTERIM / "context.parquet"),
    ("dataset",    "src.model.dataset",      [], PROCESSED / "training_data.parquet"),
    # Sampled: 6-class LightGBM over 5 spatially blocked folds on the full
    # ~1.5 M labelled rows runs far longer without moving the confusion matrix.
    ("train",      "src.model.train",        ["--sample", "300000", "--folds", "3"],
     MODELS / "classifier.txt"),
    ("predict",    "src.model.predict",      [], PROCESSED / "predictions.parquet"),
    ("anomaly",    "src.model.anomaly",      [], PROCESSED / "alerts.parquet"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full pipeline")
    ap.add_argument("--force", action="store_true", help="rebuild even if output exists")
    ap.add_argument("--from", dest="start_at", help="begin at this stage")
    ap.add_argument("--only", help="run just this stage")
    args = ap.parse_args()

    names = [s[0] for s in STAGES]
    if args.only and args.only not in names:
        raise SystemExit(f"unknown stage {args.only!r}; choose from {names}")
    if args.start_at and args.start_at not in names:
        raise SystemExit(f"unknown stage {args.start_at!r}; choose from {names}")

    started = args.start_at is None
    for name, module, extra, marker in STAGES:
        if args.only and name != args.only:
            continue
        if not started:
            if name != args.start_at:
                continue
            started = True

        if marker.exists() and not args.force and not args.only:
            print(f"[skip] {name:12s} -> {marker.name} exists")
            continue

        print(f"\n{'=' * 62}\n[run ] {name}\n{'=' * 62}")
        t0 = time.time()
        rc = subprocess.call([PYTHON, "-u", "-m", module, *extra], cwd=ROOT)
        if rc != 0:
            raise SystemExit(f"stage {name!r} failed with exit code {rc}")
        print(f"[done] {name} in {time.time() - t0:.1f}s")

    print("\nPipeline complete. Serve the UI with:")
    print("  uvicorn src.api.main:app --port 8000")


if __name__ == "__main__":
    main()
