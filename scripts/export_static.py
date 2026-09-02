"""Export the pipeline output as a static site.

The FastAPI service reads multi-hundred-megabyte parquet files off local disk.
That model cannot go on a serverless host: the function bundle limit is far
smaller than the data, there is no persistent filesystem, and loading pandas
plus 80 MB of parquet blows the cold-start timeout.

So for hosting, the layers are precomputed once into static GeoJSON. The result
is a few megabytes of files any static host serves for free, with no backend at
all. The map UI is unchanged -- it probes for the API and falls back to these
files automatically.

Usage:
    python scripts/export_static.py
    python scripts/export_static.py --max-detections 60000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED, ROOT  # noqa: E402

STATIC_SRC = ROOT / "src" / "api" / "static" / "index.html"


def to_geojson(df: pd.DataFrame, lat: str, lon: str, props: list[str]) -> dict:
    """Compact FeatureCollection; NaN dropped so the JSON stays valid."""
    keep = [c for c in props if c in df.columns]
    feats = []
    for row in df[keep + [lat, lon]].itertuples(index=False):
        d = dict(zip(keep + [lat, lon], row))
        geom = [round(float(d[lon]), 5), round(float(d[lat]), 5)]
        p = {}
        for k in keep:
            v = d[k]
            if pd.isna(v):
                continue
            if hasattr(v, "item"):
                v = v.item()
            p[k] = round(v, 4) if isinstance(v, float) else v
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": geom},
                      "properties": p})
    return {"type": "FeatureCollection", "features": feats}


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  {path.name:24s} {path.stat().st_size / 1e6:6.2f} MB "
          f"({len(payload.get('features', [])):,} features)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export static site for hosting")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--max-detections", type=int, default=40000)
    args = ap.parse_args()

    out = Path(args.out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    print(f"exporting to {out}")

    # --- registry (+ SHAP explanations) ---
    reg = pd.read_parquet(PROCESSED / "registry.parquet")
    expl_path = PROCESSED / "explanations.parquet"
    if expl_path.exists():
        reg = reg.merge(pd.read_parquet(expl_path)[["cell_id", "top_reasons"]],
                        on="cell_id", how="left")
    for c in ("first_seen", "last_seen"):
        if c in reg.columns:
            reg[c] = reg[c].astype(str)
    write(data / "registry.geojson", to_geojson(
        reg, "lat", "lon",
        ["cell_id", "pred_class", "n_detections", "frp_mean", "frp_max",
         "night_frac", "first_seen", "last_seen", "class_agreement", "top_reasons"]))

    # --- alerts ---
    alerts = pd.read_parquet(PROCESSED / "alerts.parquet")
    alerts["acq_date"] = alerts["acq_date"].astype(str)
    write(data / "alerts.geojson", to_geojson(
        alerts.nlargest(2000, "severity"), "latitude", "longitude",
        ["cell_id", "acq_date", "frp", "bright_ti4", "daynight",
         "alert_type", "severity"]))

    # --- detections: stratified sample so rare classes stay visible ---
    pred = pd.read_parquet(PROCESSED / "predictions.parquet")
    n_cls = pred["pred_class"].nunique()
    per = max(1, args.max_detections // n_cls)
    sample = (pred.groupby("pred_class", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per), random_state=0)))
    sample["acq_date"] = sample["acq_date"].astype(str)
    write(data / "detections.geojson", to_geojson(
        sample, "latitude", "longitude",
        ["acq_date", "frp", "bright_ti4", "daynight", "pred_class", "pred_conf"]))

    # --- stats ---
    stats = {
        "registry_total": int(len(reg)),
        "registry_by_class": reg["pred_class"].value_counts().to_dict(),
        "detections_total": int(len(pred)),
        "detections_by_class": pred["pred_class"].value_counts().to_dict(),
        "date_min": str(pred["acq_date"].min()),
        "date_max": str(pred["acq_date"].max()),
        "alerts_total": int(len(alerts)),
        "static_sample": int(len(sample)),
    }
    write(data / "stats.json", stats)

    shutil.copy(STATIC_SRC, out / "index.html")
    print(f"  index.html copied")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\ntotal: {total / 1e6:.1f} MB -> {out}")
    print("\nDeploy with:  cd dist && vercel --prod")
    print("(or drag the dist folder onto app.netlify.com/drop)")


if __name__ == "__main__":
    main()
