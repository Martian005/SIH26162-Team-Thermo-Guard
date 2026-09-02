"""Score every detection and produce the map-ready layers.

Three artefacts come out of this:

  predictions.parquet  class + confidence for every detection
  registry.parquet     each persistent thermal source with its dominant class,
                       i.e. the national inventory of "things that are always hot"
  explanations.parquet per-source SHAP attributions, so the UI can answer
                       "why did you call this a gas flare?" rather than asserting it

SHAP is computed only for the registry and the alert set. Running it across 2.6 M
detections would cost hours and nobody inspects an explanation for a routine
crop fire.

Usage:
    python -m src.model.predict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import MODELS, PROCESSED  # noqa: E402

TOP_REASONS = 4


def load_model() -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(MODELS / "classifier.txt"))
    meta = json.loads((MODELS / "classifier_meta.json").read_text())
    return booster, meta


def explain(booster: lgb.Booster, X: pd.DataFrame, classes: list[str],
            pred: np.ndarray) -> list[str]:
    """Top contributing features for each row's predicted class.

    LightGBM's `pred_contrib` gives exact SHAP values for trees without needing
    the shap package's slower model-agnostic path.
    """
    contrib = booster.predict(X, pred_contrib=True)
    n_feat = X.shape[1]
    # Layout is (n_features + 1) blocks per class, trailing entry is the bias.
    contrib = contrib.reshape(len(X), len(classes), n_feat + 1)

    names = np.array(X.columns)
    out = []
    for i, cls_idx in enumerate(pred):
        vals = contrib[i, cls_idx, :n_feat]
        order = np.argsort(-np.abs(vals))[:TOP_REASONS]
        parts = [
            f"{names[j]}={X.iat[i, j]:.4g} ({'+' if vals[j] > 0 else ''}{vals[j]:.2f})"
            for j in order
        ]
        out.append("; ".join(parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score detections and build map layers")
    ap.add_argument("--data", default=str(PROCESSED / "training_data.parquet"))
    ap.add_argument("--batch", type=int, default=400_000)
    args = ap.parse_args()

    booster, meta = load_model()
    classes = meta["classes"]
    feats = meta["features"]
    print(f"model: {len(classes)} classes, {len(feats)} features")

    df = pd.read_parquet(args.data)
    print(f"scoring {len(df):,} detections")

    # Batched so peak memory stays bounded on a laptop.
    pred_idx = np.empty(len(df), dtype=np.int8)
    pred_prob = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), args.batch):
        stop = min(start + args.batch, len(df))
        Xb = df.iloc[start:stop][feats].astype(np.float32)
        proba = booster.predict(Xb)
        pred_idx[start:stop] = proba.argmax(axis=1)
        pred_prob[start:stop] = proba.max(axis=1)
        print(f"  {stop:,}/{len(df):,}")

    df["pred_class"] = [classes[i] for i in pred_idx]
    df["pred_conf"] = pred_prob

    keep = ["cell_id", "latitude", "longitude", "acq_date", "acq_time", "daynight",
            "frp", "bright_ti4", "bright_ti5", "is_persistent",
            "weak_label", "pred_class", "pred_conf"]
    preds = df[[c for c in keep if c in df.columns]]
    preds.to_parquet(PROCESSED / "predictions.parquet", index=False)
    print(f"\nwrote predictions.parquet ({len(preds):,} rows)")

    print("\npredicted class distribution:")
    vc = df["pred_class"].value_counts()
    for c, n in vc.items():
        print(f"  {c:20s} {n:>9,}  {100 * n / len(df):5.1f}%")

    # --- registry: one row per persistent source, with its dominant class ---
    pers = df[df["is_persistent"].fillna(False).astype(bool)]
    if pers.empty:
        print("\nno persistent cells -- registry not written")
        return

    dominant = (
        pers.groupby(["cell_id", "pred_class"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False)
        .drop_duplicates("cell_id")
    )
    agg = pers.groupby("cell_id").agg(
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        n_detections=("frp", "size"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
        night_frac=("daynight", lambda s: s.eq("N").mean()),
        first_seen=("acq_date", "min"),
        last_seen=("acq_date", "max"),
    ).reset_index()

    registry = agg.merge(
        dominant[["cell_id", "pred_class", "n"]], on="cell_id", how="left"
    ).rename(columns={"n": "n_class_votes"})
    registry["class_agreement"] = registry["n_class_votes"] / registry["n_detections"]
    registry.to_parquet(PROCESSED / "registry.parquet", index=False)
    print(f"\nwrote registry.parquet ({len(registry):,} persistent sources)")
    print("\nregistry by class:")
    print(registry["pred_class"].value_counts().to_string())

    # --- SHAP explanations for the registry ---
    rep = pers.drop_duplicates("cell_id").set_index("cell_id").loc[registry["cell_id"]]
    Xr = rep[feats].astype(np.float32).reset_index(drop=True)
    idx = np.array([classes.index(c) for c in registry["pred_class"]])
    expl = pd.DataFrame({
        "cell_id": registry["cell_id"].to_numpy(),
        "pred_class": registry["pred_class"].to_numpy(),
        "top_reasons": explain(booster, Xr, classes, idx),
    })
    expl.to_parquet(PROCESSED / "explanations.parquet", index=False)
    print(f"wrote explanations.parquet ({len(expl):,} rows)")
    print("\nexample explanations:")
    print(expl.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
