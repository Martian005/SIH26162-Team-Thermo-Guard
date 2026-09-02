"""Train the 6-class thermal anomaly classifier.

Two design decisions carry most of the methodological weight:

1. **Spatially blocked cross-validation.** Detections cluster hard in space --
   Jharia alone contributes tens of thousands. A random split would put the same
   coal fire in train and test and report a fantasy score. Folds are grouped by
   ~1-degree spatial block so every evaluation is on ground the model never saw.

2. **Leakage guards.** Any column that fed the weak-label rules is barred from
   the feature set. `type` / `firms_type*_frac` are the sharpest example: NASA's
   static-source flag helped *write* the gas_flare labels, so training on it
   would score our own rules back to us. It is also absent from NRT data, which
   is what the deployed system actually sees.

Accuracy is not reported as a headline: the classes are wildly imbalanced, so
per-class precision/recall and the confusion matrix are what matter.

Usage:
    python -m src.model.train
    python -m src.model.train --folds 5 --sample 500000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import MODELS, PROCESSED  # noqa: E402

# Columns barred from features: identifiers, raw geography, and anything that
# participated in weak labelling.
BLOCKED_EXACT = {
    "cell_id", "cell_lat", "cell_lon", "latitude", "longitude",
    "acq_date", "acq_time", "acq_dt", "satellite", "instrument", "version",
    "sensor", "daynight", "confidence", "weak_label", "label_confidence",
    "type", "doy", "first_seen", "last_seen", "peak_month",
    "nearest_plant_fuel", "block",
}
BLOCKED_PREFIX = ("firms_type",)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric, non-leaking columns."""
    cols = []
    for c in df.columns:
        if c in BLOCKED_EXACT or c.startswith(BLOCKED_PREFIX):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def spatial_blocks(df: pd.DataFrame, deg: float = 1.0) -> pd.Series:
    """~110 km blocks used to group CV folds."""
    return (
        (df["latitude"] / deg).astype(int).astype(str)
        + "_"
        + (df["longitude"] / deg).astype(int).astype(str)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train thermal anomaly classifier")
    ap.add_argument("--data", default=str(PROCESSED / "training_data.parquet"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0, help="0 = use all labelled rows")
    ap.add_argument("--block-deg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = df[df["weak_label"].notna()].reset_index(drop=True)
    print(f"labelled rows: {len(df):,}")

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"sampled to   : {len(df):,}")

    df["block"] = spatial_blocks(df, args.block_deg)
    print(f"spatial blocks: {df['block'].nunique():,}")

    feats = feature_columns(df)
    print(f"features      : {len(feats)}")

    X = df[feats].astype(np.float32)
    classes = sorted(df["weak_label"].unique())
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y = df["weak_label"].map(cls_to_i).to_numpy()
    w = df["label_confidence"].fillna(0.5).to_numpy()

    # Counter the extreme class imbalance on top of the confidence weights.
    counts = np.bincount(y, minlength=len(classes))
    balance = len(y) / (len(classes) * np.maximum(counts, 1))
    w = w * balance[y]

    print("\nclass counts:")
    for c, n in zip(classes, counts):
        print(f"  {c:20s} {n:>9,}")

    gkf = GroupKFold(n_splits=args.folds)
    oof = np.zeros((len(df), len(classes)), dtype=np.float32)

    params = dict(
        objective="multiclass",
        num_class=len(classes),
        learning_rate=0.08,
        num_leaves=96,
        min_data_in_leaf=100,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        max_bin=255,
        verbosity=-1,
        seed=args.seed,
        num_threads=0,
    )

    models = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=df["block"]), 1):
        dtrain = lgb.Dataset(X.iloc[tr], y[tr], weight=w[tr])
        dvalid = lgb.Dataset(X.iloc[va], y[va], weight=w[va], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=600,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        models.append(booster)
        print(f"fold {fold}: best_iter={booster.best_iteration} "
              f"train={len(tr):,} valid={len(va):,}")

    pred = oof.argmax(axis=1)

    print("\n=== spatially-blocked out-of-fold report ===")
    print(classification_report(y, pred, target_names=classes, digits=3, zero_division=0))

    print("=== confusion matrix (rows = true) ===")
    cm = confusion_matrix(y, pred)
    hdr = "".join(f"{c[:11]:>12s}" for c in classes)
    print(f"{'':22s}{hdr}")
    for i, c in enumerate(classes):
        print(f"{c:22s}" + "".join(f"{v:>12,}" for v in cm[i]))

    # Retrain on everything for the deployed artefact.
    final = lgb.train(
        params,
        lgb.Dataset(X, y, weight=w),
        num_boost_round=int(np.mean([m.best_iteration for m in models])),
    )
    final.save_model(str(MODELS / "classifier.txt"))

    meta = {
        "classes": classes,
        "features": feats,
        "n_train": int(len(df)),
        "folds": args.folds,
        "block_deg": args.block_deg,
    }
    (MODELS / "classifier_meta.json").write_text(json.dumps(meta, indent=2))

    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(
        ascending=False
    )
    print("\n=== top 25 features by gain ===")
    print((100 * imp / imp.sum()).head(25).round(2).to_string())

    imp.to_frame("gain").to_csv(MODELS / "feature_importance.csv")
    print(f"\nsaved model -> {MODELS / 'classifier.txt'}")


if __name__ == "__main__":
    main()
