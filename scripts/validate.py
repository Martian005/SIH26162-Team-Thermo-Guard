"""Ground-truth spot check against known Indian industrial sites.

Run this live in front of judges. It takes a list of facilities the system was
never told about and asks what the pipeline independently concluded there.

Usage:
    python scripts/validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED  # noqa: E402

EARTH_R_KM = 6371.0

# Facilities never supplied to the model -- used only to interrogate its output.
SITES = {
    "Jamnagar refinery (RIL)":   (22.34, 69.86, "gas_flare / industrial"),
    "Hazira complex, Surat":     (21.10, 72.64, "gas_flare / industrial"),
    "Koyali refinery, Vadodara": (22.35, 73.14, "gas_flare / industrial"),
    "Jharia coalfield, Dhanbad": (23.75, 86.42, "mining (coal seam fire)"),
    "Korba thermal":             (22.36, 82.68, "thermal_power"),
    "Singrauli thermal":         (24.10, 82.67, "thermal_power"),
    "Bhilai steel":              (21.20, 81.38, "industrial"),
    "Angul / NALCO, Odisha":     (20.79, 85.25, "industrial"),
    "Paradip refinery":          (20.26, 86.67, "gas_flare / industrial"),
    "Mundra port & power":       (22.83, 69.72, "thermal_power"),
    "Ghazipur landfill, Delhi":  (28.62, 77.33, "episodic - expect none"),
    "Ludhiana cropland, Punjab": (30.90, 75.85, "agricultural_burn"),
    "Nainital forest, U'khand":  (29.38, 79.45, "wildfire"),
}


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def main() -> None:
    cells = pd.read_parquet(PROCESSED / "cells.parquet")
    pers = cells[cells["is_persistent"]]
    print(f"persistent thermal sources in registry: {len(pers):,}\n")

    reg_path = PROCESSED / "registry.parquet"
    registry = pd.read_parquet(reg_path) if reg_path.exists() else None

    hdr = f"{'site':28s}{'cells<=10km':>12s}{'nearest':>9s}{'max days':>10s}"
    if registry is not None:
        hdr += f"  {'predicted':<18s}{'expected':<24s}"
    else:
        hdr += f"  {'expected':<24s}"
    print(hdr)
    print("-" * len(hdr))

    for name, (lat, lon, expected) in SITES.items():
        d = haversine_km(lat, lon, pers["centroid_lat"].to_numpy(),
                         pers["centroid_lon"].to_numpy())
        near = pers[d <= 10]
        line = (f"{name:28s}{len(near):>12d}{d.min():>8.1f}km"
                f"{(near['n_active_days'].max() if len(near) else 0):>10.0f}")

        if registry is not None:
            if len(near):
                rd = haversine_km(lat, lon, registry["lat"].to_numpy(),
                                  registry["lon"].to_numpy())
                sub = registry[rd <= 10]
                pred = (sub["pred_class"].mode().iat[0]
                        if len(sub) and not sub["pred_class"].mode().empty else "-")
            else:
                pred = "-"
            line += f"  {pred:<18s}{expected:<24s}"
        else:
            line += f"  {expected:<24s}"
        print(line)

    alerts_path = PROCESSED / "alerts.parquet"
    if alerts_path.exists():
        alerts = pd.read_parquet(alerts_path)
        print(f"\nalerts raised: {len(alerts):,}")
        print(alerts["alert_type"].value_counts().to_string())
        print("\nhighest-severity alerts:")
        cols = [c for c in ["acq_date", "latitude", "longitude", "frp",
                            "alert_type", "severity"] if c in alerts.columns]
        print(alerts.nlargest(10, "severity")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
