"""Sample ESA WorldCover land cover at every detection.

Land cover is what separates a wildfire from a crop-residue burn from a factory:
the same 3 MW thermal anomaly means very different things over tree cover,
cropland and built-up land.

The 10 m WorldCover tiles are Cloud-Optimised GeoTIFFs on public S3. Rather than
downloading ~40 GB of tiles, each 3-degree tile is read once at a decimated
~100 m resolution (well below the 375 m VIIRS footprint) straight over HTTP.

Neighbourhood composition is computed with a box filter over the whole tile,
which is one pass per class regardless of how many points fall in the tile --
slicing a window per point would be thousands of times slower.

Usage:
    python -m src.features.landcover
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import INTERIM  # noqa: E402

S3_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)

TILE_DEG = 3
DECIMATED = 3600  # 3 degrees / 3600 px ~= 92 m per pixel

# WorldCover class codes we care about
CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    80: "water",
}

# box-filter window sizes in pixels, at ~92 m per pixel
RADII = {"1km": 11, "5km": 55}


def tile_name(lat: float, lon: float) -> str:
    """WorldCover tile id for a coordinate, named by its south-west corner."""
    t_lat = int(np.floor(lat / TILE_DEG) * TILE_DEG)
    t_lon = int(np.floor(lon / TILE_DEG) * TILE_DEG)
    ns = "N" if t_lat >= 0 else "S"
    ew = "E" if t_lon >= 0 else "W"
    return f"{ns}{abs(t_lat):02d}{ew}{abs(t_lon):03d}"


def sample_tile(tile: str, pts: pd.DataFrame) -> pd.DataFrame | None:
    """Read one tile decimated and sample class + neighbourhood fractions."""
    url = S3_BASE.format(tile=tile)
    try:
        with rasterio.open(url) as src:
            arr = src.read(
                1, out_shape=(1, DECIMATED, DECIMATED), resampling=Resampling.mode
            )
            bounds = src.bounds
    except rasterio.errors.RasterioIOError:
        return None  # ocean tiles simply do not exist

    # Map lon/lat to decimated pixel indices.
    col = ((pts["longitude"].to_numpy() - bounds.left) / (bounds.right - bounds.left)
           * DECIMATED).astype(int)
    row = ((bounds.top - pts["latitude"].to_numpy()) / (bounds.top - bounds.bottom)
           * DECIMATED).astype(int)
    col = np.clip(col, 0, DECIMATED - 1)
    row = np.clip(row, 0, DECIMATED - 1)

    out = pd.DataFrame(index=pts.index)
    out["lc_class"] = arr[row, col]

    for code, label in CLASSES.items():
        mask = (arr == code).astype(np.float32)
        if not mask.any():
            for tag in RADII:
                out[f"lc_{label}_{tag}"] = 0.0
            continue
        for tag, size in RADII.items():
            frac = uniform_filter(mask, size=size, mode="nearest")
            out[f"lc_{label}_{tag}"] = frac[row, col]

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample WorldCover at detections")
    ap.add_argument("--detections", default=str(INTERIM / "firms_gridded.parquet"))
    ap.add_argument("--out", default=str(INTERIM / "landcover.parquet"))
    ap.add_argument(
        "--by-cell",
        action="store_true",
        default=True,
        help="sample once per grid cell instead of per detection (much faster)",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.detections, columns=["cell_id", "latitude", "longitude"])
    print(f"loaded {len(df):,} detections")

    # Detections repeat in the same cell thousands of times; sampling per unique
    # cell centroid cuts the work by an order of magnitude with no real loss.
    if args.by_cell:
        pts = (
            df.groupby("cell_id")[["latitude", "longitude"]].mean().reset_index()
        )
        print(f"sampling {len(pts):,} unique cells")
    else:
        pts = df.reset_index(drop=True)

    pts["tile"] = [
        tile_name(la, lo)
        for la, lo in zip(pts["latitude"].to_numpy(), pts["longitude"].to_numpy())
    ]
    tiles = sorted(pts["tile"].unique())
    print(f"{len(tiles)} WorldCover tiles to read")

    results, missing = [], []
    for tile in tqdm(tiles, desc="tiles"):
        sub = pts[pts["tile"] == tile]
        sampled = sample_tile(tile, sub)
        if sampled is None:
            missing.append(tile)
            continue
        results.append(pd.concat([sub[["cell_id"]], sampled], axis=1))

    if not results:
        raise SystemExit("No tiles could be read -- check network access to S3.")

    out = pd.concat(results, ignore_index=True)
    out_path = Path(args.out)
    out.to_parquet(out_path, index=False)

    print(f"\nwrote {len(out):,} rows -> {out_path}")
    if missing:
        print(f"tiles unavailable (ocean/no data): {len(missing)} -> {missing[:8]}")

    named = out["lc_class"].map(CLASSES).fillna("other")
    print("\nland cover at detection point:")
    print((100 * named.value_counts(normalize=True)).round(1).to_string())


if __name__ == "__main__":
    main()
