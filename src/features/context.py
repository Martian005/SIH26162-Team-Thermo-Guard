"""Spatial context features: what infrastructure sits under or near a detection.

Thermal radiance alone cannot separate a steel plant from a burning field. The
separating evidence is geographic: a detection 200 m inside a refinery polygon
is a very different object from an identical detection 40 km from any industry.

For every ~500 m grid cell this produces:
  * great-circle distance to the nearest feature of each infrastructure class
  * boolean flags for falling *inside* an industrial / quarry / landfill polygon
  * capacity and fuel of the nearest thermal power plant

Distances use a BallTree on haversine, which is exact on the sphere and fast
enough to query 1.6 M cells against every category.

Usage:
    python -m src.features.context
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import INTERIM  # noqa: E402

EARTH_RADIUS_KM = 6371.0

# OSM categories worth their own distance feature
OSM_CATEGORIES = [
    "industrial",
    "refinery",
    "oil_gas",
    "flare",
    "power_plant",
    "mining",
    "landfill",
    "steel",
    "chemical",
]

# categories whose polygons support a meaningful "inside" test
POLYGON_FLAGS = ["industrial", "mining", "landfill", "refinery", "power_plant"]

FAR_KM = 999.0  # sentinel when a category has no features at all


def _tree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    return BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")


def nearest_km(tree: BallTree, lat: np.ndarray, lon: np.ndarray) -> tuple:
    """Distance in km and index of the nearest feature for each query point."""
    dist, idx = tree.query(np.radians(np.column_stack([lat, lon])), k=1)
    return dist[:, 0] * EARTH_RADIUS_KM, idx[:, 0]


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse polygons to interior points so they can go into a BallTree."""
    gdf = gdf.copy()
    is_poly = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    # representative_point stays inside concave shapes, unlike centroid
    gdf.loc[is_poly, "geometry"] = gdf.loc[is_poly, "geometry"].representative_point()
    return gdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Build spatial context features")
    ap.add_argument("--cells", default=str(INTERIM / "firms_gridded.parquet"))
    ap.add_argument("--osm", default=str(INTERIM / "osm_infrastructure.parquet"))
    ap.add_argument("--plants", default=str(INTERIM / "power_plants.parquet"))
    ap.add_argument("--out", default=str(INTERIM / "context.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.cells, columns=["cell_id", "latitude", "longitude"])
    cells = df.groupby("cell_id")[["latitude", "longitude"]].mean().reset_index()
    print(f"{len(cells):,} unique cells")

    lat = cells["latitude"].to_numpy()
    lon = cells["longitude"].to_numpy()
    out = cells[["cell_id"]].copy()

    # --- OSM infrastructure distances ---
    osm = gpd.read_parquet(args.osm)
    print(f"{len(osm):,} OSM features")
    osm_pts = representative_points(osm)

    for cat in OSM_CATEGORIES:
        sub = osm_pts[osm_pts["category"] == cat]
        col = f"dist_{cat}_km"
        if sub.empty:
            out[col] = FAR_KM
            continue
        tree = _tree(sub.geometry.y.to_numpy(), sub.geometry.x.to_numpy())
        out[col], _ = nearest_km(tree, lat, lon)
        print(f"  dist_{cat}: {len(sub):,} features, "
              f"median {np.median(out[col]):.1f} km")

    # Any heavy-industry feature at all, regardless of specific type.
    heavy = osm_pts[osm_pts["category"].isin(
        ["industrial", "refinery", "oil_gas", "flare", "steel", "chemical"]
    )]
    if not heavy.empty:
        tree = _tree(heavy.geometry.y.to_numpy(), heavy.geometry.x.to_numpy())
        out["dist_any_industry_km"], _ = nearest_km(tree, lat, lon)

    # --- inside-polygon flags ---
    pts_gdf = gpd.GeoDataFrame(
        cells[["cell_id"]],
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    for cat in POLYGON_FLAGS:
        polys = osm[
            osm["category"].eq(cat) & osm.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        col = f"inside_{cat}"
        if polys.empty:
            out[col] = False
            continue
        hit = gpd.sjoin(pts_gdf, polys[["geometry"]], how="inner", predicate="within")
        out[col] = out["cell_id"].isin(set(hit["cell_id"]))
        print(f"  inside_{cat}: {int(out[col].sum()):,} cells")

    # --- WRI power plants ---
    plants = gpd.read_parquet(args.plants)
    thermal = plants[plants["is_thermal"]]
    if not thermal.empty:
        tree = _tree(thermal.geometry.y.to_numpy(), thermal.geometry.x.to_numpy())
        dist, idx = nearest_km(tree, lat, lon)
        out["dist_thermal_plant_km"] = dist
        out["nearest_plant_mw"] = thermal["capacity_mw"].to_numpy()[idx]
        out["nearest_plant_fuel"] = thermal["primary_fuel"].to_numpy()[idx]
        print(f"  dist_thermal_plant: {len(thermal):,} plants, "
              f"median {np.median(dist):.1f} km")

    out_path = Path(args.out)
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {len(out):,} rows x {out.shape[1]} cols -> {out_path}")

    dist_cols = [c for c in out.columns if c.startswith("dist_")]
    print("\nfraction of cells within 2 km of each category:")
    for c in dist_cols:
        print(f"  {c:28s} {100 * (out[c] < 2).mean():5.2f}%")


if __name__ == "__main__":
    main()
