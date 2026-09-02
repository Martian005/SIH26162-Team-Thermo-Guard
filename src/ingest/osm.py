"""Extract industrial / mining / energy footprints from a local OSM pbf extract.

Overpass was tried first and proved unusable: a country-sized query split into
3-degree tiles ran at ~3.5 minutes per tile with frequent timeouts, projecting to
roughly six hours.

The obvious local alternative -- pyosmium with `locations=True` -- is also wrong
here. That builds a coordinate index for every node in India (hundreds of
millions) just to resolve a few thousand industrial ways, and was still growing
past 2.9 GB after fourteen minutes.

Instead this makes two indexless passes:

  pass 1  find ways whose tags interest us and record their node refs; collect
          tagged standalone nodes directly (nodes carry their own coordinates)
  pass 2  resolve coordinates for only the node ids pass 1 asked for

Memory is then proportional to the geometry we actually want, not to India.

Polygons come from closed ways only. Multipolygon *relations* are skipped, which
loses a few complex sites; the power plant database backfills the major thermal
facilities independently.

Usage:
    python -m src.ingest.osm --pbf data/raw/india-latest.osm.pbf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import osmium
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import INTERIM, RAW  # noqa: E402

# (key, value) -> the class of thermal source this tag hints at
TAG_MAP = {
    ("landuse", "industrial"): "industrial",
    ("landuse", "quarry"): "mining",
    ("landuse", "landfill"): "landfill",
    ("man_made", "works"): "industrial",
    ("man_made", "flare"): "flare",
    ("man_made", "petroleum_well"): "oil_gas",
    ("man_made", "gasometer"): "oil_gas",
    ("power", "plant"): "power_plant",
    ("industrial", "oil"): "oil_gas",
    ("industrial", "refinery"): "refinery",
    ("industrial", "steel"): "steel",
    ("industrial", "chemical"): "chemical",
}

WATCHED_KEYS = {k for k, _ in TAG_MAP}


def classify(tags) -> str | None:
    """Return our category for an OSM object, or None if it is not of interest."""
    for key in WATCHED_KEYS:
        value = tags.get(key)
        if value is not None and (key, value) in TAG_MAP:
            return TAG_MAP[(key, value)]
    return None


def _attrs(obj, category: str, osm_type: str) -> dict:
    t = obj.tags
    return {
        "osm_id": obj.id,
        "osm_type": osm_type,
        "category": category,
        "name": t.get("name"),
        "operator": t.get("operator"),
        "plant_source": t.get("plant:source"),
        "product": t.get("product"),
    }


class CollectRefs(osmium.SimpleHandler):
    """Pass 1: matching ways (as node-id lists) and matching standalone nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.ways: list[dict] = []
        self.nodes: list[dict] = []
        self.needed: set[int] = set()
        self.skipped_open = 0

    def node(self, n) -> None:
        category = classify(n.tags)
        if category is None or not n.location.valid():
            return
        row = _attrs(n, category, "node")
        row["geometry"] = Point(n.location.lon, n.location.lat)
        self.nodes.append(row)

    def way(self, w) -> None:
        category = classify(w.tags)
        if category is None:
            return
        refs = [n.ref for n in w.nodes]
        if len(refs) < 4 or refs[0] != refs[-1]:
            self.skipped_open += 1
            return
        row = _attrs(w, category, "way")
        row["refs"] = refs
        self.ways.append(row)
        self.needed.update(refs)


class ResolveCoords(osmium.SimpleHandler):
    """Pass 2: coordinates for just the node ids pass 1 requested."""

    def __init__(self, needed: set[int]) -> None:
        super().__init__()
        self.needed = needed
        self.coords: dict[int, tuple[float, float]] = {}

    def node(self, n) -> None:
        if n.id in self.needed and n.location.valid():
            self.coords[n.id] = (n.location.lon, n.location.lat)


def build_polygons(ways: list[dict], coords: dict) -> list[dict]:
    """Turn resolved node lists into valid polygons."""
    out, unresolved = [], 0
    for row in ways:
        pts = [coords[r] for r in row["refs"] if r in coords]
        if len(pts) < 4:
            unresolved += 1
            continue
        geom = Polygon(pts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            unresolved += 1
            continue
        row = {k: v for k, v in row.items() if k != "refs"}
        row["geometry"] = geom
        out.append(row)
    if unresolved:
        print(f"  ways dropped (unresolved/degenerate): {unresolved:,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract OSM infrastructure from pbf")
    ap.add_argument("--pbf", default=str(RAW / "india-latest.osm.pbf"))
    ap.add_argument("--out", default=str(INTERIM / "osm_infrastructure.parquet"))
    args = ap.parse_args()

    pbf = Path(args.pbf)
    if not pbf.exists():
        raise SystemExit(f"pbf not found: {pbf}")
    print(f"parsing {pbf.name} ({pbf.stat().st_size / 1e9:.2f} GB)")

    t0 = time.time()
    p1 = CollectRefs()
    p1.apply_file(str(pbf))
    print(f"pass 1 ({time.time() - t0:.0f}s): {len(p1.ways):,} closed ways, "
          f"{len(p1.nodes):,} tagged nodes, {len(p1.needed):,} node refs needed")
    print(f"  skipped open ways: {p1.skipped_open:,}")

    t1 = time.time()
    p2 = ResolveCoords(p1.needed)
    p2.apply_file(str(pbf))
    print(f"pass 2 ({time.time() - t1:.0f}s): resolved {len(p2.coords):,} node coords")

    rows = build_polygons(p1.ways, p2.coords) + p1.nodes
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.drop_duplicates(subset=["osm_id", "osm_type"]).reset_index(drop=True)

    # Projected area (EPSG:7755 = India Albers) for later filtering/weighting.
    poly = gdf.geom_type.eq("Polygon")
    gdf["area_m2"] = 0.0
    if poly.any():
        gdf.loc[poly, "area_m2"] = gdf.loc[poly, "geometry"].to_crs("EPSG:7755").area

    out_path = Path(args.out)
    gdf.to_parquet(out_path, index=False)

    print(f"\nwrote {len(gdf):,} features -> {out_path} in {time.time() - t0:.0f}s")
    print("\nby category:")
    print(gdf["category"].value_counts().to_string())
    print("\nby geometry type:")
    print(gdf.geom_type.value_counts().to_string())
    named = gdf["name"].notna().sum()
    print(f"\nnamed features: {named:,} ({100 * named / len(gdf):.1f}%)")


if __name__ == "__main__":
    main()
