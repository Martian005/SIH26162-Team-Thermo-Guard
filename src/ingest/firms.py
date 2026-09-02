"""Bulk-download NASA FIRMS thermal anomaly detections for the study area.

The FIRMS area API accepts at most 5 days per request, so a multi-year pull is
built from many small chunks. Each chunk is cached to disk as raw CSV, which
makes the whole download resumable -- rerunning skips whatever already landed.

Archive (`*_SP`) sources carry the `type` column that flags static land sources
such as gas flares; the NRT sources do not. Train on SP, infer on NRT.

Usage:
    python -m src.ingest.firms --start 2024-01-01 --end 2026-04-27
    python -m src.ingest.firms --sensors VIIRS_SNPP_SP VIIRS_NOAA20_SP
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (  # noqa: E402
    BBOX_STR,
    FIRMS_BASE,
    FIRMS_MAP_KEY,
    INTERIM,
    MAX_DAYS_PER_REQUEST,
    RAW,
)

FIRMS_RAW = RAW / "firms"
CSV_HEADER_START = "latitude"
MAX_RETRIES = 4


def chunk_starts(start: date, end: date) -> list[date]:
    """Split [start, end] into MAX_DAYS_PER_REQUEST-day windows."""
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=MAX_DAYS_PER_REQUEST)
    return out


def fetch_chunk(sensor: str, start: date, days: int = MAX_DAYS_PER_REQUEST) -> Path | None:
    """Download one chunk to cache. Returns the cached path, or None on failure.

    Already-cached chunks are returned untouched so the pull is resumable.
    """
    dest = FIRMS_RAW / sensor / f"{start.isoformat()}.csv"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"{FIRMS_BASE}/area/csv/{FIRMS_MAP_KEY}/{sensor}/{BBOX_STR}/{days}/{start.isoformat()}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            text = resp.text.lstrip()

            # The API returns plain-text errors with HTTP 200, so sniff the header.
            if not text.startswith(CSV_HEADER_START):
                snippet = text[:200].replace("\n", " ")
                if "rate" in snippet.lower() or "limit" in snippet.lower():
                    time.sleep(20 * (attempt + 1))
                    continue
                print(f"  [warn] {sensor} {start}: unexpected response: {snippet}")
                return None

            dest.write_text(text, encoding="utf-8")
            return dest

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  [warn] {sensor} {start}: giving up after {MAX_RETRIES} tries ({exc})")
                return None
            time.sleep(3 * (attempt + 1))

    return None


def download_sensor(sensor: str, start: date, end: date, workers: int = 4) -> int:
    """Download every chunk for one sensor. Returns count of chunks on disk."""
    starts = chunk_starts(start, end)
    todo = [s for s in starts if not (FIRMS_RAW / sensor / f"{s.isoformat()}.csv").exists()]
    print(f"{sensor}: {len(starts)} chunks total, {len(todo)} to fetch")

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_chunk, sensor, s): s for s in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=sensor):
                fut.result()

    return len(list((FIRMS_RAW / sensor).glob("*.csv")))


def consolidate(sensors: list[str]) -> pd.DataFrame:
    """Concat all cached chunks into one typed DataFrame."""
    frames = []
    for sensor in sensors:
        files = sorted((FIRMS_RAW / sensor).glob("*.csv"))
        for f in tqdm(files, desc=f"reading {sensor}"):
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue
            df["sensor"] = sensor
            frames.append(df)

    if not frames:
        raise SystemExit("No FIRMS data cached -- run the download first.")

    out = pd.concat(frames, ignore_index=True)

    # acq_time is HHMM as an int (e.g. 812 -> 08:12), so zero-pad before parsing.
    out["acq_time"] = out["acq_time"].astype(int).astype(str).str.zfill(4)
    out["acq_dt"] = pd.to_datetime(
        out["acq_date"] + " " + out["acq_time"], format="%Y-%m-%d %H%M", utc=True
    )
    out = out.drop_duplicates(
        subset=["latitude", "longitude", "acq_dt", "satellite", "sensor"]
    ).reset_index(drop=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download FIRMS archive for the study area")
    ap.add_argument("--start", default="2024-01-01", type=date.fromisoformat)
    ap.add_argument("--end", default="2026-04-27", type=date.fromisoformat)
    ap.add_argument("--sensors", nargs="+", default=["VIIRS_SNPP_SP"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(INTERIM / "firms_detections.parquet"))
    args = ap.parse_args()

    if not FIRMS_MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY missing -- set it in .env")

    for sensor in args.sensors:
        download_sensor(sensor, args.start, args.end, args.workers)

    df = consolidate(args.sensors)
    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)

    print(f"\nwrote {len(df):,} detections -> {out_path}")
    print(f"date range : {df['acq_date'].min()} .. {df['acq_date'].max()}")
    print(f"size on disk: {out_path.stat().st_size / 1e6:.1f} MB")
    if "type" in df.columns:
        print("\nFIRMS type breakdown:")
        print(df["type"].value_counts().to_string())
    print("\nday/night split:")
    print(df["daynight"].value_counts().to_string())


if __name__ == "__main__":
    main()
