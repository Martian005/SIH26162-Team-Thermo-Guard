"""Central config: paths, constants, FIRMS parameters."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- credentials ---
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")

# --- paths ---
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"

for _d in (RAW, INTERIM, PROCESSED, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# --- study area ---
# west, south, east, north
INDIA_BBOX = (68.0, 6.0, 98.0, 38.0)
BBOX_STR = ",".join(str(v) for v in INDIA_BBOX)

# --- FIRMS ---
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api"
MAX_DAYS_PER_REQUEST = 5  # hard API limit

# Archive (science-processed) sources carry the `type` column; NRT sources do not.
SENSORS_ARCHIVE = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]
SENSORS_NRT = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]

# FIRMS `type` field (archive only) -- partial free label
FIRMS_TYPE = {
    0: "presumed_vegetation_fire",
    1: "active_volcano",
    2: "other_static_land_source",  # gas flares, industrial plants
    3: "offshore",
}

# --- target classes ---
CLASSES = [
    "gas_flare",
    "industrial_fire",
    "thermal_power",
    "mining",
    "agricultural_burn",
    "wildfire",
]

# --- persistence gridding ---
GRID_DEG = 0.005  # ~500 m at Indian latitudes
