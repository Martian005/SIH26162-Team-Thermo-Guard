# Industrial Thermal Anomaly Monitor

**SIH 2026 &mdash; Problem SIH26162**
AI-Based Detection and Classification of Industrial Fires and Persistent Thermal
Sources Using NASA FIRMS, OSM & Satellite Data.

Classifies every satellite thermal detection over India into one of six source
types, maintains a national registry of persistent thermal sources, and raises
alerts when an industrial site burns outside its own established baseline.

---

## The core idea

NASA FIRMS reports *that* a pixel is hot. It cannot say *what* is burning &mdash;
that is exactly the gap this problem statement describes.

The instinct is to throw a CNN at satellite imagery. That is the wrong tool here:
there is no labelled corpus of "this thermal anomaly was a refinery flare", and a
375 m VIIRS pixel contains almost no visual texture to learn from.

The separating evidence is not in the pixel. It is in three things around it:

| Signal | Why it separates classes |
|---|---|
| **Persistence** | A gas flare burns in the same 500 m cell on hundreds of nights a year. A wildfire burns for days, moves, and never returns to the same pixel. Crop-residue burning collapses into a two-month seasonal window. |
| **Spatial context** | Distance to refineries, quarries, power stations and industrial footprints; land cover under the pixel. A 3 MW anomaly means something entirely different over tree cover than inside a refinery boundary. |
| **Radiometry** | Flares are small, very hot and radiometrically stable. Wildfires are large, cooler per pixel, and volatile. |

All three are tabular. So the model is gradient boosting (LightGBM) with SHAP
explanations, not a black-box vision network &mdash; and it trains in minutes on a
laptop instead of hours on a GPU.

### This is measured, not asserted

Median statistics for the ~500 m grid cells, over 2.66 M detections
(Jan 2024 &ndash; Apr 2026):

| Feature | Transient cells | Persistent cells |
|---|---|---|
| Active days | 1 | **49** |
| Distinct months | 1 | **9** |
| Night fraction | 0.00 | **0.98** |
| Month entropy | 0.00 | **0.80** |
| Spatial spread | 0 m | 152 m |
| Mean FRP | 4.03 MW | 1.53 MW |

Persistent sources are detected almost entirely at night &mdash; small flares only
clear the detection threshold without solar contamination &mdash; burn across three
quarters of the year, and never move.

### Ground validation

Persistent cells found within 10 km of known industrial sites, with no site list
given to the algorithm:

| Site | Persistent cells | Nearest | Max active days (of 848) |
|---|---|---|---|
| Jharia coalfield, Dhanbad | 118 | 0.1 km | **583** |
| Singrauli thermal | 48 | 4.7 km | 176 |
| Korba thermal | 21 | 3.1 km | 329 |
| Bhilai steel | 19 | 1.3 km | 285 |
| Jamnagar refinery | 7 | 0.5 km | 183 |
| Ghazipur landfill, Delhi | 0 | 20.9 km | &mdash; |

Jharia &mdash; the century-old underground coal seam fire &mdash; emerges as the
densest persistent cluster in the country. Ghazipur correctly returns nothing:
landfill fires are episodic, so they belong in the alert path, not the registry.

The most persistent cell nationwide is **21.105&deg;N, 72.645&deg;E &mdash; Hazira,
Surat**, hot on 657 of 848 days.

Independently, NASA's own `type=2` ("other static land source") flag agrees with
our persistence classification on essentially every top-ranked cell. Two methods
built from different evidence converging is the validation that counts.

---

## Architecture

```
NASA FIRMS archive ─┐
OSM (Geofabrik pbf) ─┤
ESA WorldCover COGs ─┼─> features ─> weak labels ─> LightGBM ─> FastAPI ─> MapLibre
WRI Power Plants    ─┘                                  │
                                                        └─> anomaly engine ─> alerts
```

### Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| `firms` | `src/ingest/firms.py` | Chunked multi-year FIRMS pull (API caps at 5 days/request), cached and resumable |
| `plants` | `src/ingest/powerplants.py` | WRI Global Power Plant Database, filtered to thermal plants |
| `osm` | `src/ingest/osm.py` | Industrial / quarry / refinery / landfill footprints from the India pbf |
| `persistence` | `src/features/persistence.py` | Per-cell recurrence stats + spatiotemporal DBSCAN fire events |
| `landcover` | `src/features/landcover.py` | ESA WorldCover sampled at each cell, with 1 km and 5 km neighbourhood composition |
| `context` | `src/features/context.py` | Nearest-infrastructure distances and point-in-polygon flags |
| `dataset` | `src/model/dataset.py` | Joins everything, applies the weak-label rule cascade |
| `train` | `src/model/train.py` | LightGBM with spatially blocked CV |
| `predict` | `src/model/predict.py` | Scores all detections, builds the registry and SHAP explanations |
| `anomaly` | `src/model/anomaly.py` | Escalation and new-source alerts against per-site baselines |

---

## Two methodological points worth defending

**Labels are programmatic, and we say so.** No ground-truth dataset exists, so
labels come from rules over independent evidence &mdash; persistence, OSM polygons,
land cover, NASA's static-source flag. Each carries a confidence weight used in
training. This is weak supervision, not hand-labelling, and the honest framing is
"programmatic labelling with human-in-the-loop verification".

**Evaluation is spatially blocked.** Detections cluster hard in space; Jharia
alone contributes tens of thousands. A random train/test split would put the same
coal fire on both sides and report a meaningless score. Folds are grouped by
~1&deg; spatial block, so every evaluation is on ground the model never saw.
Columns that fed the label rules &mdash; including FIRMS `type` &mdash; are barred
from the feature set, and `type` is absent from NRT data anyway.

Per-class precision and recall are reported rather than accuracy, because the
classes are severely imbalanced.

---

## Results, and how to read them honestly

Spatially blocked out-of-fold, 300k sampled detections, 7 classes:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| agricultural_burn | 0.999 | 1.000 | 1.000 | 59,112 |
| gas_flare | 0.917 | **0.727** | 0.811 | 1,545 |
| industrial_fire | 0.998 | 0.999 | 0.999 | 3,342 |
| industrial_plant | 0.981 | 0.991 | 0.986 | 12,279 |
| mining | 0.980 | 0.999 | 0.989 | 11,866 |
| thermal_power | 1.000 | 0.991 | 0.996 | 6,414 |
| wildfire | 1.000 | 1.000 | 1.000 | 205,442 |
| **accuracy** | | | **0.998** | 300,000 |

**That 0.998 is not a claim of real-world accuracy, and should never be presented
as one.** The labels are programmatic, so this measures how faithfully the model
reproduces the rule cascade — it is close to circular. A rule set and a model
that agree tell you the model trained correctly; they tell you nothing about
whether the rules were right.

The honest evaluation is the independent site check (`scripts/validate.py`),
which interrogates the output at facilities never supplied to the pipeline:

| site | predicted | expected | |
|---|---|---|---|
| Jharia coalfield | mining | mining | correct |
| Angul / NALCO | industrial_plant | industrial | correct |
| Jamnagar refinery | gas_flare | gas_flare | correct |
| Paradip refinery | gas_flare | gas_flare | correct |
| Korba | gas_flare | thermal_power | wrong |
| Singrauli | mining | thermal_power | wrong |
| Bhilai steel | thermal_power | industrial | wrong |
| Hazira, Surat | thermal_power | industrial | wrong |
| Koyali refinery | thermal_power | gas_flare | wrong |
| Mundra | gas_flare | thermal_power | wrong |

**4 of 10 exact.** Some misses are defensible — Korba is simultaneously a
coalfield and a power complex, Bhilai steel runs a captive power station, and
the check reports the modal class within 10 km, so genuinely mixed industrial
zones resolve to whichever source dominates. But they are counted as wrong here
rather than argued away.

Known causes, in order of impact:

1. **OSM tags only 2 refineries in all of India.** Indian refineries are mapped
   as generic `landuse=industrial` with a name, so the high-confidence gas_flare
   rule almost never fires and the class falls back to a 0.60-confidence proxy.
   Fixing this needs a curated facility list, not a better model.
2. **Co-located industry.** A single 500 m cell can sit inside a coalfield, a
   power station and an industrial estate at once. The taxonomy assumes one
   label per cell.
3. **Weak labels cap the ceiling.** No amount of model tuning beats the rules;
   improving accuracy means improving the labels, ideally with a few hundred
   hand-verified sites.

What the system demonstrably does well, independent of the classifier: it finds
persistent thermal sources. That result rests on measured recurrence statistics,
not on labels, and is corroborated by NASA's own static-source flag.

---

## The alert engine is the point

A classifier alone is academic. The operational question is *"is anything burning
today that should not be?"*

Every routine flare and boiler in the country already has a measured FRP
distribution in the registry. Against that, two things are anomalous:

- **Escalation** &mdash; heat at a known site far above *its own* normal range. A
  refinery flare at 40 MW when its two-year 95th percentile is 6 MW is no longer
  a flare.
- **New source** &mdash; heat inside an industrial footprint with no detection
  history at all.

Baselining each site against itself matters: one national FRP threshold would
drown in false positives from steel plants while missing every small facility.

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

cp .env.example .env        # then add your FIRMS_MAP_KEY
python run_pipeline.py
uvicorn src.api.main:app --port 8000
```

A free FIRMS key comes from <https://firms.modaps.eosdis.nasa.gov/api/map_key/>
and is the **only** credential the system needs. Every other source &mdash; OSM,
WorldCover, WRI &mdash; is open and unauthenticated.

The OSM extract (~1.7 GB) downloads from
<https://download.geofabrik.de/asia/india-latest.osm.pbf>.

### Deliverables mapping

- **(i) Classification and segregation of industrial fires from forest fires and
  other natural fires** &mdash; six-class LightGBM classifier with SHAP
  explanations, plus the persistent-source registry.
- **(ii) GIS-based solution for data storage and visualisation as a map overlay**
  &mdash; GeoParquet storage with GeoPandas/Shapely spatial joins, served as
  GeoJSON to a MapLibre overlay. PostGIS is the intended upgrade for the full
  build; see below.

### A note on boundaries

The study area is the **coordinate bounding box 68&ndash;98&deg;E, 6&ndash;38&deg;N**,
not a political boundary. Two consequences follow, and both are deliberate:

1. Detections appear in neighbouring countries, because a rectangle around India
   necessarily contains parts of them. Nothing is filtered by nationality.
2. Boundaries visible on the map are drawn by the third-party basemap provider
   (Esri or OpenStreetMap) and are **not authoritative**. This system asserts no
   territorial claim. The UI offers a "None" basemap that renders detections with
   no boundaries at all.

A production deployment for an Indian agency should render over an official
Survey of India or Bhuvan (NRSC) basemap and clip results to the official
national boundary. Bhuvan's public WMS was evaluated for this prototype and was
not reliable enough to depend on.

### Known prototype limits

- Storage is GeoParquet, not PostGIS. The spatial operations are identical, but
  the full build should move to PostGIS with `ST_AsMVT` vector tiles &mdash; at
  2.6 M points GeoJSON is already the bottleneck, which is why the detections
  endpoint caps and samples.
- OSM multipolygon *relations* are skipped; only closed ways become polygons.
  Some large refinery boundaries are mapped as relations. The power plant
  database backfills the major thermal sites independently.
- Sentinel-2 SWIR visual confirmation and Sentinel-5P gas-plume corroboration are
  designed but not built.
- One sensor (VIIRS S-NPP). Adding NOAA-20/21 roughly doubles overpass frequency
  and would sharpen the persistence statistics.
