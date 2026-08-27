# socio_analysis.py: Spatial Inequality in Disaster Damage

Analysis pipeline to test whether disaster damage is unequally distributed across socioeconomic communities, accounting for hazard exposure.

## Overview

The script performs a regression analysis of building-level damage from XBD labels, aggregated to census tract level and joined with ACS demographic data. It implements two key requirements:

1. **Exposure Controls**: Damage is primarily driven by hazard exposure (distance to fire/coast/tornado track, building density, age). We fit models with and without exposure to isolate the true socioeconomic effect and demonstrate confounding bias.

2. **Predictions vs. Truth**: With `--preds`, the analysis runs twice (once on ground truth, once on predictions) and outputs side-by-side coefficients. Model errors aren't random; systematic errors correlate with income and can manufacture spurious findings.

## Quick Start

### Real Data (XBD Dataset)

#### 1. Prepare Input Files

**splits.csv** — File index with columns:
```
scene_id,split,event_name,disaster_type
hurricane-harvey_00000001,train,hurricane-harvey,hurricane
socal-fire_00000042,test,socal-fire,fire
...
```

**XBD Labels** — Directory of post-disaster label JSONs:
```
xbd/
  labels/
    hurricane-harvey_00000001_post_disaster.json
    socal-fire_00000042_post_disaster.json
    ...
```

Each JSON is a GeoJSON FeatureCollection with building polygons. The script extracts:
- `geometry` (WGS84 Polygon) → centroid for spatial join
- `properties.subtype` (damage class: "no-damage", "minor-damage", "major-damage", "destroyed")
- `properties.uid` (building identifier, optional)

**Census Tracts** — TIGER census tract shapefile, WGS84:
```
census_tracts.geojson or census_tracts.shp
  Required columns: GEOID, geometry
```

Download from [TIGER/Line](https://www2.census.gov/geo/tiger/TIGER2020/BG/):
```bash
# E.g., 2020 block groups for Texas
wget https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_48_bg.zip
unzip tl_2020_48_bg.zip
```

**ACS Data** — American Community Survey aggregates, CSV:
```
GEOID,income_2017,pov_rate_2017,median_home_value_2017,renter_pct_2017,median_year_built_2017,pop_density_2017
06001420100,55000,0.15,350000,0.40,1972,2500
06001420200,75000,0.08,450000,0.32,1985,1800
...
```

Required columns (vintage matched to disaster year):
- `GEOID` — Census tract or block group identifier
- `income_2017` — Median household income ($)
- `pov_rate_2017` — Poverty rate (0–1)
- `median_home_value_2017` — Median home value ($)
- `renter_pct_2017` — Percent renting (0–1)
- `median_year_built_2017` — Median year built
- `pop_density_2017` — Population density (persons per km²)

Fetch via [Census API](https://api.census.gov/) or pre-cached tables.

#### 2. Run Analysis

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --out results/socio_analysis \
  --events hurricane-harvey socal-fire \
  --min-buildings 20 \
  --geo block_group
```

**Arguments:**
- `--splits` — Path to splits.csv
- `--xbd-dir` — Directory containing post_disaster.json labels (or subdirs with labels/)
- `--census-tracts` — Path to TIGER GeoJSON or shapefile (WGS84)
- `--acs` — Path to ACS CSV
- `--out` — Output directory (default: socio_output)
- `--events` — Filter to specific events (default: all in splits.csv)
  - Allowed events (US whitelist): 
    - `hurricane-harvey`, `hurricane-michael`, `hurricane-florence`
    - `midwest-flooding`
    - `socal-fire`, `woolsey-fire`, `santa-rosa-wildfire`
    - `joplin-tornado`, `moore-tornado`, `tuscaloosa-tornado`
  - Non-US events cause nonzero exit
- `--min-buildings` — Drop tracts with < N buildings (default: 20)
- `--geo` — Spatial unit: `block_group` or `tract` (default: block_group)
- `--dry-run` — Validate pipeline, don't write outputs
- `--preds` — Path to predictions CSV (scene_id, building_uid, pred_subtype) for side-by-side comparison

#### 3. With Model Predictions

To compare ground truth vs. model predictions:

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --preds model_predictions.csv \
  --out results/socio_analysis_with_preds \
  --min-buildings 20
```

**Predictions CSV format:**
```
scene_id,building_uid,pred_subtype
hurricane-harvey_00000001,building_0,no-damage
hurricane-harvey_00000001,building_1,minor-damage
socal-fire_00000042,building_5,destroyed
...
```

The script runs the full analysis twice — once on true labels, once on predictions — and outputs side-by-side regression tables.

---

## Output Files

Located in `--out` directory:

### buildings.csv
Building-level data after spatial join:
```
uid,damage_class,scene_id,event_name,GEOID,geometry
tract-1_building_0,destroyed,hurricane-harvey_tract-1,hurricane-harvey,tract-1,"POINT (-94.5 29.5)"
...
```

### tracts.csv
Tract-level aggregates:
```
GEOID,building_count,damage_score_mean,damage_score_std,share_no-damage,share_minor-damage,...,income_2017,pov_rate_2017,...,exposure_proxy,building_density,median_building_age
tract-1,50,1.23,0.95,0.20,0.30,0.35,0.15,45000,0.18,...,0.45,1200,35
...
```

### regression_results.csv
Regression coefficients with standard errors and p-values:
```
model,label,feature,coef,se,t,pval
naive,ground-truth,income,−0.0521,0.0234,−2.23,0.034
naive,ground-truth,pov_rate,0.1234,0.0456,2.71,0.012
controlled,ground-truth,exposure_proxy,0.0892,0.0178,5.01,0.0001
controlled,ground-truth,income,0.0187,0.0298,0.63,0.537
...
naive,predictions,income,−0.0412,0.0241,−1.71,0.103
controlled,predictions,income,0.0234,0.0312,0.75,0.459
```

**Key interpretation:**
- Naive model (socioeconomic only): captures confounding by exposure
- Controlled model (+ exposure): isolates true socioeconomic effect
- Difference in income coefficient = confounding bias

### summary.txt
Row counts at each pipeline stage:
```
=== socio_analysis.py Summary ===
Event(s): hurricane-harvey, socal-fire
Min buildings per tract: 20

Rows-in / Rows-out at each stage:
  Buildings loaded: 15420
  After spatial join to tracts: 14890 (530 dropped outside tracts)
  After tract-level aggregation: 2134
  After ACS filtering: 2089 (45 dropped, no ACS data)

Regression (ground truth, n=2089):
  Naive model R²: 0.1823
  Controlled model R²: 0.4521

Regression (predictions, n=2089):
  Naive model R²: 0.1654
  Controlled model R²: 0.3892
```

### coefficient_forest.png
Forest plot comparing naive vs. controlled model coefficients with 95% CIs:
- X-axis: standardized coefficient
- Bars: naive model (left) vs. controlled model (right)
- Isolates confounding bias visually

### damage_vs_income.png
Scatter plot: damage (Y) vs. income (X), colored by exposure proxy.
- Shows raw correlation (confounded) and exposure heterogeneity
- Interpretation depends on hazard type

---

## Exposure Metrics

Provide real hazard exposure data for the analysis. The script derives building density and building age from the attached ACS data, while hazard-specific exposure must come from the corresponding hazard layers:

### Wildfire
- **Distance to fire perimeter** — Download NIFC [Wildland Fire Perimeter History](https://data-nifc.opendata.arcgis.com/)
- Compute Euclidean distance from tract centroid to perimeter
- Filter to events within 1 year (fire footprints age rapidly)

### Hurricane
- **Distance to coast** + **elevation** — Combine:
  - Coastline shapefile (NOAA, GEBCO, OSM)
  - DEM (USGS 3DEP, GEBCO)
  - Compute distance, add elevation as separate control
- Coastal tracts near storm surge risk = higher exposure

### Tornado
- **Distance to tornado track** — Download NWS [Storm Data](https://www.spc.noaa.gov/climo/torn/tornadoarchive.html) or [tornado-history](https://www.ncei.noaa.gov/products/tornado-climatology) 
- Convert text tracks to polylines (complex; see `tornado_tracks.py` utility if available)

### Building Environment
- **Building density** — Persons per km² (from ACS) or building count from OSM
- **Structure age** — Median year built (ACS) → 2020 − year
- Older buildings more damage-prone; more dense areas = better construction codes (effect may reverse)

If perimeter/track unavailable for an event, print a labeled note in outputs:
```
WARNING: No fire perimeter for socal-fire; using distance-to-tract-center proxy
```

---

## Exit Codes

- **0** — Success
- **1** — Non-US event, no buildings parsed, CRS mismatch, or zero surviving units after filtering
- **2** — File I/O error, missing columns, or argument error

---

## Examples

### Filter to One Event

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --out results/hurricane_only \
  --events hurricane-harvey
```

### Validate Before Commit

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --out /tmp/test_run \
  --dry-run && echo "✓ OK" || echo "✗ FAILED"
```

### Compare Model Predictions to Ground Truth

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --preds model_output/predictions.csv \
  --out results/pred_vs_truth \
  --min-buildings 20
# Check regression_results.csv for side-by-side coefficient comparison
```

---

## Design Notes

### Confounding & Rationale

Regressing damage on income alone is confounded because:
1. **Hurricane** — Coastal exposure (wealthy) ∩ damage (storm surge)
2. **Wildfire** — WUI exposure (mixed-income suburban) ∩ damage
3. **Tornado** — Random exposure (but property values correlate with damage-prone construction)

The controlled model isolates the true socioeconomic effect by including exposure and built-environment controls.

### Model Errors & Prediction Bias

If a model is weaker on dense, low-rise housing (more common in low-income tracts), the model's errors correlate with income. This manufactures a spurious socioeconomic result when analysis is run on predictions. The `--preds` flag quantifies this risk by running both models side-by-side.

### Pseudo-Inverse Fallback

With small sample sizes (< ~50 tracts), design matrices may be rank-deficient. The script uses `np.linalg.pinv()` (Moore-Penrose pseudo-inverse) to handle singularity gracefully, with a logged warning.

---

## Troubleshooting

**"Event 'X' not in whitelist"**
- Only US events allowed (non-US disaster damage analysis requires different controls)
- Allowed: see `--events` argument section above

**"Spatial join produced zero results"**
- XBD label centroids outside census tract boundaries
- Likely: XBD data in a different CRS or geographic region
- Debug: Check GEOID columns in outputs

**"No tracts survived filtering"**
- All tracts have < `--min-buildings` buildings
- Solution: Lower `--min-buildings` or aggregate to broader geographic unit (try `--geo tract` instead of block_group)

**"Design matrix is singular"**
- Small sample size (< 10 tracts) or multicollinear features
- Can occur with small or highly correlated real datasets; the script uses a pseudo-inverse fallback
- For production: ensure >= 50 tracts or drop highly correlated predictors

**Plots fail to save**
- Check `--out` directory is writable
- matplotlib may need backend configuration in headless environments

---

## Dependencies

```
geopandas>=0.11
shapely>=2.0
pandas>=1.5
numpy>=1.23
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.5
seaborn>=0.12
```

Install:
```bash
pip install -r requirements.txt
# or
pip install geopandas pandas numpy scipy scikit-learn matplotlib seaborn
```

---

## Next Steps

1. **Prepare real XBD data**:
   - Download splits.csv with US events only
   - Ensure post_disaster.json labels are accessible
   - Obtain TIGER census boundaries for disaster region(s)
   - Fetch ACS data (Census API or pre-cached tables)
2. **Run analysis** with appropriate `--min-buildings` (20–50 is typical)
3. **Interpret results**:
   - Compare naive vs. controlled coefficients → confounding bias
   - If `--preds`: compare ground-truth vs. prediction coefficients → model error bias
   - Forest plot: visual comparison of effect sizes
5. **Document findings** in report: confounding rationale and error-correlation risk

---

## Reference

See [project_specification.md](../docs/project_specification.md) for full requirements and grading rubric.
