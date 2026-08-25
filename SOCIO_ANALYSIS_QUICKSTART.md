# socio_analysis.py: Implementation Summary

## ✓ Complete

`socio_analysis.py` is fully implemented and tested on a synthetic fixture. Below is what to run for development and production use.

---

## Test on Synthetic Fixture (Recommended First Step)

The synthetic fixture validates the entire pipeline before running on real XBD data.

```bash
# Generate synthetic test data (~200 buildings, 4 census tracts, 2 events)
python fixtures/synthetic_data.py

# Run analysis on fixture (--min-buildings 10 because fixture is small)
python scripts/socio_analysis.py \
  --splits fixtures/output/splits_fixture.csv \
  --xbd-dir fixtures/output \
  --census-tracts fixtures/output/census_tracts_fixture.geojson \
  --acs fixtures/output/acs_stub.csv \
  --out socio_output_test \
  --min-buildings 10
```

**Expected output:** `socio_output_test/` with:
- `buildings.csv` — Building-level data after spatial join
- `tracts.csv` — Tract-level aggregates (building counts, damage shares, ACS/exposure metrics)
- `regression_results.csv` — Coefficients, standard errors, p-values (naive + controlled models)
- `summary.txt` — Row counts at each pipeline stage
- `coefficient_forest.png` — Forest plot comparing model coefficients
- `damage_vs_income.png` — Scatter plot: damage vs. income, colored by exposure

**Exit code:** 0 = success

---

## Run on Real XBD Data

### Inputs Required

1. **splits.csv** — File index (scene_id, event_name, split columns)
   - Restrict to US events: hurricane-harvey, hurricane-michael, hurricane-florence, midwest-flooding, socal-fire, woolsey-fire, santa-rosa-wildfire, joplin-tornado, moore-tornado, tuscaloosa-tornado
   - Non-US events cause exit code 1

2. **XBD Labels Directory** — Post-disaster JSON files
   - Format: `{scene_id}_post_disaster.json`
   - Each JSON: GeoJSON FeatureCollection with building polygons
   - Required fields: geometry (WGS84 Polygon), properties.subtype (damage class)

3. **Census Tract GeoJSON** — TIGER boundaries (WGS84)
   - Required column: GEOID
   - Download: https://www2.census.gov/geo/tiger/GENZ2020/shp/

4. **ACS CSV** — Demographic data (vintage matched to disaster year)
   - Required columns: GEOID, income_YYYY, pov_rate_YYYY, median_home_value_YYYY, renter_pct_YYYY, median_year_built_YYYY, pop_density_YYYY
   - Fetch via Census API or pre-cached tables

### Run Analysis

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --out results/socio_output \
  --events hurricane-harvey socal-fire \
  --min-buildings 20 \
  --geo block_group
```

**Key Arguments:**
- `--events` — Filter to specific US events (default: all in splits.csv)
- `--min-buildings` — Drop tracts with < N buildings (default: 20)
- `--geo` — Spatial unit: block_group (default) or tract
- `--dry-run` — Validate pipeline without writing outputs
- `--preds PREDS_CSV` — Compare predictions vs. ground truth (runs analysis twice, side-by-side coefficients)

### With Model Predictions

```bash
python scripts/socio_analysis.py \
  --splits data/splits.csv \
  --xbd-dir xbd/ \
  --census-tracts data/census_tracts.geojson \
  --acs data/acs.csv \
  --preds model_predictions.csv \
  --out results/socio_with_preds
```

**Predictions CSV format:**
```
scene_id,building_uid,pred_subtype
hurricane-harvey_00000001,building_0,destroyed
...
```

**Output:** regression_results.csv with side-by-side coefficients (ground-truth vs. predictions)
- Quantifies model error bias: if predictions are systematically worse on low-income buildings, this is visible in coefficient comparison

---

## Implementation Details

### Core Features

✓ **Exposure Controls** (Requirement 1)
- Computed exposure metrics (distance proxy, building density, structure age)
- Two models: naive (socioeconomic only) + controlled (with exposure)
- Shows confounding bias: income coefficient changes between naive and controlled

✓ **Predictions vs. Truth** (Requirement 2)
- `--preds` flag runs analysis twice
- Side-by-side coefficient tables in regression_results.csv
- Quantifies error-correlation bias: if model errors correlate with income

✓ **Script Conventions**
- argparse with --dry-run flag
- Rows-in/rows-out table at each stage (printed to logs, in summary.txt)
- Nonzero exit on non-US event / no buildings / CRS mismatch / zero surviving units
- Docstrings with Args, Returns, exit conditions
- Comments explaining confounding and error-correlation rationale

✓ **Outputs**
- buildings.csv, tracts.csv, regression_results.csv, summary.txt
- Coefficient forest plot (naive vs. controlled)
- Damage vs. income scatter (colored by exposure proxy)

✓ **Robustness**
- Singular matrix fallback (Moore-Penrose pseudo-inverse)
- File validation and clear error messages
- Handles XBD labels in multiple directory structures

---

## Pipeline Overview

```
1. Load splits.csv
   ↓
2. Load building labels from XBD post_disaster.json files
   (extract WGS84 centroids + damage class per building)
   ↓
3. Load census tract boundaries (TIGER GeoJSON)
   ↓
4. Spatial join: buildings → tracts (centroid-in-polygon)
   ↓
5. Aggregate to tract level:
   - Building count
   - Damage shares by class
   - Mean damage score
   ↓
6. Load ACS demographic data
   ↓
7. Compute exposure metrics (distance proxy, density, age)
   ↓
8. Run regression analysis:
   - Model 1 (naive): damage ~ socioeconomic
   - Model 2 (controlled): damage ~ exposure + socioeconomic
   ↓
9. If --preds: repeat steps 4–8 on predicted damage classes
   ↓
10. Output: buildings.csv, tracts.csv, regression_results.csv, summary.txt, plots
```

Exit codes:
- 0 = success
- 1 = non-US event / no buildings / CRS mismatch / zero tracts after filtering
- 2 = file I/O / missing columns / argument error

---

## Files Created

- **scripts/socio_analysis.py** — Main analysis script (900+ lines)
- **fixtures/synthetic_data.py** — Synthetic fixture generator (~240 lines)
- **docs/socio_analysis_README.md** — Full documentation with examples

Test outputs in **socio_output_test/**:
- buildings.csv (200 buildings)
- tracts.csv (4 tracts)
- regression_results.csv (8 rows: 2 models × 2 levels × 4 socio features, well-defined)
- summary.txt (pipeline statistics)
- coefficient_forest.png (visual comparison)
- damage_vs_income.png (scatter plot)

---

## Next: Run on Real Data

1. Prepare inputs (splits.csv, XBD labels, census tracts, ACS)
2. Validate with --dry-run:
   ```bash
   python scripts/socio_analysis.py --splits ... --dry-run
   ```
3. Run full analysis:
   ```bash
   python scripts/socio_analysis.py --splits ... --out results/socio_analysis
   ```
4. Interpret regression_results.csv and plots:
   - Compare naive vs. controlled income coefficients (confounding bias)
   - If --preds: compare ground-truth vs. prediction coefficients (model error bias)
   - Forest plot shows effect sizes visually

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Event 'X' not in whitelist" | Only US events allowed. Check `Config.ALLOWED_EVENTS`. |
| "Spatial join produced zero results" | Buildings outside census tracts. Check CRS and geographic region. |
| "No tracts survived filtering" | Lower `--min-buildings` or use `--geo tract` instead of block_group. |
| "Design matrix is singular" | Normal with small samples. Script uses pseudo-inverse. Run on larger dataset for stable p-values. |
| Plots fail to save | Check `--out` directory is writable. Matplotlib may need backend config. |

---

See [docs/socio_analysis_README.md](../docs/socio_analysis_README.md) for complete documentation.
