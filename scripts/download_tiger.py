from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_STATES = ["01", "06", "12", "19", "29", "37", "40", "48"]
BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2020/BG"


def download_state(state_fips: str, work_dir: Path) -> Path:
    archive_path = work_dir / f"tl_2020_{state_fips}_bg.zip"
    shapefile_path = work_dir / f"tl_2020_{state_fips}_bg.shp"
    url = f"{BASE_URL}/tl_2020_{state_fips}_bg.zip"

    if not shapefile_path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(work_dir)

    return shapefile_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and merge TIGER/Line 2020 block-group boundaries."
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help="Two-digit state FIPS codes (default: disaster-event states).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "census_tracts.geojson",
        help="Output GeoJSON path.",
    )
    args = parser.parse_args()

    invalid_states = [state for state in args.states if len(state) != 2 or not state.isdigit()]
    if invalid_states:
        raise ValueError(f"State FIPS codes must be two digits: {invalid_states}")

    work_dir = DATA_DIR / "tiger_2020_block_groups"
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for state in args.states:
        shapefile = download_state(state, work_dir)
        frame = gpd.read_file(shapefile)
        if "GEOID" not in frame.columns:
            raise ValueError(f"{shapefile} does not contain a GEOID column")
        frames.append(frame[["GEOID", "geometry"]])

    boundaries = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    ).to_crs("EPSG:4326")
    boundaries["GEOID"] = boundaries["GEOID"].astype(str)
    boundaries.to_file(args.output, driver="GeoJSON")
    print(f"Saved {args.output} ({len(boundaries):,} block groups)")

    shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
