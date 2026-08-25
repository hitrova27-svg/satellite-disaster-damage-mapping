"""
Create synthetic test fixture for socio_analysis.py development.

Generates:
- splits_fixture.csv with file index
- synthetic label JSONs (post-disaster, with damage polygons)
- fake census tract shapefile (GeoJSON)
- stub ACS table (CSV)

Covers 2 events: "mock-hurricane-1" and "mock-fire-1" across 2 tracts.
~200 buildings total.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, shape


def create_fixture(output_dir: Path):
    """
    Args:
        output_dir: directory to write fixture files
    
    Returns:
        None
    
    Exit conditions:
        - Nonzero if geopandas/shapely fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Event metadata
    events = {
        "hurricane-harvey": {"date": "2017-08-25", "lat": 29.5, "lon": -94.5},
        "socal-fire": {"date": "2017-10-10", "lat": 34.2, "lon": -118.5},
    }
    
    buildings_data = []
    building_id = 0
    
    # Tract 1: WGS84 bounds (29.48, -94.52, 29.52, -94.48)
    # Tract 2: WGS84 bounds (34.18, -118.52, 34.22, -118.48)
    tracts = {
        "tract-1": {"bounds": (29.48, -94.52, 29.52, -94.48), "income": 45000, "poverty": 0.18},
        "tract-2": {"bounds": (29.50, -94.50, 29.54, -94.46), "income": 75000, "poverty": 0.08},
    }
    
    fire_tracts = {
        "tract-3": {"bounds": (34.18, -118.52, 34.22, -118.48), "income": 55000, "poverty": 0.12},
        "tract-4": {"bounds": (34.20, -118.50, 34.24, -118.46), "income": 90000, "poverty": 0.05},
    }
    
    all_tracts = {**tracts, **fire_tracts}
    
    # Damage classes distribution
    damage_classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    # --- Create label JSONs for hurricane-harvey ---
    splits_rows = []
    
    for tract_id, tract_info in tracts.items():
        min_lat, min_lon, max_lat, max_lon = tract_info["bounds"]
        
        # Create 50 buildings per tract for hurricane (total ~200 for 4 tracts)
        for i in range(50):
            # Stagger damage: higher income tracts get less damage
            income = tract_info["income"]
            if income > 60000:
                damage_class = ["no-damage", "no-damage", "minor-damage", "major-damage"][i % 4]
            else:
                damage_class = ["minor-damage", "major-damage", "major-damage", "destroyed"][i % 4]
            
            # Random building location within tract
            import random
            lat = min_lat + random.random() * (max_lat - min_lat)
            lon = min_lon + random.random() * (max_lon - min_lon)
            
            # Simple square polygon around centroid
            delta = 0.0005
            poly_coords = [
                [lon - delta, lat - delta],
                [lon + delta, lat - delta],
                [lon + delta, lat + delta],
                [lon - delta, lat + delta],
                [lon - delta, lat - delta],
            ]
            
            label_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
                        "properties": {
                            "subtype": damage_class,
                            "uid": f"{tract_id}_building_{building_id}",
                        }
                    }
                ]
            }
            
            building_id += 1
            # Use tract_id directly (it's already a string like "tract-1")
            scene_id = f"hurricane-harvey_{tract_id}"
            
            # Write label JSON
            label_file = output_dir / f"{scene_id}_post_disaster.json"
            with open(label_file, "w") as f:
                json.dump(label_data, f)
            
            splits_rows.append({
                "scene_id": scene_id,
                "split": "train",
                "event_name": "hurricane-harvey",
                "disaster_type": "hurricane"
            })
    
    # --- Create label JSONs for socal-fire ---
    for tract_id, tract_info in fire_tracts.items():
        min_lat, min_lon, max_lat, max_lon = tract_info["bounds"]
        
        # Create 50 buildings per tract for fire (total ~200 for 4 tracts)
        for i in range(50):
            income = tract_info["income"]
            if income > 70000:
                damage_class = ["no-damage", "minor-damage", "major-damage"][i % 3]
            else:
                damage_class = ["major-damage", "major-damage", "destroyed"][i % 3]
            
            import random
            lat = min_lat + random.random() * (max_lat - min_lat)
            lon = min_lon + random.random() * (max_lon - min_lon)
            
            delta = 0.0005
            poly_coords = [
                [lon - delta, lat - delta],
                [lon + delta, lat - delta],
                [lon + delta, lat + delta],
                [lon - delta, lat + delta],
                [lon - delta, lat - delta],
            ]
            
            label_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
                        "properties": {
                            "subtype": damage_class,
                            "uid": f"{tract_id}_building_{building_id}",
                        }
                    }
                ]
            }
            
            building_id += 1
            # Use tract_id directly (it's already a string like "tract-3")
            scene_id = f"socal-fire_{tract_id}"
            
            label_file = output_dir / f"{scene_id}_post_disaster.json"
            with open(label_file, "w") as f:
                json.dump(label_data, f)
            
            splits_rows.append({
                "scene_id": scene_id,
                "split": "train",
                "event_name": "socal-fire",
                "disaster_type": "fire"
            })
    
    # --- Write splits fixture ---
    splits_file = output_dir / "splits_fixture.csv"
    with open(splits_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "split", "event_name", "disaster_type"])
        writer.writeheader()
        writer.writerows(splits_rows)
    
    # --- Create census tract shapefile (GeoJSON) ---
    tract_features = []
    for tract_id, tract_info in all_tracts.items():
        min_lat, min_lon, max_lat, max_lon = tract_info["bounds"]
        poly = Polygon([
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ])
        
        tract_features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "GEOID": tract_id,
                "NAMELSAD": f"Tract {tract_id}",
            }
        })
    
    tracts_geojson = {"type": "FeatureCollection", "features": tract_features}
    tracts_file = output_dir / "census_tracts_fixture.geojson"
    with open(tracts_file, "w") as f:
        json.dump(tracts_geojson, f)
    
    # --- Create ACS stub table ---
    acs_data = []
    for tract_id, tract_info in all_tracts.items():
        acs_data.append({
            "GEOID": tract_id,
            "income_2017": tract_info["income"],
            "pov_rate_2017": tract_info["poverty"],
            "median_home_value_2017": 250000 + int(tract_info["income"]),
            "renter_pct_2017": 0.35,
            "median_year_built_2017": 1985,
            "pop_density_2017": 1000 + int(tract_info["income"] / 100),
        })
    
    acs_file = output_dir / "acs_stub.csv"
    pd.DataFrame(acs_data).to_csv(acs_file, index=False)
    
    print(f"✓ Created synthetic fixture in {output_dir}")
    print(f"  - {len(splits_rows)} buildings across {len(all_tracts)} tracts")
    print(f"  - splits_fixture.csv, *.json labels, census_tracts_fixture.geojson, acs_stub.csv")


def mapping(geom):
    """Convert Shapely geometry to GeoJSON dict."""
    return json.loads(json.dumps(geom.__geo_interface__))


if __name__ == "__main__":
    fixture_dir = Path(__file__).parent / "output"
    create_fixture(fixture_dir)
