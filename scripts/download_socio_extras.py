"""Download the two open datasets that cover disaster types other than flood.

CAL FIRE DINS   per-structure post-fire inspections, with coordinates and a damage percentage band.
FEMA IHP        per-household registrations, with an inspector-verified property loss in dollars
                and whether the household carried insurance. Covers every disaster type.

Both are public and need no key. Run once; the CSVs land in data/ and the notebooks read them from
there (see notebooks/socio_analysis.ipynb and notebooks/damage_cost_analysis.ipynb).
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / 'data'
DINS_URL = ('https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/'
            'POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query')
FEMA_ENTITY = 'IndividualsAndHouseholdsProgramValidRegistrations'
FEMA_URL = f'https://www.fema.gov/api/open/v2/{FEMA_ENTITY}'

DINS_FIRES = {'santa-rosa-wildfire': 'TUBBS', 'socal-fire': 'THOMAS', 'woolsey-fire': 'WOOLSEY'}

EVENT_COUNTIES = {
    'hurricane-harvey': (4332, ['Fort Bend (County)', 'Harris (County)']),
    'hurricane-florence': (4393, ['Duplin (County)', 'Pender (County)']),
    'hurricane-michael': (4399, ['Bay (County)']),
    'midwest-flooding': (4438, ['Creek (County)', 'Osage (County)', 'Pawnee (County)',
                                'Tulsa (County)']),
    'santa-rosa-wildfire': (4344, ['Sonoma (County)']),
    'socal-fire': (4353, ['Los Angeles (County)', 'Ventura (County)']),
    'woolsey-fire': (4407, ['Los Angeles (County)', 'Ventura (County)']),
    'joplin-tornado': (1980, ['Jasper (County)', 'Newton (County)']),
    'tuscaloosa-tornado': (1971, ['Tuscaloosa (County)']),
    'moore-tornado': (4117, ['Cleveland (County)']),
}

FEMA_FIELDS = ('disasterNumber,county,damagedZipCode,rpfvl,ppfvl,ihpAmount,homeOwnersInsurance,'
               'ownRent,destroyed,grossIncome')


def fetch_dins():
    fields = 'DAMAGE,STRUCTURETYPE,CITY,COUNTY,ZIPCODE,INCIDENTNAME'
    frames = []
    for event, incident in DINS_FIRES.items():
        rows, offset = [], 0
        while True:
            response = requests.get(DINS_URL, timeout=180, params={
                'where': f"UPPER(INCIDENTNAME) LIKE '%{incident}%'", 'outFields': fields,
                'outSR': '4326', 'f': 'geojson', 'resultOffset': offset,
                'resultRecordCount': 2000})
            features = response.json().get('features', [])
            for feature in features:
                longitude, latitude = feature['geometry']['coordinates']
                rows.append({**feature['properties'], 'longitude': longitude, 'latitude': latitude})
            if len(features) < 2000:
                break
            offset += 2000
        frame = pd.DataFrame(rows)
        frame['event_name'] = event
        frames.append(frame)
        print(f'{event:<20} {incident:<8} {len(frame):>6,} inspected structures')
    return pd.concat(frames, ignore_index=True)


def fema_get(params, attempts=8):
    """The API returns 503 under load, so every request is retried with a growing pause."""
    for attempt in range(attempts):
        response = requests.get(FEMA_URL, timeout=300, params=params)
        if response.status_code == 200:
            return response.json()
        time.sleep(5 * (attempt + 1))
    response.raise_for_status()


def fema_count(query):
    return fema_get({'$filter': query, '$top': '1', '$select': 'id',
                     '$inlinecount': 'allpages'})['metadata']['count']


def fema_slices(query, limit=20000):
    """Deep $skip values get throttled, so a large query is cut into loss bands instead."""
    if fema_count(query) <= limit:
        return [query]
    edges = [0, 3000, 8000, 20000, 50000]
    slices = [f'{query} and rpfvl gt {low} and rpfvl le {high}'
              for low, high in zip(edges, edges[1:])]
    return slices + [f'{query} and rpfvl gt {edges[-1]}']


def fema_rows(query):
    rows, skip = [], 0
    while True:
        batch = fema_get({'$filter': query, '$select': FEMA_FIELDS,
                          '$top': '1000', '$skip': skip})[FEMA_ENTITY]
        rows += batch
        if len(batch) < 1000:
            return rows
        skip += 1000
        time.sleep(0.3)


def fetch_fema_registrations():
    """Only households whose property loss was verified by an inspector: rpfvl > 0.

    Each event is written to parts/ as it finishes, so an interrupted run resumes instead of
    starting over.
    """
    parts = DATA / 'parts'
    parts.mkdir(exist_ok=True)
    for event, (number, counties) in EVENT_COUNTIES.items():
        part = parts / f'{event}.csv'
        if part.exists():
            print(f'{event:<20} already downloaded')
            continue
        county_filter = ' or '.join(f"county eq '{c}'" for c in counties)
        base = f'disasterNumber eq {number} and ({county_filter}) and rpfvl gt 0'
        rows = []
        for query in fema_slices(base):
            rows += fema_rows(query)
        frame = pd.DataFrame(rows)
        frame['event_name'] = event
        frame.to_csv(part, index=False)
        print(f'{event:<20} DR-{number:<5} {len(frame):>7,} inspected households', flush=True)
    return pd.concat([pd.read_csv(p) for p in sorted(parts.glob('*.csv'))], ignore_index=True)


if __name__ == '__main__':
    DATA.mkdir(exist_ok=True)
    if not (DATA / 'calfire_dins.csv').exists():
        print('CAL FIRE DINS')
        fetch_dins().to_csv(DATA / 'calfire_dins.csv', index=False)
    print('\nFEMA household registrations')
    fetch_fema_registrations().to_csv(DATA / 'fema_registrations.csv', index=False)
    print('\nsaved to', DATA)
