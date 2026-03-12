"""
01_download_data.py
-------------------
Downloads all required datasets for the Commuter Gaps project:
  - CTA GTFS feed
  - Pace GTFS feed
  - LEHD LODES OD data for Illinois (2021)
  - ACS block group data (B08141, B19013, B03002, B08301)
  - HIFLD hospital locations
  - Illinois census block group geometries (TIGER/Line)
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

for d in [RAW / "gtfs", RAW / "lodes", RAW / "acs", RAW / "hifld", PROC]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def download(url: str, dest: Path, desc: str = "") -> Path:
    """Stream-download url → dest with a progress bar."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest
    print(f"  Downloading {desc or dest.name} …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                       desc=dest.name, leave=False) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


# ── 1. GTFS Feeds ──────────────────────────────────────────────────────────────
def download_gtfs():
    print("\n[1/5] GTFS feeds")

    feeds = {
        "cta": "https://www.transitchicago.com/downloads/sch_data/google_transit.zip",
        "pace": "https://www.pacebus.com/sites/default/files/2024-11/google_transit.zip",
    }
    for agency, url in feeds.items():
        dest = RAW / "gtfs" / f"{agency}_gtfs.zip"
        try:
            download(url, dest, f"{agency.upper()} GTFS")
            # Unzip into sub-folder
            out_dir = RAW / "gtfs" / agency
            if not out_dir.exists():
                with zipfile.ZipFile(dest, "r") as z:
                    z.extractall(out_dir)
                print(f"  Extracted → {out_dir}")
        except Exception as e:
            print(f"  WARNING: could not download {agency} GTFS: {e}")


# ── 2. LEHD LODES OD Data (Illinois 2021) ─────────────────────────────────────
def download_lodes():
    print("\n[2/5] LEHD LODES OD data (Illinois 2021)")

    # LODES 8 — Illinois OD main file (all workers)
    base = "https://lehd.ces.census.gov/data/lodes/LODES8/il/od"
    files = [
        "il_od_main_JT00_2021.csv.gz",   # all jobs
        "il_od_aux_JT00_2021.csv.gz",    # workers living in IL, working outside
    ]
    for fname in files:
        dest = RAW / "lodes" / fname
        download(f"{base}/{fname}", dest, fname)

    # LODES workplace area characteristics (WAC) — sector info
    wac_url = "https://lehd.ces.census.gov/data/lodes/LODES8/il/wac/il_wac_S000_JT00_2021.csv.gz"
    download(wac_url, RAW / "lodes" / "il_wac_S000_JT00_2021.csv.gz", "IL WAC 2021")

    # LODES residence area characteristics (RAC)
    rac_url = "https://lehd.ces.census.gov/data/lodes/LODES8/il/rac/il_rac_S000_JT00_2021.csv.gz"
    download(rac_url, RAW / "lodes" / "il_rac_S000_JT00_2021.csv.gz", "IL RAC 2021")

    # Block crosswalk (block → block group → tract → county)
    xwalk_url = "https://lehd.ces.census.gov/data/lodes/LODES8/il/il_xwalk.csv.gz"
    download(xwalk_url, RAW / "lodes" / "il_xwalk.csv.gz", "IL crosswalk")


# ── 3. ACS Data (Census API) ──────────────────────────────────────────────────
def download_acs():
    print("\n[3/5] ACS block group data for Cook County (IL FIPS 031)")

    try:
        from census import Census
    except ImportError:
        print("  'census' package not installed — skipping ACS download")
        print("  Run: pip install census")
        return

    # Use Census API without key (limited to 500 requests/day)
    c = Census("")   # anonymous — works for small pulls

    state = "17"        # Illinois
    county = "031"      # Cook County

    tables = {
        "B08141": ["001E", "002E", "003E", "004E", "005E"],   # vehicle availability
        "B19013": ["001E"],                                      # median household income
        "B03002": ["001E", "003E", "004E", "012E"],             # race/ethnicity
        "B08301": ["001E", "010E"],                             # means of transportation
    }

    records = []
    for table, fields in tables.items():
        var_list = [f"{table}_{f}" for f in fields]
        print(f"  Fetching {table} …")
        try:
            data = c.acs5.state_county_blockgroup(
                var_list,
                state_fips=state,
                county_fips=county,
                tract="*",
                blockgroup="*",
                year=2021,
            )
            df = pd.DataFrame(data)
            df["GEOID"] = (
                df["state"] + df["county"] + df["tract"] + df["block group"]
            )
            records.append(df.set_index("GEOID").drop(
                columns=["state", "county", "tract", "block group"], errors="ignore"
            ))
        except Exception as e:
            print(f"  WARNING: could not fetch {table}: {e}")

    if records:
        acs = pd.concat(records, axis=1)
        out = RAW / "acs" / "cook_county_acs2021_bg.csv"
        acs.to_csv(out)
        print(f"  Saved → {out.name} ({len(acs):,} block groups)")


# ── 4. HIFLD Hospitals ────────────────────────────────────────────────────────
def download_hospitals():
    print("\n[4/5] HIFLD hospital locations")

    # HIFLD hospitals GeoJSON (national — filter to IL in analysis)
    url = (
        "https://opendata.arcgis.com/datasets/"
        "6ac5e325468c4cb9b905f1728d6fbf0f_0.geojson"
    )
    dest = RAW / "hifld" / "hospitals.geojson"
    try:
        download(url, dest, "HIFLD Hospitals")
        gdf = gpd.read_file(dest)
        il = gdf[gdf["STATE"] == "IL"].copy()
        il.to_file(RAW / "hifld" / "illinois_hospitals.geojson", driver="GeoJSON")
        print(f"  Illinois hospitals: {len(il):,} facilities")
    except Exception as e:
        print(f"  WARNING: {e}")


# ── 5. Census Block Group Geometries (Cook County) ────────────────────────────
def download_tiger():
    print("\n[5/5] TIGER/Line block group geometries (Illinois 2021)")

    # Illinois block groups — TIGER/Line
    url = (
        "https://www2.census.gov/geo/tiger/TIGER2021/BG/"
        "tl_2021_17_bg.zip"
    )
    dest = RAW / "acs" / "tl_2021_17_bg.zip"
    try:
        download(url, dest, "IL Block Groups TIGER")
        out_dir = RAW / "acs" / "tiger_bg"
        if not out_dir.exists():
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(out_dir)
        gdf = gpd.read_file(out_dir)
        # Filter to Cook County (COUNTYFP = 031)
        cook = gdf[gdf["COUNTYFP"] == "031"].copy()
        cook.to_file(PROC / "cook_block_groups.geojson", driver="GeoJSON")
        print(f"  Cook County block groups: {len(cook):,}")
    except Exception as e:
        print(f"  WARNING: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Commuter Gaps — Data Download")
    print("=" * 60)
    download_gtfs()
    download_lodes()
    download_acs()
    download_hospitals()
    download_tiger()
    print("\nDone. Check data/raw/ for downloaded files.")
