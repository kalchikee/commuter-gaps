"""
02_gtfs_analysis.py
-------------------
Phase 1: Transit Network Time-Dependency Analysis

Builds two transit network snapshots:
  - Peak service: Tuesday 8:00 AM
  - Overnight service: Tuesday 2:00 AM

For each Cook County census block group centroid, computes:
  - Number of stops reachable within 400m walk (served stops)
  - Route count at each time window
  - Transit Access Delta = % reduction in served stops 8 AM → 2 AM

Outputs:
  - data/processed/transit_access_delta.geojson
  - outputs/maps/service_comparison.png
"""

import warnings
warnings.filterwarnings("ignore")

import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import partridge as ptg
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
MAPS.mkdir(parents=True, exist_ok=True)

# Time windows (seconds from midnight)
PEAK_TIME      = 8 * 3600          # 08:00
OVERNIGHT_TIME = 2 * 3600          # 02:00  (next day trips use 26*3600 convention)
OVERNIGHT_ALT  = 26 * 3600         # 02:00 as 26:00 in GTFS extended format
WINDOW_SEC     = 3600              # ±30 min around target time
WALK_BUFFER_M  = 400               # meters — ~5 minute walk


def load_gtfs(agency: str) -> dict:
    """Load GTFS feed as dict of DataFrames using partridge."""
    feed_dir = RAW / "gtfs" / agency
    if not feed_dir.exists():
        raise FileNotFoundError(f"GTFS not found for {agency} at {feed_dir}")

    # partridge needs a zip or directory
    feed = ptg.load_feed(str(feed_dir))
    return {
        "stops":      feed.stops,
        "trips":      feed.trips,
        "routes":     feed.routes,
        "stop_times": feed.stop_times,
        "calendar":   feed.calendar,
    }


def get_active_stops(stop_times: pd.DataFrame,
                     trips: pd.DataFrame,
                     calendar: pd.DataFrame,
                     target_sec: int,
                     window: int = WINDOW_SEC,
                     day: str = "tuesday") -> pd.Series:
    """Return set of stop IDs with service in [target_sec - window, target_sec + window]
    on a typical Tuesday."""

    lo = target_sec - window
    hi = target_sec + window

    # Also handle overnight trips expressed as 24h+ times
    lo_alt = lo + 24 * 3600
    hi_alt = hi + 24 * 3600

    # Active service IDs on Tuesday
    active_svc = calendar[calendar[day] == 1]["service_id"].unique()
    active_trips = trips[trips["service_id"].isin(active_svc)]["trip_id"].unique()

    st = stop_times[stop_times["trip_id"].isin(active_trips)].copy()

    # departure_time may be string "HH:MM:SS" — convert to seconds
    def to_sec(t):
        if pd.isna(t):
            return np.nan
        parts = str(t).split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    if st["departure_time"].dtype == object:
        st["dep_sec"] = st["departure_time"].map(to_sec)
    else:
        st["dep_sec"] = st["departure_time"]

    mask = (
        ((st["dep_sec"] >= lo) & (st["dep_sec"] <= hi)) |
        ((st["dep_sec"] >= lo_alt) & (st["dep_sec"] <= hi_alt))
    )
    return st[mask]["stop_id"].unique()


def build_stop_gdf(stops: pd.DataFrame, crs_out: str = "EPSG:26916") -> gpd.GeoDataFrame:
    """Convert stops DataFrame to GeoDataFrame in UTM 16N."""
    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(crs_out)
    return gdf


def stops_within_buffer(bg_centroids: gpd.GeoDataFrame,
                        active_stop_gdf: gpd.GeoDataFrame,
                        buffer_m: float = WALK_BUFFER_M) -> pd.Series:
    """For each block group centroid, count active stops within buffer_m."""
    centroids_proj = bg_centroids.to_crs("EPSG:26916")
    stops_proj = active_stop_gdf

    # Spatial join: centroid buffer → stops
    buf = centroids_proj.copy()
    buf["geometry"] = buf.geometry.buffer(buffer_m)

    joined = gpd.sjoin(stops_proj[["stop_id", "geometry"]],
                       buf[["GEOID", "geometry"]],
                       how="left", predicate="within")
    counts = joined.groupby("GEOID")["stop_id"].count()
    return counts


def run_gtfs_analysis():
    print("=" * 60)
    print("Phase 1: GTFS Time-Dependency Analysis")
    print("=" * 60)

    # Load Cook County block group centroids
    bg_path = PROC / "cook_block_groups.geojson"
    if not bg_path.exists():
        print("Block groups not found — run 01_download_data.py first")
        return

    bg = gpd.read_file(bg_path).to_crs("EPSG:4326")
    bg_centroids = bg.copy()
    bg_centroids["geometry"] = bg.geometry.centroid

    print(f"Block groups: {len(bg):,}")

    # ── Load GTFS feeds ────────────────────────────────────────────────────────
    all_stops     = []
    all_stop_times = []
    all_trips     = []
    all_calendar  = []

    for agency in ["cta", "pace"]:
        gtfs_dir = RAW / "gtfs" / agency
        if not gtfs_dir.exists():
            print(f"  WARNING: {agency} GTFS not found — skipping")
            continue
        print(f"  Loading {agency.upper()} GTFS …")
        try:
            feed = ptg.load_feed(str(gtfs_dir))
            stops = feed.stops.copy()
            stops["agency"] = agency
            all_stops.append(stops)
            all_stop_times.append(feed.stop_times)
            trips = feed.trips.copy()
            trips["agency"] = agency
            all_trips.append(trips)
            all_calendar.append(feed.calendar)
        except Exception as e:
            print(f"  ERROR loading {agency}: {e}")

    if not all_stops:
        print("No GTFS data loaded. Exiting.")
        return

    stops_df   = pd.concat(all_stops, ignore_index=True)
    st_df      = pd.concat(all_stop_times, ignore_index=True)
    trips_df   = pd.concat(all_trips, ignore_index=True)
    calendar_df = pd.concat(all_calendar, ignore_index=True).drop_duplicates()

    print(f"  Total stops: {len(stops_df):,}")
    print(f"  Total stop-times: {len(st_df):,}")

    # ── Active stops at each time window ──────────────────────────────────────
    print("\nComputing active stops at 8 AM (peak) …")
    peak_ids = get_active_stops(st_df, trips_df, calendar_df,
                                target_sec=PEAK_TIME, day="tuesday")

    print("Computing active stops at 2 AM (overnight) …")
    overnight_ids = get_active_stops(st_df, trips_df, calendar_df,
                                     target_sec=OVERNIGHT_TIME, day="tuesday")

    print(f"  Peak active stops:      {len(peak_ids):,}")
    print(f"  Overnight active stops: {len(overnight_ids):,}")

    # ── Build stop GeoDataFrames ───────────────────────────────────────────────
    peak_stops_gdf      = build_stop_gdf(stops_df[stops_df["stop_id"].isin(peak_ids)])
    overnight_stops_gdf = build_stop_gdf(stops_df[stops_df["stop_id"].isin(overnight_ids)])

    # ── Count stops per block group ────────────────────────────────────────────
    print("\nCounting stops per block group …")
    peak_counts      = stops_within_buffer(bg_centroids, peak_stops_gdf)
    overnight_counts = stops_within_buffer(bg_centroids, overnight_stops_gdf)

    bg["peak_stops"]      = bg["GEOID"].map(peak_counts).fillna(0).astype(int)
    bg["overnight_stops"] = bg["GEOID"].map(overnight_counts).fillna(0).astype(int)

    # Transit Access Delta: % reduction
    bg["access_delta"] = np.where(
        bg["peak_stops"] > 0,
        ((bg["peak_stops"] - bg["overnight_stops"]) / bg["peak_stops"] * 100).round(1),
        0.0,
    )

    # Classify overnight service level
    def classify_service(row):
        if row["overnight_stops"] == 0:
            return "No Service"
        elif row["overnight_stops"] < 3:
            return "Very Limited"
        elif row["overnight_stops"] < 8:
            return "Limited"
        else:
            return "Moderate"

    bg["service_class"] = bg.apply(classify_service, axis=1)

    # Save
    out_path = PROC / "transit_access_delta.geojson"
    bg.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved → {out_path.name}")

    # Summary stats
    print("\nService Classification Summary:")
    print(bg["service_class"].value_counts().to_string())
    print(f"\nMean Transit Access Delta: {bg['access_delta'].mean():.1f}%")
    print(f"Block groups with NO overnight service: "
          f"{(bg['overnight_stops'] == 0).sum():,} / {len(bg):,}")

    # ── Map ────────────────────────────────────────────────────────────────────
    make_service_map(bg, peak_stops_gdf, overnight_stops_gdf)


def make_service_map(bg: gpd.GeoDataFrame,
                     peak_stops: gpd.GeoDataFrame,
                     overnight_stops: gpd.GeoDataFrame):
    print("\nGenerating service comparison map …")

    bg_4326 = bg.to_crs("EPSG:4326")
    peak_4326 = peak_stops.to_crs("EPSG:4326")
    night_4326 = overnight_stops.to_crs("EPSG:4326")

    fig, axes = plt.subplots(1, 3, figsize=(20, 9))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    cmap = plt.cm.YlOrRd

    # Panel 1 — Peak stops
    bg_4326.plot(ax=axes[0], column="peak_stops", cmap=cmap,
                 legend=True, linewidth=0.1, edgecolor="#333333",
                 legend_kwds={"label": "Stops within 400m", "orientation": "horizontal",
                               "pad": 0.05, "shrink": 0.8})
    axes[0].set_title("Peak Service (8 AM Tuesday)", color="white", fontsize=13, pad=10)
    axes[0].set_axis_off()

    # Panel 2 — Overnight stops
    bg_4326.plot(ax=axes[1], column="overnight_stops", cmap=cmap,
                 legend=True, linewidth=0.1, edgecolor="#333333",
                 legend_kwds={"label": "Stops within 400m", "orientation": "horizontal",
                               "pad": 0.05, "shrink": 0.8})
    axes[1].set_title("Overnight Service (2 AM Tuesday)", color="white", fontsize=13, pad=10)
    axes[1].set_axis_off()

    # Panel 3 — Access Delta
    color_map = {
        "No Service":    "#d73027",
        "Very Limited":  "#fc8d59",
        "Limited":       "#fee08b",
        "Moderate":      "#1a9850",
    }
    for cat, color in color_map.items():
        subset = bg_4326[bg_4326["service_class"] == cat]
        if not subset.empty:
            subset.plot(ax=axes[2], color=color, linewidth=0.1, edgecolor="#333333")

    legend_elements = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
    axes[2].legend(handles=legend_elements, loc="lower left",
                   facecolor="#1e2530", edgecolor="#555555", labelcolor="white",
                   fontsize=9)
    axes[2].set_title("Overnight Service Classification", color="white", fontsize=13, pad=10)
    axes[2].set_axis_off()

    fig.suptitle("Chicago Transit Accessibility: Peak vs. Overnight Service\n"
                 "Cook County Block Groups | CTA + Pace GTFS",
                 color="white", fontsize=15, y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = MAPS / "service_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out.name}")


if __name__ == "__main__":
    run_gtfs_analysis()
