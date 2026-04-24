"""
03_lodes_analysis.py
--------------------
Phase 2: Overnight Employment Flow Mapping

Uses LEHD LODES 8 data to:
  1. Filter OD flows to shift-work NAICS sectors
  2. Geolocate home and work census blocks
  3. Classify each trip as served / unserved based on 2 AM transit coverage
  4. Aggregate to block group → Stranded Worker Density surface

Outputs:
  - data/processed/od_flows_shift_work.geojson  (flow lines)
  - data/processed/stranded_worker_density.geojson  (block group aggregation)
  - outputs/maps/stranded_workers.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
MAPS.mkdir(parents=True, exist_ok=True)

# NAICS sectors associated with overnight / shift work.
# Codes follow LODES 8 WAC super-sector naming:
#   CNS08 = Transportation & Warehousing (NAICS 48-49)
#   CNS16 = Health Care & Social Assistance (NAICS 62)
#   CNS18 = Accommodation & Food Services (NAICS 72)
SHIFT_WORK_SECTORS = {
    "CNS16": "Health Care & Social Assistance",
    "CNS18": "Accommodation & Food Services",
    "CNS08": "Transportation & Warehousing",
}

# Cook County FIPS prefix
COOK_FIPS = "17031"


def load_crosswalk() -> pd.DataFrame:
    """Load block → block group crosswalk with block centroids."""
    xwalk_path = RAW / "lodes" / "il_xwalk.csv.gz"
    if not xwalk_path.exists():
        raise FileNotFoundError("Crosswalk not found — run 01_download_data.py")
    xwalk = pd.read_csv(xwalk_path, dtype={"tabblk2020": str, "bgrp": str, "cty": str},
                        usecols=["tabblk2020", "bgrp", "cty", "blklatdd", "blklondd"])
    xwalk = xwalk.rename(columns={
        "tabblk2020": "block", "bgrp": "block_group", "cty": "county",
        "blklatdd": "lat", "blklondd": "lon",
    })
    return xwalk


def load_wac() -> pd.DataFrame:
    """Load Workplace Area Characteristics to get sector employment counts per block.
    Returns w_geocode → (shift_jobs, total_jobs, shift_share)."""
    wac_path = RAW / "lodes" / "il_wac_S000_JT00_2021.csv.gz"
    if not wac_path.exists():
        raise FileNotFoundError("WAC not found — run 01_download_data.py")

    wac = pd.read_csv(wac_path, dtype={"w_geocode": str})
    wac["w_geocode"] = wac["w_geocode"].str.zfill(15)

    sector_cols = [c for c in SHIFT_WORK_SECTORS if c in wac.columns]
    missing = set(SHIFT_WORK_SECTORS) - set(sector_cols)
    if missing:
        print(f"  WARNING: missing WAC sector columns: {sorted(missing)}")

    wac["shift_jobs"] = wac[sector_cols].sum(axis=1) if sector_cols else 0
    wac["total_jobs"] = wac["C000"] if "C000" in wac.columns else wac["shift_jobs"]
    wac["shift_share"] = np.where(
        wac["total_jobs"] > 0, wac["shift_jobs"] / wac["total_jobs"], 0.0
    )
    return wac[["w_geocode", "shift_jobs", "total_jobs", "shift_share"]]


def load_od() -> pd.DataFrame:
    """Load OD main file and filter to Cook County workers."""
    od_path = RAW / "lodes" / "il_od_main_JT00_2021.csv.gz"
    if not od_path.exists():
        raise FileNotFoundError("OD data not found — run 01_download_data.py")

    print("  Loading OD file (this may take a moment) …")
    od = pd.read_csv(od_path, dtype={"w_geocode": str, "h_geocode": str})
    od["w_geocode"] = od["w_geocode"].str.zfill(15)
    od["h_geocode"] = od["h_geocode"].str.zfill(15)

    # Filter: home OR work in Cook County
    cook_mask = (
        od["w_geocode"].str.startswith(COOK_FIPS) |
        od["h_geocode"].str.startswith(COOK_FIPS)
    )
    od = od[cook_mask].copy()
    print(f"  Cook County OD pairs: {len(od):,}")
    return od


def build_block_centroids(xwalk: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Build block centroids from LODES crosswalk lat/lon columns.
    Filters to Cook County (FIPS 17031) and all blocks used in Cook OD flows.
    """
    # Cook County blocks (for home/work block group lookup)
    cook_xwalk = xwalk[xwalk["county"] == "17031"].copy()
    print(f"  Cook County blocks in crosswalk: {len(cook_xwalk):,}")

    if len(cook_xwalk) == 0:
        # County filter mismatch — inspect actual values
        sample_cty = xwalk["county"].value_counts().head(5)
        print(f"  Sample county values: {sample_cty.to_dict()}")
        raise ValueError("No Cook County blocks found — check county code in crosswalk")

    cook_xwalk = cook_xwalk.dropna(subset=["lat", "lon"])
    cook_xwalk["lat"] = pd.to_numeric(cook_xwalk["lat"], errors="coerce")
    cook_xwalk["lon"] = pd.to_numeric(cook_xwalk["lon"], errors="coerce")
    cook_xwalk = cook_xwalk.dropna(subset=["lat", "lon"])

    gdf = gpd.GeoDataFrame(
        cook_xwalk[["block", "block_group", "lat", "lon"]],
        geometry=gpd.points_from_xy(cook_xwalk["lon"], cook_xwalk["lat"]),
        crs="EPSG:4326",
    )
    gdf = gdf.drop_duplicates(subset="block")
    return gdf.set_index("block")[["block_group", "geometry"]].copy()


def classify_served(od: pd.DataFrame,
                    block_centroids: gpd.GeoDataFrame,
                    transit_bg: gpd.GeoDataFrame,
                    wac: pd.DataFrame) -> pd.DataFrame:
    """Tag each OD pair as served / unserved based on 2 AM transit coverage.
    Only classifies pairs where BOTH home and work blocks are in Cook County.
    Adds a `shift_workers` weight = S000 × shift-work share at the destination block.
    """

    # Build lookup: block → overnight_stops (via block group)
    bg_service = transit_bg.set_index("GEOID")[["overnight_stops", "service_class"]].copy()

    # Map home block → block group via crosswalk
    h_bg = block_centroids["block_group"].rename("h_bg")
    w_bg = block_centroids["block_group"].rename("w_bg")

    od = od.join(h_bg, on="h_geocode", how="left")
    od = od.join(w_bg, on="w_geocode", how="left")

    # Only keep Cook-Cook pairs (both endpoints in Cook County)
    cook_cook = od["h_bg"].notna() & od["w_bg"].notna()
    od = od[cook_cook].copy()
    print(f"  Cook County -> Cook County pairs: {len(od):,}")

    od["h_overnight"] = od["h_bg"].map(bg_service["overnight_stops"]).fillna(0)
    od["w_overnight"] = od["w_bg"].map(bg_service["overnight_stops"]).fillna(0)

    od["served"]   = (od["h_overnight"] > 0) & (od["w_overnight"] > 0)
    od["unserved"] = ~od["served"]

    # Estimated shift-work weight per OD pair:
    # scale total jobs (S000) by the shift-work employment share at the work block
    shift_share = wac.set_index("w_geocode")["shift_share"]
    od["shift_share_dest"] = od["w_geocode"].map(shift_share).fillna(0.0)
    od["shift_workers"] = od["S000"] * od["shift_share_dest"]

    return od


def build_flow_lines(od: pd.DataFrame,
                     block_centroids: gpd.GeoDataFrame,
                     wac: pd.DataFrame,
                     max_flows: int = 5000) -> gpd.GeoDataFrame:
    """Build flow line GeoDataFrame for top unserved OD pairs (shift-work weighted)."""

    # Filter to pairs with any shift-work jobs at destination. Weight by
    # shift-worker volume (S000 * shift_share_dest) already computed in classify_served.
    cand = od[od["unserved"] & (od["shift_share_dest"] > 0) & (od["shift_workers"] > 0)]
    unserved = cand.nlargest(max_flows, "shift_workers")

    records = []
    for _, row in unserved.iterrows():
        h_gc = row["h_geocode"]
        w_gc = row["w_geocode"]
        if h_gc in block_centroids.index and w_gc in block_centroids.index:
            h_pt = block_centroids.loc[h_gc, "geometry"]
            w_pt = block_centroids.loc[w_gc, "geometry"]
            records.append({
                "h_geocode": h_gc,
                "w_geocode": w_gc,
                "total_jobs":     int(row["S000"]),
                "shift_workers":  round(float(row["shift_workers"]), 2),
                "shift_share_dest": round(float(row["shift_share_dest"]), 3),
                "h_bg": row.get("h_bg"),
                "w_bg": row.get("w_bg"),
                "geometry": LineString([h_pt, w_pt]),
            })

    if not records:
        print("  WARNING: no flow lines could be built (block centroids not matched)")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def build_stranded_density(od: pd.DataFrame,
                            bg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Aggregate unserved shift workers to block group level.

    `shift_workers` is S000 × shift-work sector share at the destination block,
    so the density surface reflects stranded SHIFT workers (healthcare,
    accommodation, transportation) rather than all commuters.
    """

    unserved = od[od["unserved"]].copy()

    home_agg = (
        unserved.groupby("h_bg")
        .agg(unserved_home=("shift_workers", "sum"), flow_count=("shift_workers", "count"))
        .reset_index()
        .rename(columns={"h_bg": "GEOID"})
    )

    work_agg = (
        unserved.groupby("w_bg")
        .agg(unserved_work=("shift_workers", "sum"))
        .reset_index()
        .rename(columns={"w_bg": "GEOID"})
    )

    result = bg.copy()
    result = result.merge(home_agg, on="GEOID", how="left")
    result = result.merge(work_agg, on="GEOID", how="left")
    result["unserved_home"] = result["unserved_home"].fillna(0).round(1)
    result["unserved_work"] = result["unserved_work"].fillna(0).round(1)
    result["stranded_density"] = (result["unserved_home"] + result["unserved_work"]).round(1)

    # Normalize to per sq km
    result_proj = result.to_crs("EPSG:26916")
    result["area_sqkm"] = result_proj.geometry.area / 1e6
    result["stranded_per_sqkm"] = (
        result["stranded_density"] / result["area_sqkm"].clip(lower=0.01)
    ).round(1)

    return result


def run_lodes_analysis():
    print("=" * 60)
    print("Phase 2: LODES Employment Flow Mapping")
    print("=" * 60)

    transit_path = PROC / "transit_access_delta.geojson"
    if not transit_path.exists():
        print("Transit access delta not found — run 02_gtfs_analysis.py first")
        return

    transit_bg = gpd.read_file(transit_path)
    bg = gpd.read_file(PROC / "cook_block_groups.geojson")

    print("Loading LODES data …")
    try:
        xwalk = load_crosswalk()
        wac   = load_wac()
        od    = load_od()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print("Building block centroids …")
    try:
        block_centroids = build_block_centroids(xwalk)
    except Exception as e:
        print(f"  ERROR building centroids: {e}")
        return

    print("Classifying served/unserved trips …")
    od = classify_served(od, block_centroids, transit_bg, wac)

    total = len(od)
    served   = od["served"].sum()
    unserved = od["unserved"].sum()
    shift_total    = od["shift_workers"].sum()
    shift_unserved = od.loc[od["unserved"], "shift_workers"].sum()
    print(f"  Total Cook County OD pairs:       {total:,}")
    print(f"  Served  (any-sector):             {served:,} ({served/total*100:.1f}%)")
    print(f"  Unserved (any-sector):            {unserved:,} ({unserved/total*100:.1f}%)")
    print(f"  Shift-work weighted trips total:  {shift_total:,.0f}")
    print(f"  Shift-work unserved:              {shift_unserved:,.0f} "
          f"({(shift_unserved/shift_total*100 if shift_total else 0):.1f}% of shift trips)")

    print("Building OD flow lines …")
    flows = build_flow_lines(od, block_centroids, wac, max_flows=3000)
    if not flows.empty:
        out_flows = PROC / "od_flows_shift_work.geojson"
        flows.to_file(out_flows, driver="GeoJSON")
        print(f"  Saved → {out_flows.name} ({len(flows):,} flow lines)")

    print("Building stranded worker density surface …")
    swd = build_stranded_density(od, bg)
    out_swd = PROC / "stranded_worker_density.geojson"
    swd.to_file(out_swd, driver="GeoJSON")
    print(f"  Saved → {out_swd.name}")

    print(f"\n  Mean stranded density: {swd['stranded_density'].mean():.0f} workers/block group")
    print(f"  Max stranded density:  {swd['stranded_density'].max():.0f}")

    make_stranded_map(swd, flows if not flows.empty else None, transit_bg)


def make_stranded_map(swd: gpd.GeoDataFrame,
                      flows,
                      transit_bg: gpd.GeoDataFrame):
    print("\nGenerating stranded worker density map …")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.set_axis_off()

    swd_4326 = swd.to_crs("EPSG:4326")
    transit_4326 = transit_bg.to_crs("EPSG:4326")

    # Panel 1 — Stranded worker density
    swd_4326.plot(ax=axes[0], column="stranded_per_sqkm",
                  cmap="magma", linewidth=0.1, edgecolor="#333333",
                  legend=True, missing_kwds={"color": "#1e2530"},
                  legend_kwds={"label": "Stranded Workers / km²",
                               "orientation": "horizontal", "pad": 0.03, "shrink": 0.7})

    if flows is not None and not flows.empty:
        flows_4326 = flows.to_crs("EPSG:4326") if flows.crs else flows
        flows_4326.plot(ax=axes[0], color="#00d4ff", linewidth=0.3, alpha=0.15)

    axes[0].set_title("Stranded Worker Density\n(Unserved Shift-Work OD Flows at 2 AM)",
                       color="white", fontsize=12, pad=10)

    # Panel 2 — Overnight stops with stranded overlay
    transit_4326.plot(ax=axes[1], column="overnight_stops", cmap="Blues",
                      linewidth=0.1, edgecolor="#333333",
                      legend=True,
                      legend_kwds={"label": "Overnight Stops (400m walk)",
                                   "orientation": "horizontal", "pad": 0.03, "shrink": 0.7})

    # Overlay highest-density stranded areas
    top_stranded = swd_4326[swd_4326["stranded_per_sqkm"] >
                             swd_4326["stranded_per_sqkm"].quantile(0.85)]
    if not top_stranded.empty:
        top_stranded.plot(ax=axes[1], color="#ff4444", alpha=0.5, linewidth=0,
                          aspect=None)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff4444",
               markersize=10, label="High Stranded Density (top 15%)"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower left",
                   facecolor="#1e2530", edgecolor="#555555", labelcolor="white")
    axes[1].set_title("2 AM Transit Coverage vs. High-Density Stranded Areas",
                       color="white", fontsize=12, pad=10)

    fig.suptitle("Chicago Overnight Transit Gap Analysis\n"
                 "LEHD LODES 2021 Shift-Work Employment Flows",
                 color="white", fontsize=14, y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = MAPS / "stranded_workers.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out.name}")


if __name__ == "__main__":
    run_lodes_analysis()
