"""
04_equity_analysis.py
---------------------
Phase 3: Equity and Demographic Analysis

  1. Joins ACS data (vehicle ownership, income, race/ethnicity) to block groups
  2. Builds a Transit Dependency Index (TDI)
  3. Runs LISA bivariate clustering (Moran's I) to identify hot spots of
     high transit dependency co-located with poor overnight service
  4. Quantifies environmental justice disparities

Outputs:
  - data/processed/equity_analysis.geojson
  - outputs/maps/equity_hotspots.png
  - outputs/figures/ej_summary.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import esda
import libpysal
from libpysal.weights import Queen

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
FIGS = ROOT / "outputs" / "figures"
MAPS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


# ── ACS variable names ─────────────────────────────────────────────────────────
ACS_VARS = {
    "B08141_001E": "vehicles_total",       # total workers 16+
    "B08141_002E": "vehicles_none",        # no vehicle available
    "B19013_001E": "median_income",
    "B03002_001E": "total_pop",
    "B03002_003E": "white_nonhisp",
    "B03002_004E": "black_nonhisp",
    "B03002_012E": "hispanic",
    "B08301_001E": "commuters_total",
    "B08301_010E": "commuters_transit",
}


def load_acs(bg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load ACS data and join to block groups."""
    acs_path = RAW / "acs" / "cook_county_acs2021_bg.csv"

    if not acs_path.exists():
        print("  ACS CSV not found — using synthetic estimates for demonstration")
        return _synthetic_acs(bg)

    acs = pd.read_csv(acs_path, dtype={"GEOID": str})
    acs = acs.rename(columns={k: v for k, v in ACS_VARS.items() if k in acs.columns})

    # Coerce to numeric, treat negatives (census suppression codes) as NaN
    for col in ACS_VARS.values():
        if col in acs.columns:
            acs[col] = pd.to_numeric(acs[col], errors="coerce")
            acs.loc[acs[col] < 0, col] = np.nan

    merged = bg.merge(acs, on="GEOID", how="left")
    return merged


def _synthetic_acs(bg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Generate synthetic but spatially plausible ACS values using
    publicly known Cook County patterns as approximate ranges.
    Used only when ACS download is unavailable.
    """
    rng = np.random.default_rng(42)
    n = len(bg)

    bg = bg.copy()
    bg["vehicles_total"]   = rng.integers(100, 800, n)
    bg["vehicles_none"]    = (bg["vehicles_total"] * rng.uniform(0.05, 0.45, n)).astype(int)
    bg["median_income"]    = rng.integers(20000, 120000, n)
    bg["total_pop"]        = rng.integers(200, 3000, n)
    bg["white_nonhisp"]    = (bg["total_pop"] * rng.uniform(0.05, 0.85, n)).astype(int)
    bg["black_nonhisp"]    = (bg["total_pop"] * rng.uniform(0.02, 0.60, n)).astype(int)
    bg["hispanic"]         = (bg["total_pop"] * rng.uniform(0.02, 0.50, n)).astype(int)
    bg["commuters_total"]  = (bg["vehicles_total"] * 0.9).astype(int)
    bg["commuters_transit"]= (bg["commuters_total"] * rng.uniform(0.05, 0.55, n)).astype(int)
    return bg


def compute_tdi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Transit Dependency Index (TDI) = composite score of:
      - pct_no_vehicle  (high = more dependent)
      - pct_low_income  (income below $35k)
      - pct_transit_commute
    Each component normalized 0–1 then averaged.
    """
    gdf = gdf.copy()

    gdf["pct_no_vehicle"] = np.where(
        gdf["vehicles_total"] > 0,
        gdf["vehicles_none"] / gdf["vehicles_total"],
        np.nan,
    )
    gdf["pct_transit_commute"] = np.where(
        gdf["commuters_total"] > 0,
        gdf["commuters_transit"] / gdf["commuters_total"],
        np.nan,
    )
    # Low income flag: proxy via inverse normalized income
    inc = gdf["median_income"].clip(lower=1)
    gdf["income_burden"] = 1 - (inc - inc.min()) / (inc.max() - inc.min() + 1)

    # Minority population share
    gdf["pct_minority"] = np.where(
        gdf["total_pop"] > 0,
        1 - gdf["white_nonhisp"] / gdf["total_pop"],
        np.nan,
    )

    components = ["pct_no_vehicle", "pct_transit_commute", "income_burden"]
    for col in components:
        mn, mx = gdf[col].min(), gdf[col].max()
        gdf[f"{col}_norm"] = (gdf[col] - mn) / (mx - mn + 1e-9)

    norm_cols = [f"{c}_norm" for c in components]
    gdf["TDI"] = gdf[norm_cols].mean(axis=1)

    return gdf


def run_lisa(gdf: gpd.GeoDataFrame,
             var_x: str = "TDI",
             var_y: str = "access_delta") -> gpd.GeoDataFrame:
    """
    Bivariate LISA (Local Moran's I) between TDI and Transit Access Delta.
    Identifies spatial clusters:
      HH — high TDI + high service loss (priority areas)
      LL — low TDI + low service loss
      HL — high TDI but low service loss
      LH — low TDI but high service loss
    """
    gdf = gdf.copy()
    gdf_proj = gdf.to_crs("EPSG:26916")

    # Drop rows with missing values for LISA
    valid = gdf_proj[[var_x, var_y, "geometry"]].dropna()

    try:
        w = Queen.from_dataframe(valid, silence_warnings=True)
        w.transform = "r"

        x = np.asarray(valid[var_x].values, dtype=float)
        y = np.asarray(valid[var_y].values, dtype=float)

        # Standardize
        x_std = (x - x.mean()) / (x.std() + 1e-9)
        y_std = (y - y.mean()) / (y.std() + 1e-9)

        # Global bivariate Moran's I for reporting
        global_moran = esda.Moran_BV(x_std, y_std, w, permutations=499)

        # Local bivariate LISA for mapping
        local_lisa = esda.Moran_Local_BV(x_std, y_std, w, permutations=499)

        p_sim  = np.asarray(local_lisa.p_sim)
        lag_y  = np.asarray(libpysal.weights.lag_spatial(w, y_std))

        # Local LISA quadrants
        labels = np.full(len(valid), "NS", dtype="U10")
        for i in range(len(valid)):
            if p_sim[i] < 0.05:
                lx = x_std[i]
                ly = lag_y[i]
                if lx > 0 and ly > 0:
                    labels[i] = "HH"
                elif lx < 0 and ly < 0:
                    labels[i] = "LL"
                elif lx > 0 and ly < 0:
                    labels[i] = "HL"
                else:
                    labels[i] = "LH"

        valid["lisa_cluster"] = labels
        valid["lisa_pval"]    = p_sim
        valid["lisa_Ii"]      = np.asarray(local_lisa.Is)

        gdf = gdf.merge(
            valid[["lisa_cluster", "lisa_pval", "lisa_Ii"]],
            left_index=True, right_index=True, how="left",
        )
        gdf["lisa_cluster"] = gdf["lisa_cluster"].fillna("NS")

        p_global = float(np.asarray(global_moran.p_sim).flat[0])
        print(f"\n  Global Bivariate Moran's I: {global_moran.I:.4f}  (p = {p_global:.4f})")
        print("  LISA Cluster Distribution:")
        print(gdf["lisa_cluster"].value_counts().to_string())

    except Exception as e:
        print(f"  WARNING: LISA failed ({e}). Skipping spatial statistics.")
        gdf["lisa_cluster"] = "NS"
        gdf["lisa_pval"] = np.nan
        gdf["lisa_Ii"]   = np.nan

    return gdf


def run_equity_analysis():
    print("=" * 60)
    print("Phase 3: Equity and Demographic Analysis")
    print("=" * 60)

    # Load inputs
    transit_path = PROC / "transit_access_delta.geojson"
    swd_path     = PROC / "stranded_worker_density.geojson"

    if not transit_path.exists():
        print("transit_access_delta.geojson not found — run Phase 1 first")
        return

    transit_bg = gpd.read_file(transit_path)
    print(f"Block groups loaded: {len(transit_bg):,}")

    # Load stranded worker density if available
    if swd_path.exists():
        swd = gpd.read_file(swd_path)[["GEOID", "stranded_density", "stranded_per_sqkm"]]
        transit_bg = transit_bg.merge(swd, on="GEOID", how="left")
        transit_bg["stranded_density"] = transit_bg["stranded_density"].fillna(0)

    # Load and join ACS
    print("Loading ACS demographic data …")
    gdf = load_acs(transit_bg)

    # Compute TDI
    print("Computing Transit Dependency Index …")
    gdf = compute_tdi(gdf)
    print(f"  TDI range: {gdf['TDI'].min():.3f} – {gdf['TDI'].max():.3f}")
    print(f"  Mean TDI: {gdf['TDI'].mean():.3f}")

    # LISA bivariate clustering
    print("Running LISA bivariate clustering (TDI × Transit Access Delta) …")
    gdf = run_lisa(gdf, var_x="TDI", var_y="access_delta")

    # EJ summary: HH clusters by racial composition
    hh = gdf[gdf["lisa_cluster"] == "HH"]
    if "pct_minority" in hh.columns and len(hh) > 0:
        mean_minority = hh["pct_minority"].mean()
        overall_minority = gdf["pct_minority"].mean()
        print(f"\nEnvironmental Justice:")
        print(f"  HH cluster mean minority share:  {mean_minority:.1%}")
        print(f"  Overall Cook County avg:          {overall_minority:.1%}")
        print(f"  Disparity ratio: {mean_minority/overall_minority:.2f}x")

    # Save
    out = PROC / "equity_analysis.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"\nSaved → {out.name}")

    make_equity_map(gdf)
    make_ej_summary(gdf)


def make_equity_map(gdf: gpd.GeoDataFrame):
    print("\nGenerating equity hot spot map …")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.set_axis_off()

    gdf_4326 = gdf.to_crs("EPSG:4326")

    # Panel 1 — Transit Dependency Index
    gdf_4326.plot(ax=axes[0], column="TDI", cmap="RdYlGn_r",
                  linewidth=0.1, edgecolor="#333333", legend=True,
                  legend_kwds={"label": "Transit Dependency Index (0–1)",
                               "orientation": "horizontal", "pad": 0.03, "shrink": 0.7})
    axes[0].set_title("Transit Dependency Index\n(Low Income + No Vehicle + Transit Commuter)",
                       color="white", fontsize=11, pad=10)

    # Panel 2 — LISA clusters
    cluster_colors = {
        "HH": "#d73027",   # high dependency, high service loss → priority
        "LL": "#4575b4",
        "LH": "#91bfdb",
        "HL": "#fc8d59",
        "NS": "#2a2a3a",
    }
    for cluster, color in cluster_colors.items():
        subset = gdf_4326[gdf_4326["lisa_cluster"] == cluster]
        if not subset.empty:
            subset.plot(ax=axes[1], color=color, linewidth=0.1, edgecolor="#555555")

    legend_patches = [
        mpatches.Patch(color="#d73027", label="HH — High Dependency + High Service Loss"),
        mpatches.Patch(color="#fc8d59", label="HL — High Dependency + Low Service Loss"),
        mpatches.Patch(color="#91bfdb", label="LH — Low Dependency + High Service Loss"),
        mpatches.Patch(color="#4575b4", label="LL — Low Dependency + Low Service Loss"),
        mpatches.Patch(color="#2a2a3a", label="NS — Not Significant"),
    ]
    axes[1].legend(handles=legend_patches, loc="lower left",
                   facecolor="#1e2530", edgecolor="#555555", labelcolor="white",
                   fontsize=8)
    axes[1].set_title("LISA Bivariate Clusters\n(Transit Dependency × Overnight Service Loss)",
                       color="white", fontsize=11, pad=10)

    fig.suptitle("Chicago Transit Equity Analysis\nCook County Block Groups | ACS 2021",
                 color="white", fontsize=14, y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = MAPS / "equity_hotspots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out.name}")


def make_ej_summary(gdf: gpd.GeoDataFrame):
    print("Generating EJ summary chart …")

    cluster_order = ["HH", "HL", "LH", "LL", "NS"]
    cluster_labels = {
        "HH": "HH\n(High Need,\nPoor Service)",
        "HL": "HL\n(High Need,\nOK Service)",
        "LH": "LH\n(Low Need,\nPoor Service)",
        "LL": "LL\n(Low Need,\nOK Service)",
        "NS": "Not\nSignificant",
    }
    cluster_colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4", "#555555"]

    demo_cols = {
        "pct_minority":       "Minority Share",
        "pct_no_vehicle":     "No Vehicle (%)",
        "pct_transit_commute":"Transit Commuter (%)",
    }

    available = {k: v for k, v in demo_cols.items() if k in gdf.columns}

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")

    for ax, (col, label) in zip(axes, available.items()):
        means = []
        for cluster in cluster_order:
            subset = gdf[gdf["lisa_cluster"] == cluster][col].dropna()
            means.append(subset.mean() if len(subset) > 0 else 0)

        bars = ax.bar(
            [cluster_labels[c] for c in cluster_order],
            [m * 100 for m in means],
            color=cluster_colors,
            edgecolor="#333333",
            width=0.6,
        )
        ax.set_facecolor("#161b22")
        ax.set_title(label, color="white", fontsize=11)
        ax.set_ylabel("Percent (%)", color="white")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            ax.spines[spine if isinstance(spine, str) else "top"].set_visible(False)
        ax.spines["bottom"].set_edgecolor("#555555")
        ax.spines["left"].set_edgecolor("#555555")
        ax.yaxis.label.set_color("white")

    fig.suptitle("Demographic Profile by LISA Cluster\nEnvironmental Justice Analysis",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    out = FIGS / "ej_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out.name}")


if __name__ == "__main__":
    run_equity_analysis()
