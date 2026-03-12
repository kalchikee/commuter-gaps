"""
05_route_optimization.py
------------------------
Phase 4: Optimized Route Proposal and Web Visualization

  1. Uses Stranded Worker Density + TDI as demand weights
  2. Identifies optimal stop candidates via greedy location-allocation
  3. Proposes 3 new late-night bus routes connecting high-demand clusters
     to major overnight employment centers (hospitals, airports, distribution)
  4. Estimates coverage improvement
  5. Builds interactive Mapbox GL JS web map (../web/index.html)

Outputs:
  - data/processed/proposed_routes.geojson
  - data/processed/proposed_stops.geojson
  - outputs/maps/proposed_routes.png
  - web/index.html  (interactive map)
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import unary_union
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
WEB  = ROOT / "web"
WEB.mkdir(parents=True, exist_ok=True)

# Known overnight employment anchors in Chicago (lon, lat)
OVERNIGHT_ANCHORS = {
    "Rush University Medical Center": (-87.6713, 41.8738),
    "University of Chicago Medicine":  (-87.5986, 41.7890),
    "O'Hare International Airport":   (-87.9073, 41.9742),
    "Midway Airport":                  (-87.7524, 41.7868),
    "Amazon MDW6 (Markham)":           (-87.6939, 41.6003),
    "UPS Chicago Area Consolidation":  (-87.8200, 41.7200),
    "Northwestern Memorial Hospital":  (-87.6215, 41.8958),
    "Stroger Hospital (Cook County)":  (-87.6752, 41.8716),
}


def load_data() -> tuple:
    equity_path = PROC / "equity_analysis.geojson"
    swd_path    = PROC / "stranded_worker_density.geojson"
    transit_path = PROC / "transit_access_delta.geojson"

    if equity_path.exists():
        gdf = gpd.read_file(equity_path)
    elif transit_path.exists():
        gdf = gpd.read_file(transit_path)
        print("  (Using transit_access_delta — equity analysis not yet run)")
    else:
        raise FileNotFoundError("No processed data found — run earlier phases first")

    if swd_path.exists() and "stranded_density" not in gdf.columns:
        swd = gpd.read_file(swd_path)[["GEOID", "stranded_density", "stranded_per_sqkm"]]
        gdf = gdf.merge(swd, on="GEOID", how="left")

    gdf["stranded_density"] = gdf.get("stranded_density", pd.Series(0, index=gdf.index)).fillna(0)
    if "TDI" not in gdf.columns:
        gdf["TDI"] = 0.5  # neutral fallback

    return gdf


def compute_demand_weights(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Combine stranded density and TDI into a single demand weight."""
    gdf = gdf.copy()

    sd = gdf["stranded_density"].clip(lower=0)
    sd_norm = (sd - sd.min()) / (sd.max() - sd.min() + 1e-9)

    tdi_norm = gdf["TDI"].clip(lower=0, upper=1).fillna(0)

    # Penalize areas already served overnight
    served_penalty = (gdf.get("overnight_stops", pd.Series(0)) > 0).astype(float) * 0.5
    gdf["demand_weight"] = (sd_norm * 0.6 + tdi_norm * 0.4) * (1 - served_penalty * 0.4)

    return gdf


def select_demand_centroids(gdf: gpd.GeoDataFrame,
                             n: int = 30) -> gpd.GeoDataFrame:
    """Select top-N demand-weighted block group centroids as route anchors."""
    gdf_proj = gdf.to_crs("EPSG:26916")
    centroids = gdf_proj.copy()
    centroids["geometry"] = gdf_proj.geometry.centroid

    top = centroids.nlargest(n, "demand_weight")[
        ["GEOID", "demand_weight", "geometry"]
    ].copy()
    return top.to_crs("EPSG:4326")


def greedy_route_builder(demand_pts: gpd.GeoDataFrame,
                          anchors: dict,
                          n_routes: int = 3) -> list:
    """
    Greedy location-allocation:
    For each proposed route, connect the nearest employment anchor to the
    highest-demand unserved cluster via a simplified polyline.
    Returns list of (route_name, LineString, metadata) tuples.
    """
    anchor_gdf = gpd.GeoDataFrame(
        [{"name": k, "geometry": Point(v[0], v[1])} for k, v in anchors.items()],
        crs="EPSG:4326",
    ).to_crs("EPSG:26916")

    demand_proj = demand_pts.to_crs("EPSG:26916")
    remaining   = demand_proj.copy()
    routes      = []

    route_names = [
        "Night Owl Route 1 — South Side Medical Corridor",
        "Night Owl Route 2 — Airport / Logistics Express",
        "Night Owl Route 3 — West Side Hospital Link",
    ]
    anchor_assignments = [
        ["Rush University Medical Center", "Stroger Hospital (Cook County)",
         "University of Chicago Medicine"],
        ["O'Hare International Airport", "Midway Airport",
         "UPS Chicago Area Consolidation", "Amazon MDW6 (Markham)"],
        ["Northwestern Memorial Hospital", "Rush University Medical Center",
         "Stroger Hospital (Cook County)"],
    ]

    for i in range(min(n_routes, len(route_names))):
        if len(remaining) == 0:
            break

        # Top demand centroid for this route
        top_demand = remaining.nlargest(5, "demand_weight")

        # Relevant anchors for this route
        route_anchors = anchor_gdf[
            anchor_gdf["name"].isin(anchor_assignments[i])
        ]

        if route_anchors.empty:
            route_anchors = anchor_gdf

        # Build waypoints: demand centroids + anchor points
        waypoints = []
        for _, row in top_demand.iterrows():
            waypoints.append((row.geometry.x, row.geometry.y))
        for _, row in route_anchors.iterrows():
            waypoints.append((row.geometry.x, row.geometry.y))

        # Simple route: connect demand → closest anchor → next demand
        if len(waypoints) >= 2:
            # Sort west-to-east for a sensible routing
            waypoints_sorted = sorted(waypoints, key=lambda p: p[0])
            line = LineString(waypoints_sorted)

            # Estimate ridership: sum demand weights along corridor
            buffer = gpd.GeoDataFrame(
                geometry=[line.buffer(800)], crs="EPSG:26916"
            )
            joined = gpd.sjoin(remaining, buffer, how="inner", predicate="within")
            estimated_riders = int(joined["demand_weight"].sum() * 150)
            coverage_bgs = len(joined)

            routes.append({
                "name": route_names[i],
                "geometry_proj": line,
                "estimated_daily_riders": estimated_riders,
                "block_groups_served": coverage_bgs,
                "anchors": anchor_assignments[i],
            })

            # Remove served block groups from remaining
            remaining = remaining[~remaining.index.isin(joined.index)]

    return routes


def build_route_geodataframes(routes: list) -> tuple:
    """Convert route dicts to GeoDataFrames."""
    route_records = []
    for i, r in enumerate(routes):
        geom_4326 = gpd.GeoDataFrame(
            geometry=[r["geometry_proj"]], crs="EPSG:26916"
        ).to_crs("EPSG:4326").geometry[0]

        route_records.append({
            "route_id": i + 1,
            "name": r["name"],
            "estimated_daily_riders": r["estimated_daily_riders"],
            "block_groups_served": r["block_groups_served"],
            "anchors": ", ".join(r["anchors"]),
            "geometry": geom_4326,
        })

    routes_gdf = gpd.GeoDataFrame(route_records, crs="EPSG:4326") if route_records else gpd.GeoDataFrame()

    # Proposed stops: interpolated points along each route
    stop_records = []
    colors = ["#FF6B35", "#00B4D8", "#7B2FBE"]
    for i, r in enumerate(routes):
        line = gpd.GeoDataFrame(
            geometry=[r["geometry_proj"]], crs="EPSG:26916"
        ).to_crs("EPSG:4326").geometry[0]

        length = line.length
        n_stops = max(4, min(12, int(length / 0.008)))  # ~1 stop per 0.8km
        for j in range(n_stops + 1):
            pt = line.interpolate(j / n_stops, normalized=True)
            stop_records.append({
                "route_id": i + 1,
                "stop_seq": j,
                "stop_name": f"Route {i+1} Stop {j+1}",
                "color": colors[i],
                "geometry": pt,
            })

    stops_gdf = gpd.GeoDataFrame(stop_records, crs="EPSG:4326") if stop_records else gpd.GeoDataFrame()
    return routes_gdf, stops_gdf


def make_route_map(gdf: gpd.GeoDataFrame,
                   routes_gdf: gpd.GeoDataFrame,
                   stops_gdf: gpd.GeoDataFrame,
                   anchors: dict):
    print("Generating proposed routes map …")

    fig, ax = plt.subplots(figsize=(14, 16))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.set_axis_off()

    gdf_4326 = gdf.to_crs("EPSG:4326")

    # Base layer — stranded worker density
    col = "stranded_per_sqkm" if "stranded_per_sqkm" in gdf_4326.columns else "demand_weight"
    gdf_4326.plot(ax=ax, column=col, cmap="Purples",
                  linewidth=0.05, edgecolor="#222222",
                  legend=True,
                  legend_kwds={"label": "Stranded Workers / km²",
                               "orientation": "horizontal", "pad": 0.02, "shrink": 0.6})

    # Routes
    route_colors = ["#FF6B35", "#00B4D8", "#7B2FBE"]
    route_widths = [3.5, 3.5, 3.5]
    if not routes_gdf.empty:
        for i, (_, row) in enumerate(routes_gdf.iterrows()):
            gpd.GeoDataFrame([row], crs="EPSG:4326").plot(
                ax=ax, color=route_colors[i % 3], linewidth=route_widths[i % 3],
                zorder=5,
            )

    # Stops
    if not stops_gdf.empty:
        stops_gdf.plot(ax=ax, color=stops_gdf["color"].tolist(),
                       markersize=20, zorder=6, edgecolor="white", linewidth=0.5)

    # Employment anchors
    for name, (lon, lat) in anchors.items():
        ax.scatter(lon, lat, c="#FFD700", s=80, zorder=7, marker="*",
                   edgecolors="white", linewidths=0.5)
        ax.annotate(name.split("(")[0].strip(), (lon, lat),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=6, color="#FFD700", zorder=8)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#FF6B35", label="Route 1 — South Side Medical"),
        mpatches.Patch(color="#00B4D8", label="Route 2 — Airport / Logistics"),
        mpatches.Patch(color="#7B2FBE", label="Route 3 — West Side Hospital"),
        plt.scatter([], [], c="#FFD700", s=60, marker="*", label="Employment Anchor"),
    ]
    ax.legend(handles=legend_elements[:3] + [
        plt.Line2D([0], [0], marker="*", color="#FFD700", linestyle="None",
                   markersize=10, label="Employment Anchor")
    ], loc="lower left", facecolor="#1e2530",
              edgecolor="#555555", labelcolor="white", fontsize=8)

    if not routes_gdf.empty:
        summary_lines = []
        for _, r in routes_gdf.iterrows():
            summary_lines.append(
                f"Route {r['route_id']}: ~{r['estimated_daily_riders']:,} est. riders/day"
            )
        ax.text(0.02, 0.15, "\n".join(summary_lines),
                transform=ax.transAxes, color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2530",
                          edgecolor="#555555", alpha=0.9))

    ax.set_title("Proposed Late-Night Bus Routes\nChicago Overnight Transit Gap Analysis",
                  color="white", fontsize=14, pad=12)

    plt.tight_layout()
    out = MAPS / "proposed_routes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out.name}")


def build_web_map(gdf: gpd.GeoDataFrame,
                  routes_gdf: gpd.GeoDataFrame,
                  stops_gdf: gpd.GeoDataFrame,
                  transit_bg: gpd.GeoDataFrame,
                  anchors: dict):
    print("Building interactive web map …")

    # Prepare GeoJSON payloads (simplified for web)
    gdf_web = gdf.to_crs("EPSG:4326")[
        ["GEOID", "geometry",
         "overnight_stops", "peak_stops", "access_delta", "service_class",
         "TDI", "stranded_density", "lisa_cluster"]
    ].copy()
    for col in ["overnight_stops", "peak_stops", "access_delta",
                "TDI", "stranded_density"]:
        if col in gdf_web.columns:
            gdf_web[col] = gdf_web[col].fillna(0).round(2)
    for col in ["service_class", "lisa_cluster"]:
        if col in gdf_web.columns:
            gdf_web[col] = gdf_web[col].fillna("Unknown")

    bg_geojson      = gdf_web.to_json()
    routes_geojson  = routes_gdf.to_json() if not routes_gdf.empty else '{"type":"FeatureCollection","features":[]}'
    stops_geojson   = stops_gdf.to_json() if not stops_gdf.empty else '{"type":"FeatureCollection","features":[]}'

    anchor_features = []
    for name, (lon, lat) in anchors.items():
        anchor_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name},
        })
    anchors_geojson = json.dumps({"type": "FeatureCollection", "features": anchor_features})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>The Invisible Commute — Chicago Overnight Transit Gaps</title>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet" />
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }}

    #app {{ display: flex; flex-direction: column; height: 100vh; }}

    /* Header */
    #header {{
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 100;
    }}
    #header h1 {{ font-size: 16px; font-weight: 600; color: #f0f6fc; }}
    #header p  {{ font-size: 12px; color: #8b949e; }}

    /* Main layout */
    #main {{ display: flex; flex: 1; overflow: hidden; }}

    /* Sidebar */
    #sidebar {{
      width: 320px;
      background: #161b22;
      border-right: 1px solid #30363d;
      overflow-y: auto;
      padding: 16px;
      flex-shrink: 0;
    }}

    .panel-section {{
      margin-bottom: 20px;
      border-bottom: 1px solid #21262d;
      padding-bottom: 16px;
    }}
    .panel-section h3 {{
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8b949e;
      margin-bottom: 10px;
    }}

    /* Layer toggles */
    .layer-toggle {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 6px 0;
      cursor: pointer;
      border-radius: 4px;
    }}
    .layer-toggle:hover {{ background: #1c2128; padding-left: 4px; }}
    .layer-toggle input[type=checkbox] {{ accent-color: #2f81f7; width: 15px; height: 15px; }}
    .layer-toggle label {{ font-size: 13px; cursor: pointer; user-select: none; }}

    /* Time slider */
    #time-slider-container {{
      background: #1c2128;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    #time-display {{
      text-align: center;
      font-size: 28px;
      font-weight: 700;
      color: #2f81f7;
      font-variant-numeric: tabular-nums;
    }}
    #time-label {{ text-align: center; font-size: 11px; color: #8b949e; margin-top: 2px; }}
    #time-slider {{
      width: 100%;
      margin-top: 10px;
      accent-color: #2f81f7;
    }}
    #time-btn {{
      width: 100%;
      margin-top: 8px;
      padding: 6px;
      background: #2f81f7;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    }}
    #time-btn:hover {{ background: #388bfd; }}

    /* Legend */
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    .legend-line {{
      width: 24px;
      height: 4px;
      border-radius: 2px;
      flex-shrink: 0;
    }}

    /* Info panel */
    #info-panel {{
      background: #1c2128;
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.6;
      min-height: 60px;
      color: #8b949e;
    }}
    #info-panel strong {{ color: #f0f6fc; }}

    /* Stats */
    .stat-row {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      padding: 4px 0;
      border-bottom: 1px solid #21262d;
    }}
    .stat-val {{ font-weight: 600; color: #2f81f7; }}

    /* Map */
    #map {{ flex: 1; }}

    /* Token warning */
    #token-warning {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: #161b22;
      border: 1px solid #f85149;
      border-radius: 8px;
      padding: 20px 28px;
      text-align: center;
      z-index: 999;
      max-width: 420px;
    }}
    #token-warning h2 {{ color: #f85149; margin-bottom: 8px; font-size: 16px; }}
    #token-warning p  {{ color: #8b949e; font-size: 13px; line-height: 1.5; }}
    #token-input {{
      width: 100%;
      margin-top: 12px;
      padding: 8px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 4px;
      color: #e6edf3;
      font-size: 13px;
    }}
    #token-btn {{
      margin-top: 8px;
      padding: 8px 16px;
      background: #2f81f7;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
    }}
  </style>
</head>
<body>
<div id="app">

  <div id="header">
    <div>
      <h1>The Invisible Commute: Chicago Overnight Transit Gaps</h1>
      <p>Mapping shift worker accessibility gaps between midnight and 5 AM</p>
    </div>
    <div style="font-size:11px; color:#8b949e; text-align:right;">
      CTA + Pace GTFS &nbsp;|&nbsp; LEHD LODES 2021 &nbsp;|&nbsp; ACS 2021
    </div>
  </div>

  <div id="main">
    <div id="sidebar">

      <!-- Time of day -->
      <div class="panel-section">
        <h3>Time of Day</h3>
        <div id="time-slider-container">
          <div id="time-display">8:00 AM</div>
          <div id="time-label">Peak Service</div>
          <input type="range" id="time-slider" min="0" max="23" value="8" step="1" />
          <button id="time-btn" onclick="animateTime()">▶ Animate 24-hr Cycle</button>
        </div>
      </div>

      <!-- Layer toggles -->
      <div class="panel-section">
        <h3>Layers</h3>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-transit" checked onchange="toggleLayer('transit-fill', this)"/>
          <label for="toggle-transit">Transit Coverage (block groups)</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-stranded" onchange="toggleLayer('stranded-fill', this)"/>
          <label for="toggle-stranded">Stranded Worker Density</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-equity" onchange="toggleLayer('equity-fill', this)"/>
          <label for="toggle-equity">Equity Hot Spots (LISA)</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-routes" checked onchange="toggleRouteLayer(this)"/>
          <label for="toggle-routes">Proposed Night Owl Routes</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-anchors" checked onchange="toggleLayer('anchors-sym', this)"/>
          <label for="toggle-anchors">Employment Anchors</label>
        </div>
      </div>

      <!-- Legend -->
      <div class="panel-section">
        <h3>Legend</h3>
        <div style="font-size:11px; color:#8b949e; margin-bottom:6px;">Transit Access Delta</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#1a9850"></div> Moderate overnight service</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#fee08b"></div> Limited service</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#fc8d59"></div> Very limited</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#d73027"></div> No overnight service</div>
        <div style="margin-top:10px; font-size:11px; color:#8b949e; margin-bottom:6px;">Proposed Routes</div>
        <div class="legend-item"><div class="legend-line" style="background:#FF6B35"></div> Route 1 — South Side Medical</div>
        <div class="legend-item"><div class="legend-line" style="background:#00B4D8"></div> Route 2 — Airport / Logistics</div>
        <div class="legend-item"><div class="legend-line" style="background:#7B2FBE"></div> Route 3 — West Side Hospital</div>
        <div style="margin-top:10px; font-size:11px; color:#8b949e; margin-bottom:6px;">LISA Clusters</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#d73027; opacity:0.7"></div> HH — High Need + Poor Service</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#fc8d59; opacity:0.7"></div> HL — High Need + OK Service</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#91bfdb; opacity:0.7"></div> LH — Low Need + Poor Service</div>
      </div>

      <!-- Stats -->
      <div class="panel-section">
        <h3>Key Findings</h3>
        <div class="stat-row">
          <span>Block groups, no overnight service</span>
          <span class="stat-val" id="stat-no-service">—</span>
        </div>
        <div class="stat-row">
          <span>Mean transit access delta</span>
          <span class="stat-val" id="stat-delta">—</span>
        </div>
        <div class="stat-row">
          <span>Estimated unserved trips/day</span>
          <span class="stat-val" id="stat-unserved">—</span>
        </div>
        <div class="stat-row">
          <span>Proposed routes</span>
          <span class="stat-val">3</span>
        </div>
      </div>

      <!-- Feature info -->
      <div class="panel-section" style="border-bottom: none;">
        <h3>Selected Feature</h3>
        <div id="info-panel">Click a block group to see details.</div>
      </div>

    </div>

    <div id="map">
      <div id="token-warning">
        <h2>Mapbox Token Required</h2>
        <p>This interactive map requires a free Mapbox public token.
           Get one at <a href="https://mapbox.com" target="_blank" style="color:#2f81f7;">mapbox.com</a>
           and enter it below.</p>
        <input type="text" id="token-input" placeholder="pk.eyJ1IjoiL..." />
        <br/>
        <button id="token-btn" onclick="initMap()">Load Map</button>
      </div>
    </div>
  </div>
</div>

<script>
// ── Embedded GeoJSON ──────────────────────────────────────────────────────────
const BG_DATA      = {bg_geojson};
const ROUTES_DATA  = {routes_geojson};
const STOPS_DATA   = {stops_geojson};
const ANCHORS_DATA = {anchors_geojson};

// ── Compute stats ─────────────────────────────────────────────────────────────
function computeStats() {{
  let noService = 0, deltaSum = 0, unservedSum = 0, count = 0;
  BG_DATA.features.forEach(f => {{
    const p = f.properties;
    if (p.overnight_stops === 0) noService++;
    if (p.access_delta != null) {{ deltaSum += p.access_delta; count++; }}
    if (p.stranded_density != null) unservedSum += p.stranded_density;
  }});
  document.getElementById("stat-no-service").textContent = noService.toLocaleString();
  document.getElementById("stat-delta").textContent =
    count > 0 ? (deltaSum / count).toFixed(1) + "%" : "—";
  document.getElementById("stat-unserved").textContent =
    unservedSum > 0 ? Math.round(unservedSum).toLocaleString() : "—";
}}
computeStats();

// ── Time slider ───────────────────────────────────────────────────────────────
let animInterval = null;
const timeDisplay = document.getElementById("time-display");
const timeLabel   = document.getElementById("time-label");
const timeSlider  = document.getElementById("time-slider");

function formatHour(h) {{
  const suffix = h < 12 ? "AM" : "PM";
  const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
  return `${{h12}}:00 ${{suffix}}`;
}}
function getTimeLabel(h) {{
  if (h >= 7 && h <= 9) return "AM Peak";
  if (h >= 16 && h <= 18) return "PM Peak";
  if (h >= 0 && h <= 5) return "Overnight — Minimal Service";
  if (h >= 22 || h <= 1) return "Late Night — Reduced Service";
  return "Off-Peak";
}}

timeSlider.addEventListener("input", function() {{
  const h = parseInt(this.value);
  timeDisplay.textContent = formatHour(h);
  timeLabel.textContent   = getTimeLabel(h);
  if (map) updateTransitLayer(h);
}});

function animateTime() {{
  if (animInterval) {{ clearInterval(animInterval); animInterval = null;
    document.getElementById("time-btn").textContent = "▶ Animate 24-hr Cycle"; return; }}
  document.getElementById("time-btn").textContent = "■ Stop Animation";
  animInterval = setInterval(() => {{
    let h = (parseInt(timeSlider.value) + 1) % 24;
    timeSlider.value = h;
    timeDisplay.textContent = formatHour(h);
    timeLabel.textContent   = getTimeLabel(h);
    if (map) updateTransitLayer(h);
  }}, 600);
}}

// Approximate overnight service as fraction of peak (simplified)
function getServiceFraction(h) {{
  if (h >= 7 && h <= 9)   return 1.0;
  if (h >= 16 && h <= 18) return 0.95;
  if (h >= 10 && h <= 15) return 0.80;
  if (h >= 19 && h <= 21) return 0.65;
  if (h >= 22 || h <= 0)  return 0.30;
  return 0.15;  // 1–5 AM
}}

// ── Map ───────────────────────────────────────────────────────────────────────
let map = null;

function toggleLayer(layerId, cb) {{
  if (!map) return;
  map.setLayoutProperty(layerId, 'visibility', cb.checked ? 'visible' : 'none');
}}
function toggleRouteLayer(cb) {{
  if (!map) return;
  ['route-line-1','route-line-2','route-line-3','route-stops'].forEach(id => {{
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', cb.checked ? 'visible' : 'none');
  }});
}}

function updateTransitLayer(h) {{
  if (!map || !map.getLayer('transit-fill')) return;
  const frac = getServiceFraction(h);
  // Adjust color opacity to simulate reduced coverage
  map.setPaintProperty('transit-fill', 'fill-opacity',
    ['case',
      ['==', ['get','overnight_stops'], 0],   frac < 0.5 ? 0.7 : 0.3,
      0.55
    ]);
}}

function initMap() {{
  const token = document.getElementById("token-input").value.trim();
  if (!token) {{ alert("Please enter your Mapbox token."); return; }}

  document.getElementById("token-warning").style.display = "none";
  mapboxgl.accessToken = token;

  map = new mapboxgl.Map({{
    container: "map",
    style: "mapbox://styles/mapbox/dark-v11",
    center: [-87.65, 41.84],
    zoom: 10,
    pitch: 10,
  }});

  map.addControl(new mapboxgl.NavigationControl(), "top-right");
  map.addControl(new mapboxgl.ScaleControl({{ unit: "metric" }}), "bottom-right");

  map.on("load", () => {{

    // ── Block group sources ────────────────────────────────────────────────
    map.addSource("block-groups", {{ type: "geojson", data: BG_DATA }});

    // Transit coverage layer
    map.addLayer({{
      id: "transit-fill",
      type: "fill",
      source: "block-groups",
      paint: {{
        "fill-color": [
          "match", ["get","service_class"],
          "No Service",   "#d73027",
          "Very Limited", "#fc8d59",
          "Limited",      "#fee08b",
          "Moderate",     "#1a9850",
          "#2a2a3a"
        ],
        "fill-opacity": 0.55,
        "fill-outline-color": "#333333",
      }}
    }});

    // Stranded worker density (hidden by default)
    map.addLayer({{
      id: "stranded-fill",
      type: "fill",
      source: "block-groups",
      layout: {{ visibility: "none" }},
      paint: {{
        "fill-color": [
          "interpolate", ["linear"], ["get","stranded_density"],
          0, "#0d1117",
          10, "#3b1578",
          50, "#7b2fbe",
          200, "#c77dff",
          500, "#ff9ef7"
        ],
        "fill-opacity": 0.7,
        "fill-outline-color": "#333333",
      }}
    }});

    // LISA equity clusters (hidden by default)
    map.addLayer({{
      id: "equity-fill",
      type: "fill",
      source: "block-groups",
      layout: {{ visibility: "none" }},
      paint: {{
        "fill-color": [
          "match", ["get","lisa_cluster"],
          "HH", "#d73027",
          "HL", "#fc8d59",
          "LH", "#91bfdb",
          "LL", "#4575b4",
          "rgba(0,0,0,0)"
        ],
        "fill-opacity": 0.7,
        "fill-outline-color": "#555555",
      }}
    }});

    // Block group borders
    map.addLayer({{
      id: "bg-outline",
      type: "line",
      source: "block-groups",
      paint: {{ "line-color": "#333333", "line-width": 0.3 }}
    }});

    // ── Proposed routes ────────────────────────────────────────────────────
    map.addSource("routes", {{ type: "geojson", data: ROUTES_DATA }});
    const routeColors = ["#FF6B35", "#00B4D8", "#7B2FBE"];
    for (let i = 1; i <= 3; i++) {{
      map.addLayer({{
        id: `route-line-${{i}}`,
        type: "line",
        source: "routes",
        filter: ["==", ["get","route_id"], i],
        paint: {{
          "line-color": routeColors[i-1],
          "line-width": 4,
          "line-opacity": 0.9,
        }}
      }});
    }}

    // Proposed stops
    map.addSource("stops", {{ type: "geojson", data: STOPS_DATA }});
    map.addLayer({{
      id: "route-stops",
      type: "circle",
      source: "stops",
      paint: {{
        "circle-radius": 5,
        "circle-color": ["get","color"],
        "circle-stroke-width": 1.5,
        "circle-stroke-color": "#ffffff",
        "circle-opacity": 0.9,
      }}
    }});

    // ── Employment anchors ─────────────────────────────────────────────────
    map.addSource("anchors", {{ type: "geojson", data: ANCHORS_DATA }});
    map.addLayer({{
      id: "anchors-halo",
      type: "circle",
      source: "anchors",
      paint: {{
        "circle-radius": 14,
        "circle-color": "rgba(255,215,0,0.15)",
        "circle-stroke-width": 1.5,
        "circle-stroke-color": "#FFD700",
      }}
    }});
    map.addLayer({{
      id: "anchors-sym",
      type: "symbol",
      source: "anchors",
      layout: {{
        "text-field": ["get","name"],
        "text-font": ["DIN Offc Pro Medium","Arial Unicode MS Regular"],
        "text-size": 10,
        "text-offset": [0, 1.5],
        "text-anchor": "top",
        "text-max-width": 12,
      }},
      paint: {{
        "text-color": "#FFD700",
        "text-halo-color": "#0d1117",
        "text-halo-width": 1,
      }}
    }});

    // ── Popups ─────────────────────────────────────────────────────────────
    map.on("click", "transit-fill", (e) => {{
      const p = e.features[0].properties;
      document.getElementById("info-panel").innerHTML = `
        <strong>Block Group:</strong> ${{p.GEOID || "—"}}<br/>
        <strong>Overnight Stops (400m):</strong> ${{p.overnight_stops ?? "—"}}<br/>
        <strong>Peak Stops (400m):</strong> ${{p.peak_stops ?? "—"}}<br/>
        <strong>Transit Access Delta:</strong> ${{p.access_delta ?? "—"}}%<br/>
        <strong>Service Class:</strong> ${{p.service_class || "—"}}<br/>
        <strong>Transit Dependency Index:</strong> ${{p.TDI != null ? (+p.TDI).toFixed(3) : "—"}}<br/>
        <strong>Stranded Workers:</strong> ${{p.stranded_density != null ? Math.round(p.stranded_density) : "—"}}<br/>
        <strong>LISA Cluster:</strong> ${{p.lisa_cluster || "—"}}
      `;
    }});
    map.on("click", "route-line-1", (e) => showRouteInfo(e));
    map.on("click", "route-line-2", (e) => showRouteInfo(e));
    map.on("click", "route-line-3", (e) => showRouteInfo(e));
    map.on("click", "anchors-sym", (e) => {{
      const p = e.features[0].properties;
      document.getElementById("info-panel").innerHTML =
        `<strong>Employment Anchor</strong><br/>${{p.name}}`;
    }});

    map.on("mouseenter", "transit-fill",   () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "transit-fill",   () => map.getCanvas().style.cursor = "");
    map.on("mouseenter", "route-line-1",   () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "route-line-1",   () => map.getCanvas().style.cursor = "");

  }});
}}

function showRouteInfo(e) {{
  const p = e.features[0].properties;
  document.getElementById("info-panel").innerHTML = `
    <strong>${{p.name || "Proposed Route"}}</strong><br/>
    Est. daily riders: <strong>${{p.estimated_daily_riders != null ? (+p.estimated_daily_riders).toLocaleString() : "—"}}</strong><br/>
    Block groups served: <strong>${{p.block_groups_served ?? "—"}}</strong><br/>
    Anchors: ${{p.anchors || "—"}}
  `;
}}
</script>
</body>
</html>
"""

    out = WEB / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved → {out}")


def run_route_optimization():
    print("=" * 60)
    print("Phase 4: Route Optimization and Web Map")
    print("=" * 60)

    print("Loading processed data …")
    try:
        gdf = load_data()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    transit_bg = gpd.read_file(PROC / "transit_access_delta.geojson") \
        if (PROC / "transit_access_delta.geojson").exists() else gdf

    print("Computing demand weights …")
    gdf = compute_demand_weights(gdf)

    print("Selecting demand centroids …")
    demand_pts = select_demand_centroids(gdf, n=40)
    print(f"  Top demand centroids: {len(demand_pts):,}")

    print("Running greedy route builder …")
    routes = greedy_route_builder(demand_pts, OVERNIGHT_ANCHORS, n_routes=3)
    print(f"  Routes proposed: {len(routes)}")
    for r in routes:
        print(f"    {r['name']}")
        print(f"      Est. riders/day: {r['estimated_daily_riders']:,}")
        print(f"      Block groups served: {r['block_groups_served']}")

    routes_gdf, stops_gdf = build_route_geodataframes(routes)

    # Save
    if not routes_gdf.empty:
        routes_gdf.to_file(PROC / "proposed_routes.geojson", driver="GeoJSON")
        stops_gdf.to_file(PROC / "proposed_stops.geojson", driver="GeoJSON")
        print(f"\nSaved → proposed_routes.geojson, proposed_stops.geojson")

    make_route_map(gdf, routes_gdf, stops_gdf, OVERNIGHT_ANCHORS)
    build_web_map(gdf, routes_gdf, stops_gdf, transit_bg, OVERNIGHT_ANCHORS)

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    run_route_optimization()
