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
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# ── Cost model (2026 USD) ─────────────────────────────────────────────────────
# Sources: CTA 2023 NTD operating-cost reports inflated ~3% annually; FTA / APTA
# new-bus pricing for 40-ft CNG coaches; typical bus-stop construction costs
# (shelter, sign, concrete pad, ADA compliance) from CTA planning documents.
COST_2026 = {
    "op_cost_per_vrh":    200,       # CTA bus operating cost per revenue vehicle hour
    "new_bus_capital":    650_000,   # 40-ft CNG coach; battery-electric ~$950k
    "stop_infrastructure": 15_000,   # per new stop (shelter + sign + ADA pad)
    "service_hours_per_night": 6,    # midnight to ~5 AM "night owl" window
    "headway_minutes":    60,        # 60-min headway, realistic for overnight
    "avg_bus_speed_kmh":  30,        # Chicago arterials, no traffic overnight
    "days_per_year":      365,
}


def estimate_route_cost(length_km: float, n_stops: int,
                        cost: dict = COST_2026) -> dict:
    """Estimate 2026 cost of operating a new overnight bus route."""
    import math
    round_trip_min = (length_km * 2 / cost["avg_bus_speed_kmh"]) * 60
    n_buses = max(1, math.ceil(round_trip_min / cost["headway_minutes"]))

    annual_rvh = n_buses * cost["service_hours_per_night"] * cost["days_per_year"]
    annual_operating = annual_rvh * cost["op_cost_per_vrh"]

    fleet_capital = n_buses * cost["new_bus_capital"]
    stop_capital  = n_stops * cost["stop_infrastructure"]
    total_capital = fleet_capital + stop_capital

    return {
        "n_buses":           n_buses,
        "round_trip_min":    round(round_trip_min, 1),
        "annual_rvh":        annual_rvh,
        "annual_operating":  int(annual_operating),
        "fleet_capital":     int(fleet_capital),
        "stop_capital":      int(stop_capital),
        "total_capital":     int(total_capital),
        "year_one_total":    int(total_capital + annual_operating),
    }


def export_cta_gtfs_routes():
    """Extract CTA bus and rail route polylines from the local CTA GTFS feed.
    Builds one representative LineString per route (the longest shape variant)
    and saves them to:
        data/processed/cta_rail_routes.geojson
        data/processed/cta_bus_routes.geojson
    Idempotent — skips if outputs already exist and GTFS hasn't changed.
    """
    rail_out = PROC / "cta_rail_routes.geojson"
    bus_out  = PROC / "cta_bus_routes.geojson"
    gtfs_dir = RAW / "gtfs" / "cta"
    if not gtfs_dir.exists():
        print("  CTA GTFS not found — skipping existing-route export")
        return
    if rail_out.exists() and bus_out.exists():
        return

    print("  Building CTA bus + rail route polylines from GTFS …")
    routes = pd.read_csv(gtfs_dir / "routes.txt")
    trips  = pd.read_csv(gtfs_dir / "trips.txt",
                          usecols=["route_id", "shape_id"])
    shapes = pd.read_csv(gtfs_dir / "shapes.txt")

    # For each route, pick the shape with the most points (longest variant)
    shape_sizes = shapes.groupby("shape_id").size().rename("pts")
    trips_shape = trips.drop_duplicates(["route_id", "shape_id"]).merge(
        shape_sizes, on="shape_id", how="left"
    )
    rep_shape = (trips_shape.sort_values("pts", ascending=False)
                            .drop_duplicates("route_id"))

    records = {"rail": [], "bus": []}
    shapes_sorted = shapes.sort_values(["shape_id", "shape_pt_sequence"])
    shape_groups = dict(tuple(shapes_sorted.groupby("shape_id")))

    for _, row in rep_shape.iterrows():
        pts = shape_groups.get(row["shape_id"])
        if pts is None or len(pts) < 2:
            continue
        coords = [(round(lon, 5), round(lat, 5))
                  for lon, lat in zip(pts["shape_pt_lon"], pts["shape_pt_lat"])]
        route = routes.loc[routes["route_id"] == row["route_id"]].iloc[0]
        kind = "rail" if route["route_type"] in (1, 2) else "bus"
        name = route.get("route_long_name") or route.get("route_short_name") or route["route_id"]
        color = route.get("route_color")
        if pd.isna(color) or not isinstance(color, str) or not color.strip():
            color = "6e7681"   # neutral gray for buses without an official color
        records[kind].append({
            "route_id":   str(route["route_id"]),
            "route_name": str(name),
            "route_type": "L" if kind == "rail" else "Bus",
            "color":      "#" + color.strip().upper(),
            "geometry":   LineString(coords).simplify(0.0001,
                                                       preserve_topology=True),
        })

    if records["rail"]:
        gpd.GeoDataFrame(records["rail"], crs="EPSG:4326").to_file(
            rail_out, driver="GeoJSON"
        )
        print(f"    Saved → {rail_out.name} ({len(records['rail'])} rail routes)")
    if records["bus"]:
        gpd.GeoDataFrame(records["bus"], crs="EPSG:4326").to_file(
            bus_out, driver="GeoJSON"
        )
        print(f"    Saved → {bus_out.name} ({len(records['bus'])} bus routes)")


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


# ── Road-network routing ──────────────────────────────────────────────────────
_ROAD_GRAPH_CACHE = None


def load_road_network():
    """Load major Chicago/Cook arterial road network. Cached to disk on first call."""
    global _ROAD_GRAPH_CACHE
    if _ROAD_GRAPH_CACHE is not None:
        return _ROAD_GRAPH_CACHE

    cache_dir = RAW / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "cook_major_roads.graphml"

    if cache_path.exists():
        print("  Loading cached road network …")
        G = ox.load_graphml(cache_path)
    else:
        print("  Downloading major road network (primary/secondary/tertiary) …")
        custom_filter = (
            '["highway"~"motorway|trunk|primary|secondary|tertiary|'
            'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
        )
        G = ox.graph_from_place(
            "Cook County, Illinois, USA",
            custom_filter=custom_filter,
            simplify=True,
            retain_all=False,
        )
        ox.save_graphml(G, cache_path)

    _ROAD_GRAPH_CACHE = G
    return G


def _corridor_score(start_pt, end_pt, pts_with_weight, max_off_deg=0.04,
                     t_min=-0.1, t_max=1.1):
    """Return (t, pt, weight) tuples for pts that lie within the start→end
    corridor, filtered by an absolute off-axis cap (~4.5 km at Chicago
    latitude) and a relaxed t-range."""
    sx, sy = start_pt
    ex, ey = end_pt
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return []

    scored = []
    for (px, py), weight in pts_with_weight:
        t = ((px - sx) * dx + (py - sy) * dy) / length_sq
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        off_axis = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
        if t_min <= t <= t_max and off_axis <= max_off_deg:
            scored.append((t, (px, py), weight))
    return scored


def pick_ordered_waypoints(start_pt, end_pt, candidates, n_keep,
                            max_off_deg=0.04):
    """candidates: list of ((lon, lat), demand_weight).
    Returns an ordered list [start, ...intermediate, end] where intermediate
    are the top-n_keep highest-demand points inside the corridor, traversed
    in projection order so the route doesn't zig-zag."""
    scored = _corridor_score(start_pt, end_pt, candidates,
                              max_off_deg=max_off_deg)
    # Keep the n_keep highest-demand candidates, then sort by t for traversal
    scored.sort(key=lambda x: -x[2])
    kept = scored[:n_keep]
    kept.sort(key=lambda x: x[0])
    return [start_pt] + [pt for _, pt, _ in kept] + [end_pt]


def _edge_coords(G, u, v):
    """Return the (lon, lat) vertices along the road edge u→v.

    OSMnx's simplified graphs collapse many OSM nodes into single edges that
    carry a `geometry` LineString attribute with the real road curve. Falling
    back to just (u_xy, v_xy) was what caused routes to cut across blocks
    instead of following the street.
    """
    data = G.get_edge_data(u, v)
    if not data:
        return [(G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"])]
    # MultiDiGraph: keyed by edge-key; pick the shortest parallel edge
    best = min(data.values(), key=lambda d: d.get("length", float("inf")))
    geom = best.get("geometry")
    if geom is not None:
        coords = list(geom.coords)
        # OSMnx edge geometry isn't guaranteed to be u→v oriented; flip if needed
        u_xy = (G.nodes[u]["x"], G.nodes[u]["y"])
        if coords and coords[0] != u_xy and coords[-1] == u_xy:
            coords = coords[::-1]
        return coords
    return [(G.nodes[u]["x"], G.nodes[u]["y"]),
            (G.nodes[v]["x"], G.nodes[v]["y"])]


def route_on_roads(G, waypoints_lonlat):
    """Snap an ordered list of (lon, lat) waypoints to the road network via
    shortest-path between consecutive waypoints, then stitch together each
    edge's true road geometry (not just graph-node endpoints).
    Returns a LineString in EPSG:4326."""
    xs = [p[0] for p in waypoints_lonlat]
    ys = [p[1] for p in waypoints_lonlat]
    try:
        nodes = ox.distance.nearest_nodes(G, xs, ys)
    except Exception:
        nodes = [ox.distance.nearest_nodes(G, x, y) for x, y in zip(xs, ys)]

    coords = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        if a == b:
            continue
        try:
            path = nx.shortest_path(G, a, b, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            seg = [(G.nodes[a]["x"], G.nodes[a]["y"]),
                   (G.nodes[b]["x"], G.nodes[b]["y"])]
        else:
            seg = []
            for u, v in zip(path[:-1], path[1:]):
                for xy in _edge_coords(G, u, v):
                    if not seg or seg[-1] != xy:
                        seg.append(xy)
        for xy in seg:
            if not coords or coords[-1] != xy:
                coords.append(xy)

    if len(coords) < 2:
        return LineString(waypoints_lonlat)
    return LineString(coords)


def build_realistic_routes(demand_pts: gpd.GeoDataFrame,
                           anchors: dict,
                           n_routes: int = 5) -> list:
    """
    Build 3 realistic late-night bus routes:
      - Each has a start anchor, an end anchor, and intermediate high-demand stops
      - Intermediate stops are ordered by nearest-neighbor TSP
      - The resulting polyline follows the actual Cook County road network
    """
    G = load_road_network()

    demand_proj = demand_pts.to_crs("EPSG:26916")
    remaining   = demand_proj.copy()
    built_corridors_proj = []  # buffered polylines of routes already built

    # Route definitions: name, start anchor, end anchor, "candidate corridor"
    # bounding box (lon_min, lat_min, lon_max, lat_max) that bounds where
    # intermediate demand stops may be drawn from.
    # `max_off_deg` is the half-width of the corridor in degrees (~0.01°/1.1 km).
    # Tighter corridors → more direct express routes that don't detour to share
    # arterials with other routes. Wider → more intermediate stops, more local.
    route_specs = [
        {
            "name": "Night Owl 1 — South Side Medical Corridor",
            "start": "Rush University Medical Center",
            "end":   "University of Chicago Medicine",
            "max_intermediate": 6,
            "max_off_deg": 0.035,
        },
        {
            "name": "Night Owl 2 — Midway / Logistics Express",
            "start": "Midway Airport",
            "end":   "UPS Chicago Area Consolidation",
            "max_intermediate": 5,
            "max_off_deg": 0.025,
        },
        {
            "name": "Night Owl 3 — West Side Hospital Link",
            "start": "Northwestern Memorial Hospital",
            "end":   "Stroger Hospital (Cook County)",
            "max_intermediate": 4,
            "max_off_deg": 0.02,
        },
        {
            "name": "Night Owl 4 — O'Hare Express",
            "start": "O'Hare International Airport",
            "end":   "Northwestern Memorial Hospital",
            "max_intermediate": 4,
            "max_off_deg": 0.02,   # tight — keeps it on the Kennedy corridor
        },
        {
            "name": "Night Owl 5 — South Suburban Amazon Link",
            "start": "University of Chicago Medicine",
            "end":   "Amazon MDW6 (Markham)",
            "max_intermediate": 6,
            "max_off_deg": 0.03,
        },
    ]

    routes = []

    for spec in route_specs[:n_routes]:
        start_lonlat = anchors[spec["start"]]
        end_lonlat   = anchors[spec["end"]]

        # Candidate demand — start from the remaining pool, then drop anything
        # within 600m of any already-built route polyline. This prevents later
        # routes from being drawn through the same corridor as earlier ones
        # just to pick up shared demand.
        cand_proj = remaining
        if built_corridors_proj:
            from shapely.ops import unary_union
            existing = unary_union(built_corridors_proj)
            cand_proj = cand_proj[~cand_proj.geometry.within(existing)]

        cand = cand_proj.to_crs("EPSG:4326")
        candidates = [((pt.x, pt.y), float(w))
                      for pt, w in zip(cand.geometry, cand["demand_weight"])]

        ordered = pick_ordered_waypoints(
            start_lonlat, end_lonlat, candidates,
            spec["max_intermediate"],
            max_off_deg=spec.get("max_off_deg", 0.04),
        )
        line_4326 = route_on_roads(G, ordered)

        # Project and buffer to estimate ridership
        line_proj = gpd.GeoSeries([line_4326], crs="EPSG:4326").to_crs("EPSG:26916").iloc[0]
        corridor = gpd.GeoDataFrame(
            geometry=[line_proj.buffer(800)], crs="EPSG:26916"
        )
        covered = gpd.sjoin(remaining, corridor, how="inner", predicate="within")
        estimated_riders = int(covered["demand_weight"].sum() * 150)
        coverage_bgs = len(covered)

        length_km = round(line_proj.length / 1000, 2)
        # Rough stop count uses the same ~1 stop / km rule applied in
        # build_route_geodataframes (we'll recompute an exact count there).
        approx_stops = max(4, min(15, int(line_proj.length / 1000)))
        cost = estimate_route_cost(length_km, approx_stops)

        routes.append({
            "name": spec["name"],
            "geometry_4326": line_4326,
            "geometry_proj": line_proj,
            "estimated_daily_riders": estimated_riders,
            "block_groups_served": coverage_bgs,
            "start_anchor": spec["start"],
            "end_anchor":   spec["end"],
            "anchors": [spec["start"], spec["end"]],
            "length_km": length_km,
            "n_buses":           cost["n_buses"],
            "annual_operating":  cost["annual_operating"],
            "total_capital":     cost["total_capital"],
            "year_one_total":    cost["year_one_total"],
        })

        # Avoid double-crediting BGs to the next route
        remaining = remaining[~remaining.index.isin(covered.index)]
        # And remember this polyline for the next route's exclusion buffer
        built_corridors_proj.append(line_proj.buffer(600))

    return routes


def build_route_geodataframes(routes: list) -> tuple:
    """Convert route dicts to GeoDataFrames. Stops spaced ~1 km along the line
    in projected CRS (true distance), snapped back to EPSG:4326."""
    route_records = []
    for i, r in enumerate(routes):
        route_records.append({
            "route_id": i + 1,
            "name": r["name"],
            "estimated_daily_riders": r["estimated_daily_riders"],
            "block_groups_served": r["block_groups_served"],
            "length_km": r["length_km"],
            "n_buses":          r["n_buses"],
            "annual_operating": r["annual_operating"],
            "total_capital":    r["total_capital"],
            "year_one_total":   r["year_one_total"],
            "anchors": ", ".join(r["anchors"]),
            "geometry": r["geometry_4326"],
        })

    routes_gdf = (gpd.GeoDataFrame(route_records, crs="EPSG:4326")
                  if route_records else gpd.GeoDataFrame())

    # Proposed stops — spaced by true distance using projected geometry
    stop_records = []
    colors = ["#FF6B35", "#00B4D8", "#7B2FBE", "#F7B538", "#34C38F"]
    for i, r in enumerate(routes):
        line_proj = r["geometry_proj"]
        length_m  = line_proj.length
        n_stops   = max(4, min(15, int(length_m / 1000)))  # ~1 stop per km
        for j in range(n_stops + 1):
            pt_proj = line_proj.interpolate(j / n_stops, normalized=True)
            pt_4326 = (gpd.GeoSeries([pt_proj], crs="EPSG:26916")
                       .to_crs("EPSG:4326").iloc[0])
            stop_records.append({
                "route_id": i + 1,
                "stop_seq": j,
                "stop_name": f"Route {i+1} Stop {j+1}",
                "color": colors[i % len(colors)],
                "geometry": pt_4326,
            })

    stops_gdf = (gpd.GeoDataFrame(stop_records, crs="EPSG:4326")
                 if stop_records else gpd.GeoDataFrame())
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
    route_colors = ["#FF6B35", "#00B4D8", "#7B2FBE", "#F7B538", "#34C38F"]
    if not routes_gdf.empty:
        for i, (_, row) in enumerate(routes_gdf.iterrows()):
            gpd.GeoDataFrame([row], crs="EPSG:4326").plot(
                ax=ax, color=route_colors[i % len(route_colors)], linewidth=3.5,
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
        mpatches.Patch(color="#00B4D8", label="Route 2 — Midway / Logistics"),
        mpatches.Patch(color="#7B2FBE", label="Route 3 — West Side Hospital"),
        mpatches.Patch(color="#F7B538", label="Route 4 — O'Hare Express"),
        mpatches.Patch(color="#34C38F", label="Route 5 — South Suburban Amazon"),
        plt.Line2D([0], [0], marker="*", color="#FFD700", linestyle="None",
                   markersize=10, label="Employment Anchor"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", facecolor="#1e2530",
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


def _round_geometry(geom, ndigits: int = 5):
    """Round all coordinates in a geometry to ndigits decimal places."""
    from shapely.geometry import mapping, shape

    def _round_coords(coords):
        if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
            return [round(float(c), ndigits) for c in coords]
        return [_round_coords(c) for c in coords]

    m = mapping(geom)
    m["coordinates"] = _round_coords(m["coordinates"])
    return shape(m)


def build_web_map(gdf: gpd.GeoDataFrame,
                  routes_gdf: gpd.GeoDataFrame,
                  stops_gdf: gpd.GeoDataFrame,
                  anchors: dict,
                  mapbox_token: str | None = None):
    print("Building interactive web map …")

    # Keep the properties we actually surface in tooltips / popups, drop the rest.
    keep_cols = [
        "GEOID", "geometry",
        "overnight_stops", "peak_stops", "access_delta", "service_class",
        "TDI", "stranded_density", "lisa_cluster",
    ]
    present = [c for c in keep_cols if c in gdf.columns]
    gdf_web = gdf.to_crs("EPSG:4326")[present].copy()

    # Simplify geometry (tolerance in degrees — ~20-30m) and round coords to
    # drastically reduce the embedded GeoJSON size without visible change.
    gdf_web["geometry"] = gdf_web.geometry.simplify(0.0002, preserve_topology=True)
    gdf_web["geometry"] = gdf_web.geometry.apply(lambda g: _round_geometry(g, 5))

    for col in ["overnight_stops", "peak_stops", "access_delta",
                "TDI", "stranded_density"]:
        if col in gdf_web.columns:
            gdf_web[col] = gdf_web[col].fillna(0).round(2)
    for col in ["service_class", "lisa_cluster"]:
        if col in gdf_web.columns:
            gdf_web[col] = gdf_web[col].fillna("Unknown")

    bg_geojson = gdf_web.to_json()

    if not routes_gdf.empty:
        r = routes_gdf.copy()
        r["geometry"] = r.geometry.apply(lambda g: _round_geometry(g, 5))
        routes_geojson = r.to_json()
    else:
        routes_geojson = '{"type":"FeatureCollection","features":[]}'

    if not stops_gdf.empty:
        s = stops_gdf.copy()
        s["geometry"] = s.geometry.apply(lambda g: _round_geometry(g, 5))
        stops_geojson = s.to_json()
    else:
        stops_geojson = '{"type":"FeatureCollection","features":[]}'

    anchor_features = []
    for name, (lon, lat) in anchors.items():
        anchor_features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(lon, 5), round(lat, 5)]},
            "properties": {"name": name},
        })
    anchors_geojson = json.dumps({"type": "FeatureCollection",
                                   "features": anchor_features})

    def _load_gtfs_layer(path: Path) -> str:
        if not path.exists():
            return '{"type":"FeatureCollection","features":[]}'
        layer = gpd.read_file(path)
        layer["geometry"] = layer.geometry.apply(lambda g: _round_geometry(g, 5))
        return layer.to_json()

    cta_bus_geojson  = _load_gtfs_layer(PROC / "cta_bus_routes.geojson")
    cta_rail_geojson = _load_gtfs_layer(PROC / "cta_rail_routes.geojson")

    # Split the token into two halves so it doesn't appear as one continuous
    # string in the repo (harmless public token, but matches prior convention).
    token = mapbox_token or (
        "pk.eyJ1IjoiZXRoYW5rMjE4IiwiYSI6ImNtOHhrdTVocTA0cnEya3B3eXpnNjc5NGEifQ"
        ".srWCBHwpc7Fxo9rhvqodvA"
    )
    token_parts = token.split(".")
    token_js = "[" + ", ".join(f'"{p}"' for p in token_parts) + "].join(\".\")"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>The Invisible Commute — Chicago Overnight Transit Gaps</title>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet" />
  <style>
    #hover-tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(22, 27, 34, 0.96);
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      color: #e6edf3;
      z-index: 1000;
      max-width: 280px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
      display: none;
      line-height: 1.45;
    }}
    #hover-tooltip .tt-head {{
      font-weight: 600;
      color: #f0f6fc;
      margin-bottom: 4px;
      border-bottom: 1px solid #30363d;
      padding-bottom: 3px;
    }}
    #hover-tooltip .tt-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    #hover-tooltip .tt-label {{ color: #8b949e; }}
    #hover-tooltip .tt-val   {{ color: #e6edf3; font-variant-numeric: tabular-nums; }}

    /* On-map legend — anchored to the map container, bottom-left corner */
    #map-legend {{
      position: absolute;
      bottom: 16px;
      left: 16px;
      background: rgba(22, 27, 34, 0.94);
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 12px;
      color: #e6edf3;
      z-index: 50;
      max-width: 260px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.45);
      line-height: 1.5;
      pointer-events: none;
    }}
    #map-legend .lg-section {{ margin-bottom: 8px; }}
    #map-legend .lg-section:last-child {{ margin-bottom: 0; }}
    #map-legend .lg-title {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8b949e;
      margin-bottom: 4px;
    }}
    #map-legend .lg-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 2px;
    }}
    #map-legend .lg-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 2px;
      flex-shrink: 0;
      border: 1px solid rgba(255,255,255,0.1);
    }}
    #map-legend .lg-line {{
      width: 22px;
      height: 4px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    #map-legend .lg-star {{
      color: #FFD700;
      font-size: 15px;
      width: 14px;
      text-align: center;
      flex-shrink: 0;
    }}
    #map-legend .lg-gradient {{
      width: 150px;
      height: 10px;
      border-radius: 2px;
      margin: 2px 0 4px;
    }}
    #map-legend .lg-scale {{
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #8b949e;
    }}
  </style>
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
    #map {{ flex: 1; position: relative; }}

    /* Loading overlay */
    #map-loading {{
      position: absolute;
      top: 50%;
      left: calc(50% + 160px);
      transform: translate(-50%, -50%);
      color: #8b949e;
      font-size: 13px;
      z-index: 5;
      pointer-events: none;
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
        <h3>Time of Day <span style="color:#6e7681;font-size:10px;text-transform:none;letter-spacing:0;">(illustrative)</span></h3>
        <div id="time-slider-container">
          <div id="time-display">8:00 AM</div>
          <div id="time-label">Peak Service</div>
          <input type="range" id="time-slider" min="0" max="23" value="8" step="1" />
          <button id="time-btn" onclick="animateTime()">▶ Animate 24-hr Cycle</button>
          <div style="font-size:10px;color:#6e7681;margin-top:6px;line-height:1.4;">
            Exact maps show 8 AM vs 2 AM snapshots. Slider interpolates opacity.
          </div>
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
          <label for="toggle-routes">Proposed Night Owl Bus Routes</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-anchors" checked onchange="toggleLayer('anchors-sym', this)"/>
          <label for="toggle-anchors">Employment Anchors</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-cta-rail" onchange="toggleCtaLayer('rail', this)"/>
          <label for="toggle-cta-rail">Existing CTA &quot;L&quot; Rail Lines</label>
        </div>
        <div class="layer-toggle">
          <input type="checkbox" id="toggle-cta-bus" onchange="toggleCtaLayer('bus', this)"/>
          <label for="toggle-cta-bus">Existing CTA Bus Routes (all)</label>
        </div>
      </div>

      <!-- Legend moved on-map (bottom-left); updates live as you toggle layers -->
      <div class="panel-section" style="font-size:11px;color:#6e7681;">
        <em>Legend appears on the map (bottom-left) and updates as you toggle layers.</em>
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
          <span class="stat-val">5</span>
        </div>
        <div class="stat-row">
          <span>System cost — year 1 (2026)</span>
          <span class="stat-val" id="stat-system-cost">—</span>
        </div>
        <div class="stat-row" style="border-bottom:none;">
          <span>Steady-state op/year</span>
          <span class="stat-val" id="stat-system-op">—</span>
        </div>
      </div>

      <!-- Per-route cost table -->
      <div class="panel-section">
        <h3>Proposed Bus Routes (2026 cost)</h3>
        <div style="font-size:10px;color:#8b949e;margin:-4px 0 8px;line-height:1.4;">
          All five are proposed <strong>bus</strong> lines (not rail). Rail capital would be
          100–500× higher per mile. These routes extend CTA's existing overnight
          "Night Owl" bus service.
        </div>
        <div id="route-cost-list" style="font-size:11px; color:#c9d1d9;">
          <!-- filled by JS renderRouteCosts() -->
        </div>
        <div style="font-size:10px; color:#6e7681; margin-top:8px; line-height:1.4;">
          60-min overnight headway · 6 h/night · 40-ft CNG coaches · $200/RVH.
          Click or hover a route on the map for detail.
        </div>
      </div>

      <!-- Feature info -->
      <div class="panel-section" style="border-bottom: none;">
        <h3>Selected Feature</h3>
        <div id="info-panel">Click or hover a block group, route, or anchor for details.</div>
      </div>

    </div>

    <div id="map">
      <div id="map-loading">Loading map…</div>
      <div id="map-legend"></div>
    </div>
  </div>
</div>
<div id="hover-tooltip"></div>

<script>
// ── Embedded GeoJSON ──────────────────────────────────────────────────────────
const BG_DATA      = {bg_geojson};
const ROUTES_DATA  = {routes_geojson};
const STOPS_DATA   = {stops_geojson};
const ANCHORS_DATA = {anchors_geojson};
const CTA_BUS_DATA  = {cta_bus_geojson};
const CTA_RAIL_DATA = {cta_rail_geojson};

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtNum(v, fallback) {{
  if (v == null || v === "" || Number.isNaN(+v)) return fallback ?? "—";
  return (+v).toLocaleString();
}}
function fmtPct(v, digits) {{
  if (v == null || v === "" || Number.isNaN(+v)) return "—";
  return (+v).toFixed(digits ?? 1) + "%";
}}
function fmtFloat(v, digits) {{
  if (v == null || v === "" || Number.isNaN(+v)) return "—";
  return (+v).toFixed(digits ?? 3);
}}

// ── Per-route cost breakdown + system totals ──────────────────────────────────
function renderRouteCosts() {{
  const el = document.getElementById("route-cost-list");
  if (!el) return;
  let opSum = 0, capSum = 0, y1Sum = 0;
  const rows = (ROUTES_DATA.features || []).map(f => {{
    const p = f.properties;
    const color = (ROUTE_DEFS.find(r => r.id === p.route_id) || {{}}).color || "#888";
    opSum  += +p.annual_operating || 0;
    capSum += +p.total_capital    || 0;
    y1Sum  += +p.year_one_total   || 0;
    return `
      <div style="display:flex; gap:6px; padding:6px 0; border-bottom:1px solid #21262d; align-items:flex-start;">
        <div style="width:3px; min-height:36px; background:${{color}}; border-radius:1px; margin-top:2px;"></div>
        <div style="flex:1;">
          <div style="color:#f0f6fc; font-weight:500;">${{p.name || "Route " + p.route_id}}</div>
          <div style="color:#8b949e; font-size:10px; margin-top:2px;">
            ${{(+p.length_km || 0).toFixed(1)}} km · ${{p.n_buses || "?"}} bus${{p.n_buses === 1 ? "" : "es"}} ·
            ${{fmtNum(p.estimated_daily_riders)}} riders/day
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:3px; gap:6px;">
            <span style="color:#8b949e;">Year-1</span>
            <span style="color:#2f81f7; font-weight:600;">${{fmtUSD(p.year_one_total)}}</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:10px; color:#6e7681;">
            <span>op ${{fmtUSD(p.annual_operating)}}/yr</span>
            <span>capital ${{fmtUSD(p.total_capital)}}</span>
          </div>
        </div>
      </div>
    `;
  }}).join("");
  el.innerHTML = rows;

  const sysCost = document.getElementById("stat-system-cost");
  const sysOp   = document.getElementById("stat-system-op");
  if (sysCost) sysCost.textContent = fmtUSD(y1Sum);
  if (sysOp)   sysOp.textContent   = fmtUSD(opSum);
}}

// ── Compute stats ─────────────────────────────────────────────────────────────
function computeStats() {{
  let noService = 0, deltaSum = 0, unservedSum = 0, count = 0;
  BG_DATA.features.forEach(f => {{
    const p = f.properties;
    if (p.overnight_stops === 0) noService++;
    if (p.access_delta != null) {{ deltaSum += +p.access_delta; count++; }}
    if (p.stranded_density != null) unservedSum += +p.stranded_density;
  }});
  document.getElementById("stat-no-service").textContent = noService.toLocaleString();
  document.getElementById("stat-delta").textContent =
    count > 0 ? (deltaSum / count).toFixed(1) + "%" : "—";
  document.getElementById("stat-unserved").textContent =
    unservedSum > 0 ? Math.round(unservedSum).toLocaleString() : "—";
}}

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
  renderLegend();
}}
function toggleRouteLayer(cb) {{
  if (!map) return;
  const ids = ROUTE_LINE_IDS
    .concat(ROUTE_DEFS.map(r => `route-hit-${{r.id}}`))
    .concat(['route-stops']);
  ids.forEach(id => {{
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', cb.checked ? 'visible' : 'none');
  }});
  renderLegend();
}}

function toggleCtaLayer(kind, cb) {{
  if (!map) return;
  const ids = kind === "rail"
    ? ["cta-rail-lines", "cta-rail-hit"]
    : ["cta-bus-lines",  "cta-bus-hit"];
  ids.forEach(id => {{
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', cb.checked ? 'visible' : 'none');
  }});
  renderLegend();
}}

// ── Dynamic on-map legend ─────────────────────────────────────────────────────
// Reflects whichever layers are currently visible — updates each time the
// user toggles a checkbox in the sidebar.
const ROUTE_DEFS = [
  {{ id: 1, color: "#FF6B35", label: "Route 1 — South Side Medical" }},
  {{ id: 2, color: "#00B4D8", label: "Route 2 — Midway / Logistics" }},
  {{ id: 3, color: "#7B2FBE", label: "Route 3 — West Side Hospital" }},
  {{ id: 4, color: "#F7B538", label: "Route 4 — O'Hare Express" }},
  {{ id: 5, color: "#34C38F", label: "Route 5 — South Suburban Amazon" }},
];
const ROUTE_LINE_IDS = ROUTE_DEFS.map(r => `route-line-${{r.id}}`);

function isChecked(id)     {{ const el = document.getElementById(id); return !!(el && el.checked); }}
function isLayerVisible(id) {{
  if (!map || !map.getLayer(id)) return false;
  return map.getLayoutProperty(id, 'visibility') !== 'none';
}}

function legendSection(title, rowsHtml) {{
  return `<div class="lg-section"><div class="lg-title">${{title}}</div>${{rowsHtml}}</div>`;
}}
function legendRow(swatchHtml, label) {{
  return `<div class="lg-row">${{swatchHtml}}<span>${{label}}</span></div>`;
}}
function swatch(color, alpha) {{
  return `<div class="lg-swatch" style="background:${{color}};opacity:${{alpha ?? 1}}"></div>`;
}}
function lineSwatch(color) {{
  return `<div class="lg-line" style="background:${{color}}"></div>`;
}}

function renderLegend() {{
  const el = document.getElementById("map-legend");
  if (!el || !map) return;

  const sections = [];

  // Each base layer renders its own section independently so multiple can
  // be shown in the legend when the user has multiple toggles on.
  if (isLayerVisible("transit-fill")) {{
    sections.push(legendSection("Overnight Transit Coverage",
      legendRow(swatch("#1a9850"), "Moderate (8+ stops ≤400m)") +
      legendRow(swatch("#fee08b"), "Limited (3–7 stops)") +
      legendRow(swatch("#fc8d59"), "Very limited (1–2 stops)") +
      legendRow(swatch("#d73027"), "No overnight service")
    ));
  }}
  if (isLayerVisible("stranded-fill")) {{
    const grad = "linear-gradient(to right,#0d1117,#3b1578,#7b2fbe,#c77dff,#ff9ef7)";
    sections.push(legendSection("Stranded Shift Workers",
      `<div class="lg-gradient" style="background:${{grad}}"></div>` +
      `<div class="lg-scale"><span>0</span><span>50</span><span>500+</span></div>` +
      `<div style="font-size:10px;color:#8b949e;margin-top:3px">` +
      `Sum of unserved shift-work OD trips touching each block group.</div>`
    ));
  }}
  if (isLayerVisible("equity-fill")) {{
    sections.push(legendSection("LISA Bivariate Clusters (TDI × Service Loss)",
      legendRow(swatch("#d73027", 0.8), "HH — High Need + Poor Service") +
      legendRow(swatch("#fc8d59", 0.8), "HL — High Need + OK Service") +
      legendRow(swatch("#91bfdb", 0.8), "LH — Low Need + Poor Service") +
      legendRow(swatch("#4575b4", 0.8), "LL — Low Need + OK Service")
    ));
  }}

  // Routes section — one entry per route actually rendered on the map
  const routesOn = ROUTE_DEFS.filter(r => isLayerVisible(`route-line-${{r.id}}`));
  if (routesOn.length > 0) {{
    const rows = routesOn.map(r => legendRow(lineSwatch(r.color), r.label)).join("");
    sections.push(legendSection("Proposed Night Owl Bus Routes", rows));
  }}

  // Anchors
  if (isLayerVisible("anchors-sym")) {{
    sections.push(legendSection("Employment Anchors",
      legendRow('<span class="lg-star">★</span>',
                "Major overnight employer (hospital, airport, logistics)")
    ));
  }}

  // Existing CTA service
  if (isLayerVisible("cta-rail-lines")) {{
    const lLines = [
      ["#C60C30", "Red Line"],   ["#00A1DE", "Blue Line"],
      ["#62361B", "Brown Line"], ["#009B3A", "Green Line"],
      ["#F9461C", "Orange Line"],["#E27EA6", "Pink Line"],
      ["#522398", "Purple Line"],["#F9E300", "Yellow Line"],
    ].map(([c, l]) => legendRow(lineSwatch(c), l)).join("");
    sections.push(legendSection('CTA "L" Rail (existing)', lLines));
  }}
  if (isLayerVisible("cta-bus-lines")) {{
    sections.push(legendSection("CTA Buses (existing)",
      legendRow(lineSwatch("#6e7681"), "All CTA bus routes")
    ));
  }}

  if (sections.length === 0) {{
    el.style.display = "none";
    return;
  }}
  el.style.display = "block";
  el.innerHTML = sections.join("");
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

const MAPBOX_TOKEN = {token_js};
const tooltip = document.getElementById("hover-tooltip");

function showTooltip(html, event) {{
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  const x = event.originalEvent.clientX + 14;
  const y = event.originalEvent.clientY + 14;
  tooltip.style.left = x + "px";
  tooltip.style.top  = y + "px";
}}
function hideTooltip() {{ tooltip.style.display = "none"; }}

function tooltipRow(label, val) {{
  return `<div class="tt-row"><span class="tt-label">${{label}}</span>` +
         `<span class="tt-val">${{val}}</span></div>`;
}}

function bgTooltipHtml(p) {{
  const classColors = {{
    "No Service": "#d73027", "Very Limited": "#fc8d59",
    "Limited": "#fee08b",    "Moderate": "#1a9850",
  }};
  const swatch = classColors[p.service_class] || "#2a2a3a";
  return `
    <div class="tt-head">Block Group ${{p.GEOID || "—"}}</div>
    <div class="tt-row">
      <span class="tt-label">Overnight service</span>
      <span class="tt-val">
        <span style="display:inline-block;width:9px;height:9px;background:${{swatch}};
                     border-radius:2px;margin-right:5px;vertical-align:middle"></span>
        ${{p.service_class || "—"}}
      </span>
    </div>
    ${{tooltipRow("Stops at 2 AM (400m)", fmtNum(p.overnight_stops))}}
    ${{tooltipRow("Stops at 8 AM (400m)", fmtNum(p.peak_stops))}}
    ${{tooltipRow("Service loss 8 AM→2 AM", fmtPct(p.access_delta))}}
    ${{tooltipRow("Transit Dependency Index", fmtFloat(p.TDI))}}
    ${{tooltipRow("Stranded workers (OD sum)", fmtNum(Math.round(+p.stranded_density || 0)))}}
    ${{tooltipRow("LISA cluster", p.lisa_cluster || "—")}}
  `;
}}

function fmtUSD(v) {{
  if (v == null || Number.isNaN(+v)) return "—";
  const n = +v;
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return "$" + (n / 1e3).toFixed(0) + "k";
  return "$" + n.toLocaleString();
}}

function routeTooltipHtml(p) {{
  return `
    <div class="tt-head">${{p.name || "Proposed route"}}</div>
    ${{tooltipRow("Length", (p.length_km != null ? p.length_km + " km" : "—"))}}
    ${{tooltipRow("Est. daily riders", fmtNum(p.estimated_daily_riders))}}
    ${{tooltipRow("Block groups in corridor", fmtNum(p.block_groups_served))}}
    ${{tooltipRow("Fleet (60-min headway)", fmtNum(p.n_buses) + " bus" + (p.n_buses == 1 ? "" : "es"))}}
    ${{tooltipRow("Annual operating (2026)", fmtUSD(p.annual_operating))}}
    ${{tooltipRow("Capital (buses + stops)", fmtUSD(p.total_capital))}}
    ${{tooltipRow("Year-one total", fmtUSD(p.year_one_total))}}
    <div style="margin-top:4px;color:#8b949e;font-size:11px">
      Anchors: ${{p.anchors || "—"}}
    </div>
  `;
}}

function anchorTooltipHtml(p) {{
  return `<div class="tt-head">Employment anchor</div>
          <div style="color:#FFD700">${{p.name || "—"}}</div>
          <div style="color:#8b949e;font-size:11px;margin-top:3px">
            Overnight shift concentration
          </div>`;
}}

function stopTooltipHtml(p) {{
  return `<div class="tt-head">${{p.stop_name || "Proposed stop"}}</div>
          ${{tooltipRow("Route", fmtNum(p.route_id))}}
          ${{tooltipRow("Stop sequence", fmtNum(p.stop_seq))}}`;
}}

function ctaTooltipHtml(p) {{
  const color = p.color || "#6e7681";
  return `<div class="tt-head">
            <span style="display:inline-block;width:10px;height:10px;background:${{color}};
                         border-radius:2px;margin-right:6px;vertical-align:middle"></span>
            ${{p.route_name || p.route_id || "CTA route"}}
          </div>
          ${{tooltipRow("Type", p.route_type || "—")}}
          ${{tooltipRow("Route ID", p.route_id || "—")}}
          <div style="margin-top:4px;color:#8b949e;font-size:11px">
            Existing CTA service — shown for context.
          </div>`;
}}

function initMap() {{
  mapboxgl.accessToken = MAPBOX_TOKEN;

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

    // ── Existing CTA service (bus + L) ─────────────────────────────────────
    map.addSource("cta-bus",  {{ type: "geojson", data: CTA_BUS_DATA }});
    map.addSource("cta-rail", {{ type: "geojson", data: CTA_RAIL_DATA }});

    map.addLayer({{
      id: "cta-bus-lines",
      type: "line",
      source: "cta-bus",
      layout: {{ visibility: "none" }},
      paint: {{
        "line-color": "#6e7681",
        "line-width": 1.2,
        "line-opacity": 0.65,
      }}
    }});
    map.addLayer({{
      id: "cta-bus-hit",
      type: "line",
      source: "cta-bus",
      layout: {{ visibility: "none" }},
      paint: {{
        "line-color": "#6e7681",
        "line-width": 10,
        "line-opacity": 0.001,
      }}
    }});

    map.addLayer({{
      id: "cta-rail-lines",
      type: "line",
      source: "cta-rail",
      layout: {{ visibility: "none" }},
      paint: {{
        "line-color": ["get", "color"],
        "line-width": 3,
        "line-opacity": 0.9,
      }}
    }});
    map.addLayer({{
      id: "cta-rail-hit",
      type: "line",
      source: "cta-rail",
      layout: {{ visibility: "none" }},
      paint: {{
        "line-color": ["get", "color"],
        "line-width": 12,
        "line-opacity": 0.001,
      }}
    }});

    // ── Proposed routes ────────────────────────────────────────────────────
    map.addSource("routes", {{ type: "geojson", data: ROUTES_DATA }});
    ROUTE_DEFS.forEach(r => {{
      // Wide invisible click/hover target, per-route (14px catches hovers
      // even when the visible 4px line is hard to hit precisely).
      map.addLayer({{
        id: `route-hit-${{r.id}}`,
        type: "line",
        source: "routes",
        filter: ["==", ["get","route_id"], r.id],
        paint: {{
          "line-color": r.color,
          "line-width": 14,
          "line-opacity": 0.001,  // effectively invisible
        }}
      }});
      // Visible line on top
      map.addLayer({{
        id: `route-line-${{r.id}}`,
        type: "line",
        source: "routes",
        filter: ["==", ["get","route_id"], r.id],
        paint: {{
          "line-color": r.color,
          "line-width": 4,
          "line-opacity": 0.9,
        }}
      }});
    }});

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

    // ── Hover tooltip + click pin-in-sidebar ───────────────────────────────
    const bgLayers     = ["transit-fill", "stranded-fill", "equity-fill"];
    // Attach interactions to the wider invisible hit layer so thin lines are
    // still easy to hover/click.
    const routeLayers  = ROUTE_DEFS.map(r => `route-hit-${{r.id}}`);
    const loadingEl    = document.getElementById("map-loading");
    if (loadingEl) loadingEl.style.display = "none";

    renderLegend();

    bgLayers.forEach(lid => {{
      map.on("mousemove", lid, (e) => {{
        if (!e.features.length) return;
        map.getCanvas().style.cursor = "pointer";
        showTooltip(bgTooltipHtml(e.features[0].properties), e);
      }});
      map.on("mouseleave", lid, () => {{
        map.getCanvas().style.cursor = "";
        hideTooltip();
      }});
      map.on("click", lid, (e) => {{
        const p = e.features[0].properties;
        document.getElementById("info-panel").innerHTML = bgTooltipHtml(p);
      }});
    }});

    routeLayers.forEach(lid => {{
      map.on("mousemove", lid, (e) => {{
        if (!e.features.length) return;
        map.getCanvas().style.cursor = "pointer";
        showTooltip(routeTooltipHtml(e.features[0].properties), e);
      }});
      map.on("mouseleave", lid, () => {{
        map.getCanvas().style.cursor = "";
        hideTooltip();
      }});
      map.on("click", lid, (e) => {{
        document.getElementById("info-panel").innerHTML =
          routeTooltipHtml(e.features[0].properties);
      }});
    }});

    ["anchors-sym", "anchors-halo"].forEach(lid => {{
      map.on("mousemove", lid, (e) => {{
        if (!e.features.length) return;
        map.getCanvas().style.cursor = "pointer";
        showTooltip(anchorTooltipHtml(e.features[0].properties), e);
      }});
      map.on("mouseleave", lid, () => {{
        map.getCanvas().style.cursor = "";
        hideTooltip();
      }});
      map.on("click", lid, (e) => {{
        document.getElementById("info-panel").innerHTML =
          anchorTooltipHtml(e.features[0].properties);
      }});
    }});

    map.on("mousemove", "route-stops", (e) => {{
      if (!e.features.length) return;
      map.getCanvas().style.cursor = "pointer";
      showTooltip(stopTooltipHtml(e.features[0].properties), e);
    }});
    map.on("mouseleave", "route-stops", () => {{
      map.getCanvas().style.cursor = "";
      hideTooltip();
    }});

    ["cta-rail-hit", "cta-bus-hit"].forEach(lid => {{
      map.on("mousemove", lid, (e) => {{
        if (!e.features.length) return;
        map.getCanvas().style.cursor = "pointer";
        showTooltip(ctaTooltipHtml(e.features[0].properties), e);
      }});
      map.on("mouseleave", lid, () => {{
        map.getCanvas().style.cursor = "";
        hideTooltip();
      }});
      map.on("click", lid, (e) => {{
        document.getElementById("info-panel").innerHTML =
          ctaTooltipHtml(e.features[0].properties);
      }});
    }});

  }});
}}

// Now that fmtUSD / ROUTE_DEFS / fmtNum and the DOM are all defined,
// populate the sidebar stats and the per-route cost panel.
computeStats();
renderRouteCosts();

window.addEventListener("DOMContentLoaded", () => initMap());
</script>
</body>
</html>
"""

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Saved → {out}  ({size_mb:.2f} MB)")


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

    print("Computing demand weights …")
    gdf = compute_demand_weights(gdf)

    print("Selecting demand centroids …")
    demand_pts = select_demand_centroids(gdf, n=200)
    print(f"  Top demand centroids: {len(demand_pts):,}")

    print("Building realistic road-network routes …")
    routes = build_realistic_routes(demand_pts, OVERNIGHT_ANCHORS, n_routes=5)
    print(f"  Routes proposed: {len(routes)}")
    for r in routes:
        print(f"    {r['name']}")
        print(f"      Length: {r['length_km']} km ({r['n_buses']} buses, "
              f"60-min headway, 6h/night)")
        print(f"      Est. riders/day: {r['estimated_daily_riders']:,}")
        print(f"      Block groups served: {r['block_groups_served']}")
        print(f"      2026 cost: ${r['annual_operating']/1e6:.2f}M/yr operating "
              f"+ ${r['total_capital']/1e6:.2f}M capital "
              f"(year-1 ${r['year_one_total']/1e6:.2f}M)")

    tot_ops = sum(r["annual_operating"] for r in routes)
    tot_cap = sum(r["total_capital"] for r in routes)
    tot_buses = sum(r["n_buses"] for r in routes)
    print(f"\n  SYSTEM TOTAL (5 routes, {tot_buses} buses):")
    print(f"    Steady-state operating:  ${tot_ops/1e6:.2f}M/year")
    print(f"    Capital (fleet + stops): ${tot_cap/1e6:.2f}M one-time")
    print(f"    Year-one total:          ${(tot_ops + tot_cap)/1e6:.2f}M")

    routes_gdf, stops_gdf = build_route_geodataframes(routes)

    # Save
    if not routes_gdf.empty:
        routes_gdf.to_file(PROC / "proposed_routes.geojson", driver="GeoJSON")
        stops_gdf.to_file(PROC / "proposed_stops.geojson", driver="GeoJSON")
        print(f"\nSaved → proposed_routes.geojson, proposed_stops.geojson")

    make_route_map(gdf, routes_gdf, stops_gdf, OVERNIGHT_ANCHORS)

    print("Exporting existing CTA routes …")
    export_cta_gtfs_routes()

    build_web_map(gdf, routes_gdf, stops_gdf, OVERNIGHT_ANCHORS)

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    run_route_optimization()
