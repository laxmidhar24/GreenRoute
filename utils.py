import math
import random
from typing import Dict, List, Tuple

import polyline
import requests

Node = Tuple[float, float]


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def haversine_km(a: Node, b: Node) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def encode_path_to_polyline(path: List[Node]) -> str:
    return polyline.encode([(float(a), float(b)) for a, b in path])


# ------------------------------------------------------------------
# Google Elevation API
# ------------------------------------------------------------------
def compute_elevation_gain(polyline_str: str, api_key: str, max_points: int = 100) -> float:
    """Total uphill climb (metres) along a route. Returns 0.0 on any failure."""
    try:
        pts = polyline.decode(polyline_str)
        if len(pts) > max_points:
            step = len(pts) / max_points
            pts = [pts[int(i * step)] for i in range(max_points)] + [pts[-1]]
        locations = "|".join(f"{la},{lo}" for la, lo in pts)
        r = requests.get(
            "https://maps.googleapis.com/maps/api/elevation/json",
            params={"locations": locations, "key": api_key},
            timeout=15,
        ).json()
        if r.get("status") != "OK":
            print("Elevation API:", r.get("status"), r.get("error_message"))
            return 0.0
        elev = [x["elevation"] for x in r["results"]]
        return float(sum(max(0.0, b - a) for a, b in zip(elev, elev[1:])))
    except Exception as e:
        print("Elevation error:", e)
        return 0.0


# ------------------------------------------------------------------
# Rule-based emission estimate (kg CO2) - fallback + training label
# ------------------------------------------------------------------
def estimate_emission(distance_km, elevation_gain_m, base_emission_per_km, congestion_index) -> float:
    base = base_emission_per_km * distance_km * (1.0 + 0.5 * max(congestion_index, 0.0))
    climb = 0.0005 * elevation_gain_m * (base_emission_per_km / 0.192)
    return base + climb


# ------------------------------------------------------------------
# AQI via OpenWeather Air Pollution API (optional)
# ------------------------------------------------------------------
_PM25_BP = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
]


def _pm25_to_aqi(pm25: float) -> int:
    for c_lo, c_hi, i_lo, i_hi in _PM25_BP:
        if pm25 <= c_hi:
            return int(round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo))
    return 500


def aqi_category(aqi) -> str:
    if aqi is None:
        return "N/A"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 200:
        return "Unhealthy for sensitive groups"
    if aqi <= 300:
        return "Poor"
    return "Hazardous"


def get_aqi(polyline_str: str, api_key: str):
    """AQI at the route midpoint. Returns (aqi, category); (None, 'N/A') if unavailable."""
    if not api_key:
        return None, "N/A"
    try:
        coords = polyline.decode(polyline_str)
        lat, lon = coords[len(coords) // 2]
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": api_key},
            timeout=10,
        ).json()
        pm25 = r["list"][0]["components"]["pm2_5"]
        aqi = _pm25_to_aqi(pm25)
        return aqi, aqi_category(aqi)
    except Exception as e:
        print("AQI error:", e)
        return None, "N/A"


# ------------------------------------------------------------------
# Graph built from the alternative routes
# graph[u][v] = (distance_km, emission_kg)
# ------------------------------------------------------------------
def build_graph_from_routes(route_objs: List[Dict], ndigits: int = 3):
    """
    route_objs: [{"polyline": str, "emission_kg": float}, ...]
    Points are rounded (~110 m at ndigits=3) so routes that overlap share nodes;
    this is what lets ants 'switch' between alternatives.
    Returns (graph, start, end).
    """
    graph: Dict[Node, Dict[Node, Tuple[float, float]]] = {}
    start = end = None

    for ro in route_objs:
        raw = polyline.decode(ro["polyline"])
        pts: List[Node] = []
        for la, lo in raw:
            p = (round(la, ndigits), round(lo, ndigits))
            if not pts or pts[-1] != p:
                pts.append(p)
        if len(pts) < 2:
            continue
        if start is None:
            start, end = pts[0], pts[-1]

        seg = [haversine_km(a, b) for a, b in zip(pts, pts[1:])]
        total = sum(seg) or 1e-9
        emis_per_km = ro["emission_kg"] / total

        for (a, b), d in zip(zip(pts, pts[1:]), seg):
            e = d * emis_per_km
            for u, v in ((a, b), (b, a)):
                graph.setdefault(u, {})
                # shared edge: keep the cheaper emission estimate
                if v not in graph[u] or e < graph[u][v][1]:
                    graph[u][v] = (d, e)
    return graph, start, end


# ------------------------------------------------------------------
# Ant Colony Optimization
# cost(edge) = w_dist * distance + w_emis * emission
# ------------------------------------------------------------------
def aco_optimize(graph, start, end, n_ants=20, n_iter=40, alpha=1.0, beta=2.0,
                 rho=0.1, w_dist=0.3, w_emis=0.7, seed=None) -> Dict:
    if seed is not None:
        random.seed(seed)
    if not graph or start not in graph or end not in graph:
        return {}

    def ecost(u, v):
        d, e = graph[u][v]
        return w_dist * d + w_emis * e

    pher = {(u, v): 1.0 for u in graph for v in graph[u]}
    dist_to_end = {n: haversine_km(n, end) for n in graph}
    max_steps = len(graph)
    best_path, best_cost = None, float("inf")

    for _ in range(n_iter):
        solutions = []
        for _ant in range(n_ants):
            cur, path, visited = start, [start], {start}
            for _step in range(max_steps):
                if cur == end:
                    break
                cands = [n for n in graph[cur] if n not in visited]
                if not cands:
                    break
                weights = []
                for n in cands:
                    eta = 1.0 / (ecost(cur, n) + 1e-9)
                    prog = 1.0 / (dist_to_end[n] + 0.01)  # bias towards destination
                    weights.append((pher[(cur, n)] ** alpha) * ((eta * prog) ** beta))
                nxt = random.choices(cands, weights=weights, k=1)[0]
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
            if cur == end:
                cost = sum(ecost(path[i], path[i + 1]) for i in range(len(path) - 1))
                solutions.append((path, cost))
                if cost < best_cost:
                    best_path, best_cost = path, cost

        # evaporation
        for k in pher:
            pher[k] *= (1 - rho)
        # deposit (lower cost => more pheromone)
        for path, cost in solutions:
            if cost <= 0:
                continue
            dep = 1.0 / cost
            for i in range(len(path) - 1):
                pher[(path[i], path[i + 1])] += dep
                pher[(path[i + 1], path[i])] += dep
        # elitist reinforcement
        if best_path:
            dep = 2.0 / best_cost
            for i in range(len(best_path) - 1):
                pher[(best_path[i], best_path[i + 1])] += dep
                pher[(best_path[i + 1], best_path[i])] += dep

    if not best_path:
        return {}

    dist = sum(graph[best_path[i]][best_path[i + 1]][0] for i in range(len(best_path) - 1))
    emis = sum(graph[best_path[i]][best_path[i + 1]][1] for i in range(len(best_path) - 1))
    return {
        "path": best_path,
        "distance_km": round(dist, 3),
        "emission_kgco2": round(emis, 4),
        "polyline": encode_path_to_polyline(best_path),
    }
