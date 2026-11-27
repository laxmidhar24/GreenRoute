import csv
import os
import subprocess
import sys
from datetime import datetime

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from utils import (aco_optimize, build_graph_from_routes, compute_elevation_gain,
                   estimate_emission, get_aqi)

# ===============================
# 1. SETUP
# ===============================
load_dotenv()
MAPS_API_KEY = os.getenv("MAPS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # optional

app = Flask(__name__)

RF_PATH, LGB_PATH = "emission_rf.pkl", "emission_lgbm.pkl"
FEATURES = ["distance_km", "elevation_gain_m", "congestion_index",
            "base_emission_per_km", "fuel_encoded"]
FUEL_MAP = {"petrol": 0, "diesel": 1, "ev": 2, "bike": 3, "bus": 4}
BASE_EMISSION = {"petrol": 0.192, "diesel": 0.171, "bike": 0.103, "bus": 0.27, "ev": 0.07}  # kg CO2/km


def load_models():
    rf = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None
    lgb = joblib.load(LGB_PATH) if os.path.exists(LGB_PATH) else None
    print(f"Models loaded - RF: {rf is not None}, LightGBM: {lgb is not None}")
    return rf, lgb


rf_model, lgb_model = load_models()


def fetch_directions(origin, destination, api_key):
    r = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params={"origin": origin, "destination": destination, "mode": "driving",
                "alternatives": "true", "departure_time": "now",
                "traffic_model": "best_guess", "key": api_key},
        timeout=20,
    )
    return r.json()


def log_and_maybe_retrain(row):
    global rf_model, lgb_model
    try:
        with open("training_data.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        with open("training_data.csv") as f:
            n = sum(1 for _ in f)
        if n % 500 == 0:
            print(f"Retraining after {n} samples...")
            subprocess.run([sys.executable, "train_model.py"], check=False)
            rf_model, lgb_model = load_models()
    except Exception as e:
        print("Could not log/retrain:", e)


@app.route("/")
def index():
    return render_template("index.html", MAPS_API_KEY=MAPS_API_KEY)


@app.route("/compare_routes", methods=["GET"])
def compare_routes():
    origin = request.args.get("origin")
    destination = request.args.get("destination")
    vehicle = (request.args.get("vehicle") or "car").lower()
    fuel = (request.args.get("fuel") or "petrol").lower()
    # bike / bus override the fuel dropdown
    key = vehicle if vehicle in ("bike", "bus") else fuel
    base = BASE_EMISSION.get(key, 0.192)
    fuel_encoded = FUEL_MAP.get(key, 0)

    print(f"\nRequest: {origin} -> {destination}, fuel={fuel}, vehicle={vehicle}")
    if not origin or not destination:
        return jsonify({"error": "origin and destination required", "routes": [], "aco_route": {}})

    directions = fetch_directions(origin, destination, MAPS_API_KEY)
    if directions.get("status") != "OK":
        print("Directions API error:", directions)
        return jsonify({"error": directions.get("error_message", directions.get("status", "Directions failed")),
                        "routes": [], "aco_route": {}})

    routes_out, route_objs, log_rows = [], [], []

    for idx, route in enumerate(directions.get("routes", [])):
        try:
            leg = route["legs"][0]
            distance_km = leg["distance"]["value"] / 1000.0
            duration_s = leg["duration"]["value"]
            traffic_s = leg.get("duration_in_traffic", {}).get("value", duration_s)
            congestion = (traffic_s / duration_s) - 1 if duration_s > 0 else 0.0
            poly = route["overview_polyline"]["points"]
            elev = compute_elevation_gain(poly, MAPS_API_KEY)
            aqi, aqi_cat = get_aqi(poly, OPENWEATHER_API_KEY)

            rule_emission = estimate_emission(distance_km, elev, base, congestion)
            X = pd.DataFrame([[distance_km, elev, congestion, base, fuel_encoded]], columns=FEATURES)

            try:
                rf_em = float(rf_model.predict(X)[0]) if rf_model else rule_emission
            except Exception as e:
                print("RF predict error:", e)
                rf_em = rule_emission
            try:
                lgb_em = float(lgb_model.predict(X)[0]) if lgb_model else rule_emission
            except Exception as e:
                print("LGBM predict error:", e)
                lgb_em = rule_emission

            routes_out.append({
                "route_index": idx + 1,
                "distance_km": round(distance_km, 3),
                "duration_min": round(duration_s / 60.0, 1),
                "duration_traffic_min": round(traffic_s / 60.0, 1),
                "elevation_gain_m": round(elev, 1),
                "congestion_index": round(congestion, 3),
                "rf_emission": round(rf_em, 4),
                "lgbm_emission": round(lgb_em, 4),
                "aqi": aqi,
                "aqi_category": aqi_cat,
                "polyline": poly,
            })
            route_objs.append({"polyline": poly, "emission_kg": max(lgb_em, 1e-6)})
            log_rows.append([datetime.now().isoformat(), distance_km, elev, congestion,
                             key, base, rule_emission, aqi if aqi is not None else ""])
        except Exception as e:
            print("Error processing route:", e)
            continue

    if not routes_out:
        return jsonify({"error": "No routes could be processed", "routes": [], "aco_route": {}})

    best_idx = min(range(len(routes_out)), key=lambda i: routes_out[i]["lgbm_emission"])
    best = routes_out[best_idx]

    # ---- ACO over the graph made from all alternatives ----
    aco = {}
    try:
        graph, start, end = build_graph_from_routes(route_objs)
        print(f"Running ACO on {len(graph)} nodes...")
        aco = aco_optimize(graph, start, end)
    except Exception as e:
        print("ACO error:", e)

    if not aco:  # fall back to best ML route
        print("ACO returned no route - using best ML route")
        aco = {"distance_km": best["distance_km"], "emission_kgco2": best["lgbm_emission"],
               "polyline": best["polyline"]}
        aco_better = False
    else:
        aco_better = aco["emission_kgco2"] < best["lgbm_emission"] * 0.999
    aco.pop("path", None)  # tuples not needed by frontend

    print(f"Processed {len(routes_out)} routes. Best (LGBM)={best_idx + 1}; ACO better: {aco_better}")

    for row in log_rows:
        log_and_maybe_retrain(row)

    return jsonify({
        "routes": routes_out,
        "best_route_index": best_idx + 1,
        "aco_route": aco,
        "aco_better_than_lgbm": aco_better,
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # reloader off: retraining rewrites .pkl files
