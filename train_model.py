import os
import sys

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

if not os.path.exists("training_data.csv"):
    print("training_data.csv not found! Run: python generate_seed_data.py")
    sys.exit(1)

df = pd.read_csv("training_data.csv", header=None)
df.columns = ["timestamp", "distance_km", "elevation_gain_m", "congestion_index",
              "fuel_type", "base_emission_per_km", "emission", "aqi"]
df = df.dropna(subset=["distance_km", "elevation_gain_m", "congestion_index", "emission"])
if len(df) < 50:
    print(f"Only {len(df)} rows - need more data. Run generate_seed_data.py")
    sys.exit(1)

fuel_map = {"petrol": 0, "diesel": 1, "ev": 2, "bike": 3, "bus": 4}
df["fuel_encoded"] = df["fuel_type"].astype(str).str.lower().map(fuel_map).fillna(0)

FEATURES = ["distance_km", "elevation_gain_m", "congestion_index", "base_emission_per_km", "fuel_encoded"]
X, y = df[FEATURES], df["emission"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1).fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2, rf_mae = r2_score(y_test, rf_pred), mean_absolute_error(y_test, rf_pred)

lgb = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                    verbose=-1).fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
lgb_r2, lgb_mae = r2_score(y_test, lgb_pred), mean_absolute_error(y_test, lgb_pred)

joblib.dump(rf, "emission_rf.pkl")
joblib.dump(lgb, "emission_lgbm.pkl")

print("\nTraining Summary:")
print(f"Random Forest -> R2 = {rf_r2:.3f}, MAE = {rf_mae:.3f}")
print(f"LightGBM      -> R2 = {lgb_r2:.3f}, MAE = {lgb_mae:.3f}")
print("Models saved. Ready for GreenRoute.")
