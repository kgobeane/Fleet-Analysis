"""
retrain_models.py
-----------------
Run this ONCE on your machine to generate pkl files
that match your exact Python + scikit-learn versions.

Usage:
    cd "Fleet Analysis"
    python retrain_models.py

This replaces all pkl files in the models/ folder.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

Path("models").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — FUEL FORECAST MODEL
# Mirrors fuel_forecast.ipynb exactly
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART 1: Training Fuel Forecast Model (LinearRegression)")
print("=" * 60)

fuel = pd.read_csv("Fleet_Analysis/data/fleet_fuel_weekly.csv")
fuel["week_start"] = pd.to_datetime(fuel["week_start"])

for col in ["vehicle_id", "vehicle_type", "fuel_type"]:
    fuel[col] = fuel[col].str.strip().str.upper()

fuel = fuel.sort_values(["vehicle_id", "week_start"]).reset_index(drop=True)

fuel["gallons"]     = fuel.groupby("vehicle_id")["gallons"].ffill()
fuel["distance_km"] = fuel.groupby("vehicle_id")["distance_km"].ffill()
fuel["gallons"]     = fuel.groupby("vehicle_id")["gallons"].transform(
    lambda x: x.fillna(x.median()))
fuel["distance_km"] = fuel.groupby("vehicle_id")["distance_km"].transform(
    lambda x: x.fillna(x.median()))

# Aggregate to fleet-level weekly
weekly_fuel = fuel.groupby("week_start", as_index=False)["gallons"].sum()
weekly_fuel = weekly_fuel.set_index("week_start")
weekly_fuel.index = pd.DatetimeIndex(weekly_fuel.index)

# Feature engineering
weekly_fuel["year"]    = weekly_fuel.index.year
weekly_fuel["quarter"] = weekly_fuel.index.quarter
weekly_fuel["month"]   = weekly_fuel.index.month
weekly_fuel["week"]    = weekly_fuel.index.isocalendar().week.astype(int)
weekly_fuel["trend"]   = np.arange(len(weekly_fuel))

for lag in [1, 2, 4, 8, 12]:
    weekly_fuel[f"lag_{lag}"] = weekly_fuel["gallons"].shift(lag)

for w in [4, 8, 12]:
    weekly_fuel[f"rolling_mean_{w}"] = weekly_fuel["gallons"].shift(1).rolling(w).mean()
    weekly_fuel[f"rolling_std_{w}"]  = weekly_fuel["gallons"].shift(1).rolling(w).std()

weekly_fuel["expanding_mean"] = weekly_fuel["gallons"].shift(1).expanding().mean()
weekly_fuel["weekly_growth"]  = weekly_fuel["gallons"].pct_change()
weekly_fuel["week_sin"] = np.sin(2 * np.pi * weekly_fuel["week"] / 52)
weekly_fuel["week_cos"] = np.cos(2 * np.pi * weekly_fuel["week"] / 52)
weekly_fuel = weekly_fuel.dropna().copy()

FUEL_FEATURES = [
    "year", "quarter", "month", "week", "trend",
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_12",
    "rolling_std_4",  "rolling_std_8",  "rolling_std_12",
    "expanding_mean", "weekly_growth",
    "week_sin", "week_cos",
]

X = weekly_fuel[FUEL_FEATURES]
y = weekly_fuel["gallons"]

# Train on full dataset (as notebook Cell 78 does)
fuel_model = LinearRegression()
fuel_model.fit(X, y)

# Quick validation
split = int(len(X) * 0.8)
val_model = LinearRegression().fit(X.iloc[:split], y.iloc[:split])
val_preds = val_model.predict(X.iloc[split:])
from sklearn.metrics import r2_score, mean_absolute_error
r2  = r2_score(y.iloc[split:], val_preds)
mae = mean_absolute_error(y.iloc[split:], val_preds)
print(f"Validation  ->  R² = {r2:.4f}   MAE = {mae:.1f} gallons")

# Save — plain list (not dict) so the app loads it without any conversion
joblib.dump(fuel_model,    "models/fuel_forecast_model.pkl")
joblib.dump(FUEL_FEATURES, "models/forecast_features.pkl")
print("Saved: models/fuel_forecast_model.pkl")
print("Saved: models/forecast_features.pkl")
print(f"Features ({len(FUEL_FEATURES)}): {FUEL_FEATURES}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — MAINTENANCE MODEL
# Mirrors maintenance_timeline.ipynb exactly
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 2: Training Maintenance Model (GradientBoostingRegressor)")
print("=" * 60)

maint = pd.read_csv("Fleet_Analysis/data/fleet_maintenance_log.csv")
maint["event_date"]          = pd.to_datetime(maint["event_date"])
maint["vehicle_id"]          = maint["vehicle_id"].astype("string")
maint["vehicle_type"]        = maint["vehicle_type"].astype("category")
maint["service_type"]        = maint["service_type"].astype("category")
maint["odometer_km"]         = pd.to_numeric(maint["odometer_km"],         errors="coerce")
maint["cost_usd"]            = pd.to_numeric(maint["cost_usd"],            errors="coerce")
maint["days_in_shop"]        = pd.to_numeric(maint["days_in_shop"],        errors="coerce")
maint["next_service_due_km"] = pd.to_numeric(maint["next_service_due_km"], errors="coerce")

maint = maint.sort_values(["vehicle_id", "event_date"]).reset_index(drop=True)
maint = maint.drop_duplicates()
maint = maint[
    (maint["odometer_km"] >= 0) &
    (maint["cost_usd"] >= 0) &
    (maint["days_in_shop"] >= 0) &
    (maint["next_service_due_km"] > maint["odometer_km"])
].reset_index(drop=True)

maint["maintenance_count"]           = maint.groupby("vehicle_id").cumcount()
maint["previous_date"]               = maint.groupby("vehicle_id")["event_date"].shift(1)
maint["previous_odometer"]           = maint.groupby("vehicle_id")["odometer_km"].shift(1)
maint["previous_cost"]               = maint.groupby("vehicle_id")["cost_usd"].shift(1)
maint["days_since_last_service"]     = (maint["event_date"] - maint["previous_date"]).dt.days
maint["distance_since_last_service"] = maint["odometer_km"] - maint["previous_odometer"]
maint["cumulative_cost"]             = maint.groupby("vehicle_id")["cost_usd"].cumsum()
maint["rolling_cost_3"]              = maint.groupby("vehicle_id")["cost_usd"].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
maint["avg_service_distance"]        = maint.groupby("vehicle_id")["distance_since_last_service"].transform(
    lambda x: x.expanding(min_periods=1).mean())

maint["year"]        = maint["event_date"].dt.year
maint["month"]       = maint["event_date"].dt.month
maint["quarter"]     = maint["event_date"].dt.quarter
maint["day_of_week"] = maint["event_date"].dt.dayofweek
maint["month_sin"]   = np.sin(2 * np.pi * maint["month"] / 12)
maint["month_cos"]   = np.cos(2 * np.pi * maint["month"] / 12)

maint = maint.fillna({
    "days_since_last_service": 0,
    "distance_since_last_service": 0,
    "previous_cost": 0,
    "previous_odometer": 0,
    "avg_service_distance": 0,
})

encoded = pd.get_dummies(maint, columns=["vehicle_type", "service_type"], drop_first=True)
bool_cols = encoded.select_dtypes(include="bool").columns
encoded[bool_cols] = encoded[bool_cols].astype(int)

exclude_cols = ["vehicle_id", "date", "event_date", "previous_date", "next_service_due_km"]
MAINT_FEATURES = [c for c in encoded.columns if c not in exclude_cols]

X_m = encoded[MAINT_FEATURES]
y_m = encoded["next_service_due_km"]

split_m = int(len(X_m) * 0.8)
X_tr, X_te = X_m.iloc[:split_m], X_m.iloc[split_m:]
y_tr, y_te = y_m.iloc[:split_m], y_m.iloc[split_m:]

maint_model = GradientBoostingRegressor(random_state=42)
maint_model.fit(X_tr, y_tr)

val_preds_m = maint_model.predict(X_te)
r2_m  = r2_score(y_te, val_preds_m)
mae_m = mean_absolute_error(y_te, val_preds_m)
print(f"Validation  ->  R² = {r2_m:.4f}   MAE = {mae_m:,.0f} km")

# Retrain on full data for deployment
maint_model.fit(X_m, y_m)

joblib.dump(maint_model,    "models/maintenance_model.pkl")
joblib.dump(MAINT_FEATURES, "models/maintenance_features.pkl")
print("Saved: models/maintenance_model.pkl")
print("Saved: models/maintenance_features.pkl")
print(f"Features ({len(MAINT_FEATURES)}): {MAINT_FEATURES[:5]} ...")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY — reload and check types immediately
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("VERIFICATION — reloading all pkl files")
print("=" * 60)

fm  = joblib.load("models/fuel_forecast_model.pkl")
ff  = joblib.load("models/forecast_features.pkl")
mm  = joblib.load("models/maintenance_model.pkl")
mf  = joblib.load("models/maintenance_features.pkl")

print(f"fuel_forecast_model.pkl  : {type(fm).__name__}")
print(f"forecast_features.pkl    : {type(ff).__name__}  len={len(ff)}")
print(f"maintenance_model.pkl    : {type(mm).__name__}")
print(f"maintenance_features.pkl : {type(mf).__name__}  len={len(mf)}")
print()
print("=" * 60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print("Now run:  streamlit run fleet_analysis.py")
print("=" * 60)
