"""
predictive_maintenance.py
Matches maintenance_timeline.ipynb exactly:
- GradientBoostingRegressor on next_service_due_km
- Same feature engineering as notebook Cells 5-7
- Same priority thresholds: <=1000 Immediate, <=3000 Due Soon
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


@st.cache_resource
def _load_model():
    model    = joblib.load("Fleet_Analysis/models/maintenance_model.pkl")
    features = joblib.load("Fleet_Analysis/models/maintenance_features.pkl")
    # Normalise: might be list or dict
    if isinstance(features, dict):
        features = features.get("feature_columns", list(features.keys()))
    return model, features


@st.cache_data
def _load_and_engineer():
    """Full pipeline from notebook Cells 2-7."""
    df = pd.read_csv("Fleet_Analysis/data/fleet_maintenance_log.csv")
    df["event_date"]          = pd.to_datetime(df["event_date"])
    df["vehicle_id"]          = df["vehicle_id"].astype("string")
    df["vehicle_type"]        = df["vehicle_type"].astype("category")
    df["service_type"]        = df["service_type"].astype("category")
    df["odometer_km"]         = pd.to_numeric(df["odometer_km"],         errors="coerce")
    df["cost_usd"]            = pd.to_numeric(df["cost_usd"],            errors="coerce")
    df["days_in_shop"]        = pd.to_numeric(df["days_in_shop"],        errors="coerce")
    df["next_service_due_km"] = pd.to_numeric(df["next_service_due_km"], errors="coerce")

    df = df.sort_values(["vehicle_id","event_date"]).reset_index(drop=True)
    df = df.drop_duplicates()
    df = df[
        (df["odometer_km"] >= 0) & (df["cost_usd"] >= 0) &
        (df["days_in_shop"] >= 0) & (df["next_service_due_km"] > df["odometer_km"])
    ].reset_index(drop=True)

    df["maintenance_count"]             = df.groupby("vehicle_id").cumcount()
    df["previous_date"]                 = df.groupby("vehicle_id")["event_date"].shift(1)
    df["previous_odometer"]             = df.groupby("vehicle_id")["odometer_km"].shift(1)
    df["previous_cost"]                 = df.groupby("vehicle_id")["cost_usd"].shift(1)
    df["days_since_last_service"]       = (df["event_date"] - df["previous_date"]).dt.days
    df["distance_since_last_service"]   = df["odometer_km"] - df["previous_odometer"]
    df["cumulative_cost"]               = df.groupby("vehicle_id")["cost_usd"].cumsum()
    df["rolling_cost_3"]                = df.groupby("vehicle_id")["cost_usd"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df["avg_service_distance"]          = df.groupby("vehicle_id")["distance_since_last_service"].transform(
        lambda x: x.expanding(min_periods=1).mean())
    df["year"]        = df["event_date"].dt.year
    df["month"]       = df["event_date"].dt.month
    df["quarter"]     = df["event_date"].dt.quarter
    df["day_of_week"] = df["event_date"].dt.dayofweek
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    df = df.fillna({"days_since_last_service": 0, "distance_since_last_service": 0,
                    "previous_cost": 0, "previous_odometer": 0, "avg_service_distance": 0})

    encoded = pd.get_dummies(df, columns=["vehicle_type","service_type"], drop_first=True)
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)

    return df, encoded


def _assign_priority(km):
    if km <= 1000:  return "🔴 Immediate Service Required"
    if km <= 3000:  return "🟡 Service Due Soon"
    return "🟢 Low Priority"


def predictive_maintenance():
    """Renders the Predictive Maintenance tab. Imported by fleet_analysis.py."""

    st.title("🔧 Fleet Predictive Maintenance")
    st.markdown("Predict next service due mileage and classify vehicles by urgency.")

    with st.spinner("Loading model and engineering features…"):
        model, feature_columns = _load_model()
        raw_df, encoded_df     = _load_and_engineer()

    # Align columns to training feature set
    for col in feature_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    X = encoded_df[feature_columns]

    # Predict on ALL records
    encoded_df["predicted_next_service_km"] = model.predict(X)
    encoded_df["km_until_service"] = (
        encoded_df["predicted_next_service_km"] - encoded_df["odometer_km"]
    )
    encoded_df["Maintenance_Status"] = encoded_df["km_until_service"].apply(_assign_priority)

    # Latest record per vehicle (most current status)
    latest = (
        encoded_df.sort_values("event_date")
        .groupby("vehicle_id")
        .last()
        .reset_index()
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vehicles",  latest["vehicle_id"].nunique())
    c2.metric("🔴 Immediate",    (latest["Maintenance_Status"]=="🔴 Immediate Service Required").sum())
    c3.metric("🟡 Due Soon",     (latest["Maintenance_Status"]=="🟡 Service Due Soon").sum())
    c4.metric("🟢 Low Priority", (latest["Maintenance_Status"]=="🟢 Low Priority").sum())

    st.divider()

    COLOR_MAP = {
        "🔴 Immediate Service Required": "#EF553B",
        "🟡 Service Due Soon":           "#FECB52",
        "🟢 Low Priority":               "#00CC96",
    }

    # ── Chart 1: Priority distribution ───────────────────────────────────────
    st.subheader("Fleet Maintenance Urgency Breakdown")
    status_counts = latest["Maintenance_Status"].value_counts().reset_index()
    status_counts.columns = ["Status","Count"]
    fig1 = px.bar(status_counts, x="Count", y="Status", orientation="h",
                  color="Status", color_discrete_map=COLOR_MAP,
                  text_auto=True, template="plotly_white")
    fig1.update_layout(showlegend=False, height=300,
                       xaxis_title="Number of Vehicles", yaxis_title="")
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ── Chart 2: Vehicle trajectory (from notebook Cell 11) ──────────────────
    st.subheader("Vehicle Maintenance Horizons")
    latest_sorted = latest.sort_values("km_until_service")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=latest_sorted["vehicle_id"], y=latest_sorted["odometer_km"],
        name="Current Odometer (km)", marker_color="#1f77b4",
    ))
    fig2.add_trace(go.Bar(
        x=latest_sorted["vehicle_id"], y=latest_sorted["km_until_service"],
        name="KM Until Next Service", marker_color="#7f7f7f", opacity=0.6,
        hovertext=latest_sorted["Maintenance_Status"],
    ))
    fig2.update_layout(
        barmode="stack", template="plotly_white", height=480,
        xaxis_title="Vehicle", yaxis_title="Odometer / Distance (km)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Maintenance schedule table ────────────────────────────────────────────
    st.subheader("📋 Vehicle Maintenance Schedule")
    display_cols = ["vehicle_id","vehicle_type","service_type",
                    "odometer_km","predicted_next_service_km",
                    "km_until_service","Maintenance_Status"]
    # vehicle_type / service_type may have been dropped by get_dummies — pull from raw
    disp = latest[["vehicle_id","odometer_km","predicted_next_service_km",
                   "km_until_service","Maintenance_Status"]].copy()
    disp["predicted_next_service_km"] = disp["predicted_next_service_km"].round(0).astype(int)
    disp["km_until_service"]          = disp["km_until_service"].round(0).astype(int)
    disp = disp.sort_values("km_until_service")
    st.dataframe(disp, use_container_width=True, height=400)

    st.divider()

    # ── Cost by service type ──────────────────────────────────────────────────
    st.subheader("Maintenance Cost by Service Type")
    svc_cost = raw_df.groupby("service_type")["cost_usd"].sum().reset_index()
    fig3 = px.bar(svc_cost, x="service_type", y="cost_usd", color="service_type",
                  template="plotly_white",
                  color_discrete_sequence=["#D85A30","#378ADD","#1D9E75","#BA7517"],
                  labels={"cost_usd":"Total Cost (USD)","service_type":""})
    fig3.update_layout(showlegend=False, height=300,
                       yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Maintenance Schedule", csv,
                       "maintenance_schedule.csv", "text/csv")
    st.success("✅ Predictive Maintenance Analysis Complete")
