"""
fuel_forecasting_run.py
Matches fuel_forecast.ipynb exactly:
- LinearRegression on fleet-level weekly aggregated gallons
- Same 20 features as notebook
- recursive_forecast() mirrors notebook Cell 81
"""
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


@st.cache_data
def _load_raw():
    fuel = pd.read_csv("data/fleet_fuel_weekly.csv")
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
    return fuel


@st.cache_data
def _build_weekly(fuel):
    wf = fuel.groupby("week_start", as_index=False)["gallons"].sum()
    wf = wf.set_index("week_start")
    wf.index = pd.DatetimeIndex(wf.index)
    wf["year"]    = wf.index.year
    wf["quarter"] = wf.index.quarter
    wf["month"]   = wf.index.month
    wf["week"]    = wf.index.isocalendar().week.astype(int)
    wf["trend"]   = np.arange(len(wf))
    for lag in [1, 2, 4, 8, 12]:
        wf[f"lag_{lag}"] = wf["gallons"].shift(lag)
    for w in [4, 8, 12]:
        wf[f"rolling_mean_{w}"] = wf["gallons"].shift(1).rolling(w).mean()
        wf[f"rolling_std_{w}"]  = wf["gallons"].shift(1).rolling(w).std()
    wf["expanding_mean"] = wf["gallons"].shift(1).expanding().mean()
    wf["weekly_growth"]  = wf["gallons"].pct_change()
    wf["week_sin"] = np.sin(2 * np.pi * wf["week"] / 52)
    wf["week_cos"] = np.cos(2 * np.pi * wf["week"] / 52)
    return wf.dropna().copy()


@st.cache_resource
def _load_model():
    model    = joblib.load("models/fuel_forecast_model.pkl")
    features = joblib.load("models/forecast_features.pkl")
    # Normalise: notebook saved features as a plain list
    if isinstance(features, dict):
        features = features.get("features", list(features.keys()))
    return model, features


def _recursive_forecast(history_df, model, features, periods):
    """Mirrors notebook Cell 81 exactly."""
    history = history_df.copy()
    rows = []
    for _ in range(periods):
        next_date = history.index[-1] + pd.Timedelta(weeks=1)
        row = {}
        row["year"]    = next_date.year
        row["quarter"] = next_date.quarter
        row["month"]   = next_date.month
        row["week"]    = int(next_date.isocalendar()[1])
        row["trend"]   = int(history["trend"].iloc[-1]) + 1
        row["lag_1"]   = history["gallons"].iloc[-1]
        row["lag_2"]   = history["gallons"].iloc[-2]  if len(history) >= 2  else history["gallons"].iloc[-1]
        row["lag_4"]   = history["gallons"].iloc[-4]  if len(history) >= 4  else history["gallons"].iloc[-1]
        row["lag_8"]   = history["gallons"].iloc[-8]  if len(history) >= 8  else history["gallons"].iloc[-1]
        row["lag_12"]  = history["gallons"].iloc[-12] if len(history) >= 12 else history["gallons"].iloc[-1]
        row["rolling_mean_4"]  = history["gallons"].iloc[-4:].mean()
        row["rolling_mean_8"]  = history["gallons"].iloc[-8:].mean()
        row["rolling_mean_12"] = history["gallons"].iloc[-12:].mean()
        row["rolling_std_4"]   = history["gallons"].iloc[-4:].std()
        row["rolling_std_8"]   = history["gallons"].iloc[-8:].std()
        row["rolling_std_12"]  = history["gallons"].iloc[-12:].std()
        row["expanding_mean"]  = history["gallons"].mean()
        prev = history["gallons"].iloc[-1]
        prev2 = history["gallons"].iloc[-2] if len(history) >= 2 else prev
        row["weekly_growth"] = (prev - prev2) / prev2 if prev2 != 0 else 0
        row["week_sin"] = np.sin(2 * np.pi * row["week"] / 52)
        row["week_cos"] = np.cos(2 * np.pi * row["week"] / 52)

        X_row = pd.DataFrame([row])[features]
        pred  = float(model.predict(X_row)[0])

        # Append predicted row to history so next iteration has fresh lags
        new_row = pd.DataFrame(
            {**row, "gallons": pred},
            index=[next_date]
        )
        history = pd.concat([history, new_row])

        rows.append({"week_start": next_date, "forecast_gallons": round(pred, 2)})

    return pd.DataFrame(rows)


def fuel_forecasting():
    """Renders the Fuel Forecasting tab. Imported by fleet_analysis.py."""

    with st.sidebar:
        st.header("⚙️ Forecast Settings")
        n_weeks = st.selectbox("Forecast horizon (weeks)", [12, 26, 52], index=0)
        st.info("Linear Regression model trained on fleet-level weekly aggregated fuel demand.")

    st.title("⛽ Fleet Fuel Demand Forecasting")
    st.caption("Linear Regression · Fleet weekly aggregated · 2022–2023 training data · R² = 0.999")

    with st.spinner("Loading data and model…"):
        fuel_raw  = _load_raw()
        weekly_df = _build_weekly(fuel_raw)
        model, features = _load_model()

    with st.spinner(f"Generating {n_weeks}-week recursive forecast…"):
        forecast_df = _recursive_forecast(weekly_df, model, features, n_weeks)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    hist_mean = weekly_df["gallons"].mean()
    fc_mean   = forecast_df["forecast_gallons"].mean()
    fc_total  = forecast_df["forecast_gallons"].sum()
    pct_diff  = (fc_mean - hist_mean) / hist_mean * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Horizon",    f"{n_weeks} weeks")
    c2.metric("Total Forecast Fuel", f"{fc_total:,.0f} gal")
    c3.metric("Avg Weekly Forecast", f"{fc_mean:,.0f} gal",
              delta=f"{pct_diff:+.1f}% vs historical")
    c4.metric("Historical Avg/Week", f"{hist_mean:,.0f} gal")

    st.divider()

    # ── Main chart ────────────────────────────────────────────────────────────
    st.subheader("📈 Historical + Forecast")

    fc_upper = forecast_df["forecast_gallons"] * 1.05
    fc_lower = forecast_df["forecast_gallons"] * 0.95

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_df.index, y=weekly_df["gallons"],
        name="Historical", mode="lines",
        line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["week_start"], forecast_df["week_start"][::-1]]),
        y=pd.concat([fc_upper, fc_lower[::-1]]),
        fill="toself", fillcolor="rgba(216,90,48,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="±5% band",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["week_start"], y=forecast_df["forecast_gallons"],
        name="Forecast", mode="lines+markers",
        line=dict(color="#D85A30", width=2.5, dash="dash"),
        marker=dict(size=5),
    ))
    fig.add_vline(x=weekly_df.index.max(), line_dash="dot", line_color="#888",
                  annotation_text="Forecast start", annotation_position="top right")
    fig.update_layout(
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Week", tickformat="%b '%y"),
        yaxis=dict(title="Gallons / week", tickformat=",.0f"),
        margin=dict(t=60, b=40, l=60, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Forecast table ────────────────────────────────────────────────────────
    st.subheader("📋 Forecast Table")
    disp = forecast_df.copy()
    disp["week_start"]       = disp["week_start"].dt.strftime("%Y-%m-%d")
    disp["forecast_gallons"] = disp["forecast_gallons"].map("{:,.1f}".format)
    st.dataframe(disp.rename(columns={"week_start":"Week","forecast_gallons":"Forecast Gallons"}),
                 use_container_width=True, height=320)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download forecast CSV", csv,
                       f"fuel_forecast_{n_weeks}wk.csv", "text/csv")

    with st.expander("ℹ️ Model details"):
        st.markdown(f"""
| Item | Value |
|---|---|
| Algorithm | Linear Regression (trained on full dataset) |
| Input | Fleet-level weekly aggregated gallons |
| Features | {len(features)} (calendar, lags, rolling stats, cyclical) |
| Forecast method | Recursive — each week feeds next week's lags |
| Test R² | 0.9993 |
| Confidence band | ±5% |
""")
