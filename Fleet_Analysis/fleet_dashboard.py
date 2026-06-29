"""
fleet_dashboard.py — Executive Dashboard tab
Uses the same raw data pipeline as the fuel notebook.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


@st.cache_data
def _load_data():
    fuel = pd.read_csv("Fleet_Analysis/data/fleet_fuel_weekly.csv")
    fuel["week_start"] = pd.to_datetime(fuel["week_start"])
    for col in ["vehicle_id","vehicle_type","fuel_type"]:
        fuel[col] = fuel[col].str.strip().str.upper()
    fuel = fuel.sort_values(["vehicle_id","week_start"]).reset_index(drop=True)
    fuel["gallons"]     = fuel.groupby("vehicle_id")["gallons"].ffill()
    fuel["distance_km"] = fuel.groupby("vehicle_id")["distance_km"].ffill()
    fuel["km_per_gallon"] = fuel["distance_km"] / fuel["gallons"].replace(0, np.nan)

    maint = pd.read_csv("data/fleet_maintenance_log.csv")
    maint["event_date"] = pd.to_datetime(maint["event_date"])
    return fuel, maint


def executive_dashboard():
    """Renders the Executive Dashboard tab."""
    st.title("📊 Fleet Executive Dashboard")
    st.caption("2-year historical telemetry · 20 vehicles · Jan 2022 – Dec 2023")

    fuel, maint = _load_data()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Fuel Consumed",    f"{fuel['gallons'].sum():,.0f} gal")
    k2.metric("Fleet Size",             f"{fuel['vehicle_id'].nunique()} vehicles")
    k3.metric("Total Maintenance Cost", f"${maint['cost_usd'].sum():,.0f}")
    k4.metric("Avg Efficiency",         f"{fuel['km_per_gallon'].mean():.1f} km/gal")

    st.divider()

    # Weekly trend
    st.subheader("Weekly Fleet Fuel Consumption")
    weekly = fuel.groupby("week_start")["gallons"].sum().reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=weekly["week_start"], y=weekly["gallons"], mode="lines",
        line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.07)",
    ))
    fig1.update_layout(template="plotly_white", height=280,
                       xaxis=dict(tickformat="%b '%y"),
                       yaxis=dict(title="Gallons / week", tickformat=",.0f"),
                       margin=dict(t=20,b=40,l=60,r=20))
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    TYPE_COLORS = {"HEAVY TRUCK":"#D85A30","LIGHT TRUCK":"#378ADD",
                   "VAN":"#1D9E75","SUV":"#BA7517"}

    with col1:
        st.subheader("Fuel by Vehicle Type")
        vt = fuel.groupby("vehicle_type")["gallons"].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(vt, x="vehicle_type", y="gallons", color="vehicle_type",
                      color_discrete_map=TYPE_COLORS, template="plotly_white",
                      labels={"vehicle_type":"","gallons":"Total gallons"})
        fig2.update_layout(showlegend=False, height=300, yaxis_tickformat=",.0f")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Diesel vs Gasoline Split")
        ft = fuel.groupby("fuel_type")["gallons"].sum().reset_index()
        fig3 = go.Figure(go.Pie(
            labels=ft["fuel_type"], values=ft["gallons"], hole=0.45,
            marker_colors=["#D85A30","#378ADD"], textinfo="label+percent"))
        fig3.update_layout(template="plotly_white", height=300,
                           margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Maintenance Events by Type")
        svc = maint.groupby("service_type").size().reset_index(name="count")
        fig4 = px.bar(svc, x="service_type", y="count", color="service_type",
                      template="plotly_white",
                      color_discrete_sequence=["#378ADD","#BA7517","#D85A30","#1D9E75"],
                      labels={"service_type":"","count":"Events"})
        fig4.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("Top 10 Fuel Consumers")
        top10 = fuel.groupby("vehicle_id")["gallons"].sum()\
                    .sort_values(ascending=False).head(10).reset_index()
        fig5 = px.bar(top10, x="vehicle_id", y="gallons",
                      color="gallons", color_continuous_scale="Reds",
                      template="plotly_white",
                      labels={"vehicle_id":"","gallons":"Total gallons"})
        fig5.update_layout(showlegend=False, height=300,
                           yaxis_tickformat=",.0f", coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)
