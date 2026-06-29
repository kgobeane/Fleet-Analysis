"""
fleet_analysis.py — Main Streamlit entry point
Run with: streamlit run fleet_analysis.py
"""
import streamlit as st

from fleet_dashboard import executive_dashboard
from fuel_forecasting_run import fuel_forecasting
from predictive_maintenance import predictive_maintenance

st.set_page_config(
    page_title="Fleet Analytics Platform",
    page_icon="🚛",
    layout="wide",
)

st.title("🚛 Fleet Analytics Platform")

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Dashboard",
    "⛽ Fuel Forecasting",
    "🔧 Predictive Maintenance",
])

with tab1:
    executive_dashboard()

with tab2:
    fuel_forecasting()

with tab3:
    predictive_maintenance()
