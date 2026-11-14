import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path

st.set_page_config(page_title="IFRS 9 ECL Engine", layout="wide")
st.title("🏦 IFRS 9 AI Stress Testing Engine")
st.markdown("Simulate macroeconomic shocks and observe the impact on Expected Credit Loss (ECL) and IFRS 9 Staging.")

@st.cache_data
def load_data():
    BASE = Path(r"C:\CreditRiskCourse\01_microcredit_portfolio")
    perf = pd.read_parquet(BASE / "data/raw/parquet/loan_monthly_performance.parquet")
    master = pd.read_parquet(BASE / "data/raw/parquet/loan_master.parquet")
    
    df = perf[perf["month_id"] == 36].copy()
    df = df.merge(master[["loan_id", "sector", "collateral_type", "true_annual_pd"]], on="loan_id")
    
    df["ead"] = df["outstanding_balance"]
    df["base_pd"] = df["true_annual_pd"]
    
    lgd_map = {"None": 0.85, "Movable assets": 0.55, "Property": 0.25, 
               "Cash/Receivables": 0.35, "DFI guarantee": 0.10, "Group guarantee": 0.60}
    df["base_lgd"] = df["collateral_type"].map(lgd_map)
    
    df["stage"] = 1
    df.loc[(df["dpd_bucket"] >= 1) | (df["restructure_flag"] == 1), "stage"] = 2
    df.loc[df["dpd_bucket"] == 4, "stage"] = 3
    
    return df

df_raw = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🤖 AI Agent Input")
user_prompt = st.sidebar.text_area(
    "Describe the stress scenario:",
    "A severe systemic shock similar to COVID-19. Unemployment rises by 8.0% and GDP drops by 6.0%."
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Portfolio Slicers")
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector:", 
    options=df_raw["sector"].unique(), 
    default=df_raw["sector"].unique()
)

if st.sidebar.button("Run ECL Simulation"):
    # 1. FILTER DATA
    df = df_raw[df_raw["sector"].isin(selected_sectors)].copy()
    
    # 2. NLP PARSING
    delta_unemp = 4.5
    unemp_match = re.search(r'unemployment.*?(\d+\.?\d*)', user_prompt, re.IGNORECASE)
    if unemp_match: delta_unemp = float(unemp_match.group(1))

    delta_gdp = -2.0
    gdp_match = re.search(r'gdp.*?(\d+\.?\d*)', user_prompt, re.IGNORECASE)
    if gdp_match:
        val = float(gdp_match.group(1))
        if any(word in user_prompt.lower() for word in ['drop', 'fall', 'decline', 'decreas', 'shrink', 'contract', 'down']):
            delta_gdp = -abs(val)
        else:
            delta_gdp = val
            
    st.sidebar.success(f"Parsed:\nUnemployment Δ: +{delta_unemp}%\nGDP Δ: {delta_gdp}%")
    
    # 3. STRESS CALCULATIONS
    macro_mult = np.exp(0.08 * delta_unemp) * np.exp(-0.15 * delta_gdp)
    
    sector_mult = {"Construction": 1.5, "Agriculture": 1.4, "Trade/Retail": 1.3, 
                   "Transport": 1.2, "Manufacturing": 1.2, "Tourism": 1.3, 
                   "Services": 1.1, "ICT": 0.9, "Agro-processing": 1.1}
    
    # Base ECL
    df["base_ecl"] = df.apply(lambda r: r["base_pd"] * r["base_lgd"] * r["ead"] * (1 if r["stage"]==1 else 3), axis=1)
    
    # Stressed ECL
    df["stressed_pd"] = (df["base_pd"] * macro_mult * df["sector"].map(sector_mult).fillna(1.2)).clip(0, 1)
    df["stressed_lgd"] = (df["base_lgd"] * 1.10).clip(0, 1) 
    
    rel_inc = (df["stressed_pd"] - df["base_pd"]) / df["base_pd"]
    abs_inc = df["stressed_pd"] - df["base_pd"]
    sicr_mask = (df["stage"] == 1) & (rel_inc > 0.50) & (abs_inc > 0.01)
    df["final_stage"] = np.where(sicr_mask, 2, df["stage"])
    
    def calc(row):
        if row["final_stage"] == 1: return row["stressed_pd"] * row["stressed_lgd"] * row["ead"] * 1
        elif row["final_stage"] == 2: return row["stressed_pd"] * row["stressed_lgd"] * row["ead"] * 3
        else: return 1.0 * row["stressed_lgd"] * row["ead"]
        
    df["ecl_stressed"] = df.apply(calc, axis=1)
    
       # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Portfolio EAD", f"${df['ead'].sum():,.0f}")
    
    col2.metric("Base ECL", f"${df['base_ecl'].sum():,.0f}")
    
    ecl_delta = df['ecl_stressed'].sum() - df['base_ecl'].sum()
    col3.metric("Stressed ECL", f"${df['ecl_stressed'].sum():,.0f}", 
                delta=f"${ecl_delta:,.0f} vs Base", delta_color="inverse")
                
    col4.metric("SICR Migrations (Stage 1→2)", f"{sicr_mask.sum():,} loans", 
                delta=f"{(sicr_mask.sum()/len(df))*100:.1f}% of Filter", delta_color="inverse")

    st.markdown("---")
    
    # --- ROW 1: CHARTS ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Chart 1: Base vs Stressed Total
        chart_data = pd.DataFrame({
            "Scenario": ["Base ECL", "Stressed ECL"],
            "ECL": [df['base_ecl'].sum(), df['ecl_stressed'].sum()]
        })
        fig1 = px.bar(chart_data, x="Scenario", y="ECL", color="Scenario",
                     title="Total ECL: Base vs Stressed",
                     color_discrete_sequence=["#4C72B0", "#C44E52"])
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        # Chart 2: ECL by Stage (Base vs Stressed)
        stage_data = df.groupby("final_stage")[["base_ecl", "ecl_stressed"]].sum().reset_index()
        fig2 = go.Figure(data=[
            go.Bar(name='Base ECL', x=stage_data['final_stage'], y=stage_data['base_ecl'], marker_color='#4C72B0'),
            go.Bar(name='Stressed ECL', x=stage_data['final_stage'], y=stage_data['ecl_stressed'], marker_color='#C44E52')
        ])
        fig2.update_layout(barmode='group', title="ECL by IFRS 9 Stage (Base vs Stressed)",
                           xaxis_title="Stage", yaxis_title="ECL ($)")
        st.plotly_chart(fig2, use_container_width=True)

    # --- ROW 2: SECTOR CHART ---
    st.subheader("Sector Concentration")
    sec_chart = df.groupby("sector")[["base_ecl", "ecl_stressed"]].sum().reset_index()
    sec_chart = sec_chart.sort_values(by="ecl_stressed", ascending=True) # Sort for horizontal bar
    fig3 = px.bar(sec_chart, y="sector", x="ecl_stressed", orientation='h',
                  title="Stressed ECL by Sector",
                  labels={"ecl_stressed": "Stressed ECL ($)", "sector": ""},
                  color="ecl_stressed", color_continuous_scale="Reds")
    st.plotly_chart(fig3, use_container_width=True)

    # --- ROW 3: COLLAPSIBLE TABLE ---
    with st.expander("🔍 View Detailed Sector Data Table"):
        sec_sum = df.groupby("sector").agg({"ead": "sum", "base_ecl": "sum", "ecl_stressed": "sum", "loan_id": "count"}).reset_index()
        sec_sum.columns = ["Sector", "Total EAD", "Base ECL", "Stressed ECL", "Loan Count"]
        st.dataframe(sec_sum.style.format({"Total EAD": "${:,.0f}", "Base ECL": "${:,.0f}", "Stressed ECL": "${:,.0f}"}), use_container_width=True)