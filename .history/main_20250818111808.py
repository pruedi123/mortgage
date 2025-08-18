

import os
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# ---------------- Data Classes ----------------
@dataclass
class TaxParams:
    ltcg_rate: float = 0.15
    qualified_div_rate: float = 0.15
    ordinary_income_rate: float = 0.22
    state_tax_rate: float = 0.0
    pay_taxes_from_portfolio: bool = True
    basis_ratio_for_upfront_sale: float = 0.70
    def ltcg_total(self): return self.ltcg_rate + self.state_tax_rate
    def qualified_div_total(self): return self.qualified_div_rate + self.state_tax_rate
    def ordinary_total(self): return self.ordinary_income_rate + self.state_tax_rate

@dataclass
class PortfolioParams:
    dividend_yield_annual: float = 0.02
    dividend_qualified_pct: float = 1.0
    annual_turnover: float = 0.10

@dataclass
class MortgageParams:
    principal: float
    rate_annual: float
    term_months: int

@dataclass
class SimulationParams:
    initial_taxable_value: float
    horizon_months: int
    basis_ratio_for_upfront_sale: float = 0.70
    liquidate_at_end_and_pay_taxes: bool = True

# ---------------- Utilities ----------------
def amortized_payment(principal, rate_annual, term_months):
    r = rate_annual / 12
    if r == 0: return principal / term_months
    return principal * (r / (1 - (1 + r) ** (-term_months)))

def gross_sale_for_net_cash(net_cash_needed, basis_ratio, ltcg_total):
    denom = 1 - (1 - basis_ratio) * ltcg_total
    return net_cash_needed / denom if denom > 0 else float("inf")

def load_allocation_xlsx(path: str) -> Dict[str, Any]:
    xl = pd.read_excel(path, header=None)
    name = None
    header_row = 0
    header_col = 2
    for r in range(min(6, len(xl))):
        val = xl.iat[r, 2]
        if isinstance(val, str) and val.strip():
            name = val.strip()
            header_row = r
            break
    if name is None:
        name = os.path.basename(path).replace(".xlsx", "")
    start = header_row + 1
    dates = pd.to_datetime(xl.iloc[start:, 1], errors="coerce")
    fac = pd.to_numeric(xl.iloc[start:, 2], errors="coerce")
    mask = fac.notna()
    dates = dates[mask].dropna()
    fac = fac[mask].loc[dates.index].astype(float).values
    return {"name": name, "dates": dates, "factors": fac}

def load_allocations_folder(folder: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xlsx"):
            rec = load_allocation_xlsx(os.path.join(folder, fn))
            out[rec["name"]] = rec
    return out

def align_two(rec_a, rec_b):
    sA = pd.Series(rec_a["factors"], index=rec_a["dates"])
    sB = pd.Series(rec_b["factors"], index=rec_b["dates"])
    common = sA.index.intersection(sB.index)
    sA2 = sA.loc[common].dropna()
    sB2 = sB.loc[common].dropna()
    common2 = sA2.index.intersection(sB2.index)
    sA2 = sA2.loc[common2]
    sB2 = sB2.loc[common2]
    return sA2.values, sB2.values, common2

def apply_tax_drag(returns, port: PortfolioParams, tax: TaxParams):
    # Apply annual dividend and turnover drag (approximate: assumes monthly returns)
    months = len(returns)
    years = months // 12
    taxed_returns = returns.copy()
    for yr in range(years):
        idx = slice(yr * 12, (yr + 1) * 12)
        # Dividend drag
        div = port.dividend_yield_annual
        div_tax = div * (tax.qualified_div_total() * port.dividend_qualified_pct +
                         tax.ordinary_total() * (1 - port.dividend_qualified_pct))
        # Turnover drag
        turnover = port.annual_turnover
        turnover_tax = turnover * tax.ltcg_total()
        # Reduce the product of returns in that year by drag
        taxed_returns[idx] = taxed_returns[idx] * (1 - div_tax - turnover_tax)
    return taxed_returns

# ---------------- Simulation Core ----------------
def rolling_simulation(
    lump_factors,
    contrib_factors,
    tax: TaxParams,
    port_lump: PortfolioParams,
    port_contrib: PortfolioParams,
    mort: MortgageParams,
    sim: SimulationParams,
    dates
) -> pd.DataFrame:
    n = sim.horizon_months
    window = n
    n_paths = len(lump_factors) - window + 1
    results = []
    for i in range(n_paths):
        lf = lump_factors[i : i + window].copy()
        cf = contrib_factors[i : i + window].copy()
        # Apply tax drag
        lf = apply_tax_drag(lf, port_lump, tax)
        cf = apply_tax_drag(cf, port_contrib, tax)
        mpmt = amortized_payment(mort.principal, mort.rate_annual, mort.term_months)
        # Borrow scenario: keep lump fully invested, no contribs
        borrow_val = sim.initial_taxable_value * np.prod(lf)
        # Mortgage principal is owed if not paid off yet
        if n >= mort.term_months:
            borrow_NW = borrow_val + mort.principal
        else:
            borrow_NW = borrow_val
        # Pay Cash: sell upfront with gross-up
        gross_sale = gross_sale_for_net_cash(
            mort.principal,
            sim.basis_ratio_for_upfront_sale,
            tax.ltcg_total()
        )
        lump_after = sim.initial_taxable_value - gross_sale
        # Lump after sale, grows
        lump_val = lump_after * np.prod(lf)
        # Invest freed-up mortgage payments monthly into contrib allocation
        contrib_vals = []
        balance = 0.0
        for m in range(window):
            if m < mort.term_months:
                balance += mpmt
            balance *= cf[m]
            contrib_vals.append(balance)
        contrib_val = contrib_vals[-1] if contrib_vals else 0.0
        # At end, liquidate and pay taxes (if applicable)
        if sim.liquidate_at_end_and_pay_taxes:
            # Assume all gain is LTCG, basis ratio applies to lump and contrib
            lump_basis = lump_after * sim.basis_ratio_for_upfront_sale
            lump_gain = max(lump_val - lump_basis, 0)
            lump_tax = lump_gain * tax.ltcg_total()
            lump_val -= lump_tax
            contrib_basis = sum([mpmt * sim.basis_ratio_for_upfront_sale for _ in range(min(mort.term_months, window))])
            contrib_gain = max(contrib_val - contrib_basis, 0)
            contrib_tax = contrib_gain * tax.ltcg_total()
            contrib_val -= contrib_tax
        cash_NW = lump_val + contrib_val + mort.principal
        results.append({
            "start_date": dates[i],
            "Borrow_End_NW": borrow_NW,
            "Cash_End_NW": cash_NW,
            "Advantage": cash_NW - borrow_NW
        })
    return pd.DataFrame(results)

# ---------------- Streamlit UI ----------------
st.title("Mortgage vs Invest — Dual Allocation (Real Returns)")
st.markdown("Assumes allocation files are in the `data/` subfolder.")

folder = "data"
if not os.path.isdir(folder):
    st.error(f"Data folder '{folder}' not found. Please create it and add .xlsx files.")
    st.stop()

allocs = load_allocations_folder(folder)
if not allocs:
    st.error("No .xlsx files found in 'data/'.")
    st.stop()

names = sorted(allocs.keys())
lump_choice = st.selectbox("Lump Sum Allocation", names)
contrib_choice = st.selectbox("Contribution Allocation", names)

st.header("Mortgage Inputs")
col1, col2 = st.columns(2)
with col1:
    principal = st.number_input("Loan Principal", min_value=10000.0, value=300000.0, step=10000.0)
    term = st.slider("Term (months)", 12, 480, 360, step=12)
with col2:
    apr = st.number_input("APR (%)", min_value=0.01, value=5.0, step=0.01) / 100
    horizon = st.slider("Simulation Horizon (months)", 12, 480, 360, step=12)

st.header("Tax Inputs")
col3, col4 = st.columns(2)
with col3:
    ltcg = st.number_input("Long-Term Cap Gains Rate (%)", min_value=0.0, value=15.0, step=0.1) / 100
    div = st.number_input("Qualified Dividend Rate (%)", min_value=0.0, value=15.0, step=0.1) / 100
    ordinary = st.number_input("Ordinary Income Rate (%)", min_value=0.0, value=22.0, step=0.1) / 100
with col4:
    state = st.number_input("State Tax Rate (%)", min_value=0.0, value=0.0, step=0.1) / 100
    basis_pct = st.number_input("Basis Ratio for Upfront Sale (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0) / 100
    init_taxable = st.number_input("Initial Taxable Assets", min_value=0.0, value=300000.0, step=10000.0)

st.header("Portfolio Inputs")
col5, col6 = st.columns(2)
with col5:
    div_yield = st.number_input("Dividend Yield (%)", min_value=0.0, value=2.0, step=0.1) / 100
    div_qual = st.number_input("Dividends Qualified (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0) / 100
with col6:
    turnover = st.number_input("Annual Turnover (%)", min_value=0.0, value=10.0, step=0.1) / 100

if st.button("Run Simulation"):
    lump = allocs[lump_choice]
    contrib = allocs[contrib_choice]
    lf, cf, dates = align_two(lump, contrib)
    # Ensure enough data for the chosen horizon
    min_len = min(len(lf), len(cf), len(dates))
    if min_len < horizon:
        st.error(f"Not enough data for {horizon} months. Only {min_len} months available.")
        st.stop()
    tax = TaxParams(
        ltcg_rate=ltcg,
        qualified_div_rate=div,
        ordinary_income_rate=ordinary,
        state_tax_rate=state,
        basis_ratio_for_upfront_sale=basis_pct
    )
    port_lump = PortfolioParams(
        dividend_yield_annual=div_yield,
        dividend_qualified_pct=div_qual,
        annual_turnover=turnover
    )
    port_contrib = PortfolioParams(
        dividend_yield_annual=div_yield,
        dividend_qualified_pct=div_qual,
        annual_turnover=turnover
    )
    mort = MortgageParams(
        principal=principal,
        rate_annual=apr,
        term_months=term
    )
    sim = SimulationParams(
        initial_taxable_value=init_taxable,
        horizon_months=horizon,
        basis_ratio_for_upfront_sale=basis_pct
    )
    df = rolling_simulation(lf, cf, tax, port_lump, port_contrib, mort, sim, dates)
    # Summary metrics
    p_cash_beats = np.mean(df["Advantage"] > 0)
    median_adv = np.median(df["Advantage"])
    st.subheader("Summary Metrics")
    st.metric("P(Pay Cash > Borrow)", f"{p_cash_beats:.1%}")
    st.metric("Median Advantage (Cash - Borrow)", f"${median_adv:,.0f}")
    st.metric("Max Advantage", f"${df['Advantage'].max():,.0f}")
    st.metric("Min Advantage", f"${df['Advantage'].min():,.0f}")
    # Line chart
    st.subheader("Rolling Outcomes")
    chart_df = df.set_index("start_date")[["Borrow_End_NW", "Cash_End_NW", "Advantage"]]
    st.line_chart(chart_df)
    # CSV download
    st.download_button(
        label="Download Results CSV",
        data=chart_df.to_csv().encode("utf-8"),
        file_name="mortgage_vs_invest_results.csv",
        mime="text/csv"
    )