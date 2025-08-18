import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Simplified model (NO TAXES)
# =============================
# Assumptions:
# - Returns provided are REAL (inflation-adjusted). We do not add inflation.
# - Borrow: keep full lump P invested for the horizon, no contributions (mortgage paid from outside income).
# - Pay Cash: invest the would-be mortgage payment M every month for up to the term (or horizon, if shorter).
# - We ignore home value/equity since it is the same in both scenarios; we compare only portfolio terminal values.

# ---------------- Utilities ----------------

def amortized_payment(P: float, annual_rate: float, term_months: int) -> float:
    """Standard mortgage payment formula."""
    r = annual_rate / 12.0
    if term_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return P / term_months
    return P * (r / (1.0 - (1.0 + r) ** (-term_months)))


def load_allocation_xlsx(path: str) -> Dict[str, Any]:
    """Load one allocation file.
    Expected columns:
      Col A: Beginning Date (ignored if Ending provided)
      Col B: Ending Date (preferred for monthly timestamp)
      Col C: Monthly total return factors (REAL), with a header cell (e.g. "60 E") in first few rows.
    """
    xl = pd.read_excel(path, header=None)
    name, header_row, header_col = None, None, 2
    for r in range(min(6, len(xl))):
        v = xl.iat[r, 2]
        if isinstance(v, str) and v.strip():
            name = v.strip(); header_row = r; break
    if name is None:
        name = os.path.basename(path).replace('.xlsx', '')
        header_row = 0
    start = header_row + 1
    beg = pd.to_datetime(xl.iloc[start:, 0], errors='coerce')
    end = pd.to_datetime(xl.iloc[start:, 1], errors='coerce')
    fac = pd.to_numeric(xl.iloc[start:, 2], errors='coerce')
    mask = fac.notna()
    dates = end.where(end.notna(), beg)[mask].dropna()
    fac = fac[mask].loc[dates.index].astype(float).values
    return {"name": name, "dates": pd.DatetimeIndex(dates), "factors": fac}


def load_allocations_folder(folder: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith('.xlsx'):
            rec = load_allocation_xlsx(os.path.join(folder, fn))
            out[rec['name']] = rec
    return out


def align_two(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Align two allocation series by common dates and return equal-length factor arrays + common dates."""
    sA = pd.Series(rec_a['factors'], index=rec_a['dates'])
    sB = pd.Series(rec_b['factors'], index=rec_b['dates'])
    common = sA.index.intersection(sB.index).sort_values()
    sA2 = sA.reindex(common).dropna()
    sB2 = sB.reindex(common).dropna()
    common2 = sA2.index.intersection(sB2.index)
    return sA2.loc[common2].values.astype(float), sB2.loc[common2].values.astype(float), common2


# ---------------- Core Simulation ----------------

def simulate_window_simple(lump_factors: np.ndarray,
                           contrib_factors: np.ndarray,
                           P: float, annual_rate: float, term_months: int,
                           horizon_months: int) -> Tuple[float, float]:
    """Return (Borrow_End_Portfolio, PayCash_End_Portfolio) for one window.
    Borrow: initial lump P invested, no monthly contributions.
    Pay Cash: invest monthly payment M up to min(term, horizon). No taxes.
    """
    n = horizon_months
    M = amortized_payment(P, annual_rate, term_months)

    # Borrow scenario: P grows by product of monthly factors over the horizon
    borrow_val = float(P) * float(np.prod(lump_factors[:n]))

    # Pay Cash scenario: monthly contributions of M for up to the term
    balance = 0.0
    for i in range(n):
        if i < term_months:
            balance += M
        balance *= contrib_factors[i]
    cash_val = float(balance)

    return borrow_val, cash_val


def run_rolling_windows_simple(rec_lump: Dict[str, Any],
                               rec_contrib: Dict[str, Any],
                               P: float, annual_rate: float, term_months: int,
                               horizon_months: int, step: int = 1) -> pd.DataFrame:
    lf_all, cf_all, dates = align_two(rec_lump, rec_contrib)
    n = horizon_months
    rows = []
    for start in range(0, len(dates) - n + 1, step):
        lf = lf_all[start:start + n]
        cf = cf_all[start:start + n]
        b, c = simulate_window_simple(lf, cf, P, annual_rate, term_months, n)
        rows.append({
            'start_date': str(dates[start].date()),
            'Borrow_End_Portfolio': b,
            'PayCash_End_Portfolio': c,
            'Advantage_(PayCash_minus_Borrow)': c - b,
        })
    return pd.DataFrame(rows)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Mortgage vs Invest — Simple (No Taxes)", layout="wide")
st.title("🏠 Mortgage vs. Keep Investing — Simple Model (No Taxes)")
st.caption("Uses REAL monthly return factors from \"data/\". Compares *Borrow 100%* vs *Pay Cash & invest payments*. Home equity ignored.")

# Fixed data folder per your setup
DATA_FOLDER = "data"
if not os.path.isdir(DATA_FOLDER):
    st.error(f"Data folder '{DATA_FOLDER}' not found. Create it and add .xlsx files (e.g., '0 E.xlsx', '60 E.xlsx').")
    st.stop()

allocs = load_allocations_folder(DATA_FOLDER)
if not allocs:
    st.error("No .xlsx files found in 'data/'.")
    st.stop()

names = sorted(allocs.keys())
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    lump_choice = st.selectbox("Lump Sum Allocation (used when Borrowing)", names, index=min(len(names)-1, names.index(names[len(names)//2])) if names else 0)
with col_sel2:
    contrib_choice = st.selectbox("Contribution Allocation (used when Paying Cash)", names, index=min(len(names)-1, names.index(names[len(names)//2])) if names else 0)

st.header("Mortgage & Simulation Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    principal = st.number_input("House Price / Loan Principal (P)", min_value=10000.0, value=300000.0, step=10000.0)
    apr = st.number_input("Mortgage APR (%)", min_value=0.01, value=5.0, step=0.01) / 100.0
with col2:
    term = st.slider("Mortgage Term (months)", min_value=12, max_value=480, value=360, step=12)
    horizon = st.slider("Horizon (months) — rolling window length", min_value=12, max_value=480, value=360, step=12)
with col3:
    step = st.slider("Rolling step (months)", min_value=1, max_value=12, value=3)

run = st.button("▶ Run Simple Backtest")

if run:
    rec_lump = allocs[lump_choice]
    rec_contrib = allocs[contrib_choice]

    # Validate data coverage
    lf, cf, dates = align_two(rec_lump, rec_contrib)
    min_len = len(dates)
    if min_len < horizon:
        st.error(f"Not enough overlapping data for {horizon} months. Only {min_len} months available.")
        st.stop()

    df = run_rolling_windows_simple(rec_lump, rec_contrib, principal, apr, term, horizon, step=step)

    if df.empty:
        st.error("No rolling windows produced (check horizon and data length).")
        st.stop()

    # Summary metrics
    adv = df['Advantage_(PayCash_minus_Borrow)']
    p_cash_better = float((adv > 0).mean())
    median_adv = float(adv.median())
    min_adv = float(adv.min())
    max_adv = float(adv.max())

    st.subheader("Summary Metrics")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("P(Pay Cash > Borrow)", f"{p_cash_better:.1%}")
    cB.metric("Median Advantage (PayCash − Borrow)", f"${median_adv:,.0f}")
    cC.metric("Min Advantage", f"${min_adv:,.0f}")
    cD.metric("Max Advantage", f"${max_adv:,.0f}")

    # Rolling outcome chart
    st.subheader("Rolling Outcomes (Terminal Portfolio Values)")
    chart_df = df.set_index('start_date')[['Borrow_End_Portfolio', 'PayCash_End_Portfolio']]
    st.line_chart(chart_df)

    # Advantage line
    st.subheader("Advantage Over Time (PayCash − Borrow)")
    st.line_chart(df.set_index('start_date')[['Advantage_(PayCash_minus_Borrow)']])

    # Download
    st.download_button(
        label="Download Results CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="mortgage_vs_invest_simple_results.csv",
        mime="text/csv",
    )