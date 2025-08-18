

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =====================================
# Simplified model (NO TAXES)
# =====================================
# Assumptions:
# - Returns provided are REAL (inflation-adjusted). We do not add inflation.
# - Borrow: keep full lump P invested for the horizon, no contributions (mortgage paid from outside income).
# - Pay Cash: invest the would-be mortgage payment M each month up to the term.
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
    # find header name in first few rows
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


# ---------------- Core Single-Window Comparison ----------------

def compute_ending_values_first_window(rec_lump: Dict[str, Any],
                                       rec_contrib: Dict[str, Any],
                                       P: float, annual_rate: float, term_months: int,
                                       horizon_months: int) -> Tuple[float, float, str, str]:
    """Compute Borrow vs Pay-Cash ending portfolio values for the FIRST available window.
    Returns (borrow_end, cash_end, start_date_str, end_date_str).
    """
    lf_all, cf_all, dates = align_two(rec_lump, rec_contrib)
    if len(dates) < horizon_months:
        raise ValueError(f"Not enough overlapping data for {horizon_months} months. Only {len(dates)} months available.")

    lf = lf_all[:horizon_months]
    cf = cf_all[:horizon_months]

    # Monthly payment and contribution cap
    M = amortized_payment(P, annual_rate, term_months)

    # Borrow: invest lump P for horizon
    borrow_end = float(P) * float(np.prod(lf))

    # Pay Cash: invest M each month up to term, then just grow
    cash_end = 0.0
    for i in range(horizon_months):
        if i < term_months:
            cash_end += M
        cash_end *= cf[i]

    return borrow_end, cash_end, str(dates[0].date()), str(dates[horizon_months-1].date())


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Mortgage vs Invest — Simple (No Taxes)", layout="wide")
st.title("🏠 Mortgage vs. Keep Investing — Simple Model (No Taxes)")
st.caption("Uses REAL monthly return factors from the 'data/' folder. Compares *Borrow 100%* vs *Pay Cash & invest payments*. Home equity ignored.")

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
    lump_choice = st.selectbox("Lump Sum Allocation (Borrow)", names)
with col_sel2:
    contrib_choice = st.selectbox("Contribution Allocation (Pay Cash)", names)

st.header("Mortgage & Horizon")
col1, col2, col3 = st.columns(3)
with col1:
    principal = st.number_input("House Price / Loan Principal (P)", min_value=10000.0, value=300000.0, step=10000.0)
with col2:
    apr = st.number_input("Mortgage APR (%)", min_value=0.01, value=5.0, step=0.01) / 100.0
with col3:
    term = st.slider("Mortgage Term (months)", min_value=12, max_value=480, value=360, step=12)

horizon = st.slider("Horizon (months)", min_value=12, max_value=480, value=360, step=12)

if st.button("▶ Compare Ending Portfolio Values"):
    try:
        rec_lump = allocs[lump_choice]
        rec_contrib = allocs[contrib_choice]
        borrow_end, cash_end, d0, d1 = compute_ending_values_first_window(
            rec_lump, rec_contrib, principal, apr, term, horizon
        )

        st.subheader("Ending Portfolio Values — First Available Window")
        c1, c2, c3 = st.columns(3)
        c1.metric("Borrow — Ending Portfolio", f"${borrow_end:,.0f}")
        c2.metric("Pay Cash — Ending Portfolio", f"${cash_end:,.0f}")
        c3.metric("Advantage (Pay Cash − Borrow)", f"${(cash_end - borrow_end):,.0f}")

        st.caption(f"Evaluated horizon: {horizon} months, from {d0} to {d1}")
    except Exception as e:
        st.error(str(e))