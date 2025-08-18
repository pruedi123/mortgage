import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# Ending value of fixed monthly contribution, using
# ANNUAL returns/factors spaced 12 rows apart.
# For a start row s and horizon of M months:
#   year k (0-indexed) uses index s + 12*k
#   monthly_factor = (annual_factor) ** (1/12)
# Contribution timing: start-of-month, then apply factor.
# =====================================================

DATA_FOLDER = "data"
DEFAULT_CONTRIB = 2000.0
DEFAULT_MONTHS = 360

st.set_page_config(page_title="Ending Value per Start Month — Annual Inputs", layout="wide")
st.title("Ending Value of Investing the Mortgage Payment — Annual Inputs (12-row spacing)")
st.caption("Column 3 is ANNUAL data: for year 1 use current row, year 2 use row+12, etc. Each month uses (annual_factor)**(1/12).")

# ---------- Mortgage helper ----------
def amortized_payment(principal: float, annual_rate: float, term_months: int) -> float:
    r = annual_rate / 12.0
    if term_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return principal / term_months
    return principal * (r / (1.0 - (1.0 + r) ** (-term_months)))

# ---------- Helpers ----------

def load_allocation_xlsx(path: str) -> Dict[str, Any]:
    xl = pd.read_excel(path, header=None)
    name, header_row = None, None
    for r in range(min(6, len(xl))):
        v = xl.iat[r, 2] if xl.shape[1] > 2 else None
        if isinstance(v, str) and v.strip():
            name = v.strip(); header_row = r; break
    if name is None:
        name = os.path.basename(path).replace(".xlsx", "")
        header_row = 0
    start = header_row + 1
    beg = pd.to_datetime(xl.iloc[start:, 0], errors="coerce") if xl.shape[1] > 0 else pd.Series(dtype="datetime64[ns]")
    end = pd.to_datetime(xl.iloc[start:, 1], errors="coerce") if xl.shape[1] > 1 else pd.Series(dtype="datetime64[ns]")
    col3 = pd.to_numeric(xl.iloc[start:, 2], errors="coerce") if xl.shape[1] > 2 else pd.Series(dtype=float)
    mask = col3.notna()
    dates = end.where(end.notna(), beg)[mask].dropna()
    col3 = col3[mask].loc[dates.index].astype(float).values
    return {"name": name, "dates": pd.DatetimeIndex(dates), "annual_col": col3}

def load_allocations(folder: str):
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xlsx"):
            rec = load_allocation_xlsx(os.path.join(folder, fn))
            out[rec["name"]] = rec
    return out

def annual_input_to_factor(a: float, mode: str) -> float:
    """Convert a single ANNUAL input to an annual factor (1+R). Mode:
       - 'Annual returns' : a is return; detect decimal vs percent
       - 'Annual factors (1+R)' : a is already a factor
    """
    if mode == "Annual factors (1+R)":
        return float(a)
    # Annual returns: detect percent vs decimal by magnitude
    if abs(a) > 1.0:
        return 1.0 + a / 100.0  # percent
    else:
        return 1.0 + a          # decimal

# ---------- UI ----------

allocs = load_allocations(DATA_FOLDER)
if not allocs:
    st.error("No .xlsx files found in 'data/'.")
    st.stop()

names = sorted(allocs.keys())
choice = st.selectbox("Choose allocation file", names)
rec = allocs[choice]

mode = st.radio("Third column contains:", ["Annual returns", "Annual factors (1+R)"], index=0)

st.subheader("Mortgage & Horizon")
col1, col2, col3 = st.columns(3)
with col1:
    principal = st.number_input("Loan Principal ($)", min_value=10000.0, value=300000.0, step=10000.0)
with col2:
    apr_pct = st.number_input("APR (%)", min_value=0.0, value=5.0, step=0.01)
with col3:
    term = st.number_input("Term (months)", min_value=12, max_value=600, value=360, step=12)

months = st.number_input("Horizon in months", min_value=12, max_value=600, value=DEFAULT_MONTHS, step=12)

# Compute monthly payment (will be used as contribution up to `term` months)
mpmt = amortized_payment(float(principal), float(apr_pct) / 100.0, int(term))
st.info(f"Calculated monthly payment: ${mpmt:,.2f}")

if st.button("▶ Compute Ending Values for All Start Periods (Using Mortgage Payment)"):
    a = rec["annual_col"].astype(float)  # annual inputs at monthly rows

    # We need enough annual rows at offsets s + 12*k for k up to years-1
    years_needed = int(np.ceil(int(months) / 12.0))
    max_start = len(a) - 12 * (years_needed - 1) - 1
    if max_start < 0:
        st.error(
            f"Not enough rows for {int(months)} months (needs {years_needed} annual steps). Available rows: {len(a)}."
        )
    else:
        rows = []
        months_int = int(months)
        contrib_float = float(mpmt)
        for s in range(0, max_start + 1):
            bal = 0.0
            for m in range(months_int):
                year_idx = m // 12
                a_idx = s + 12 * year_idx
                ann_factor = annual_input_to_factor(float(a[a_idx]), mode)
                month_factor = ann_factor ** (1.0 / 12.0)
                # Contribute only while mortgage is active (m < term)
                if m < int(term):
                    bal += float(mpmt)
                bal *= month_factor
            start_date = str(rec["dates"][s].date())
            end_row = s + months_int - 1
            if end_row < len(rec["dates"]):
                end_date = str(rec["dates"][end_row].date())
            else:
                # fallback if monthly date rows are shorter than months horizon
                end_date = start_date
            window_contrib_months = min(months_int, int(term))
            window_total_contrib = float(mpmt) * window_contrib_months
            rows.append({
                "start_date": start_date,
                "end_date": end_date,
                "ending_value": bal,
                "monthly_payment": mpmt,
                "contrib_months_in_window": window_contrib_months,
                "total_contributions": window_total_contrib,
                "gain_over_contributions": bal - window_total_contrib,
            })
        df = pd.DataFrame(rows)
        st.subheader("Ending Value per Starting Month (Annual inputs, 12-row spacing)")
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index("start_date")["ending_value"])
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ending_values_all_start_periods_annual.csv",
            mime="text/csv",
        )

        with st.expander("Peek at first 12 annual inputs (raw)"):
            st.write(pd.Series(a[:12], name="annual_col_raw"))