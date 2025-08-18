import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------
# Minimal app: compute FV of $2,000/month over 360 months
# using REAL monthly return factors from one allocation.
# Contribution timing: START of month, then apply factor:
#    balance = (balance + 2000) * factor
# -------------------------------------------------------

DATA_FOLDER = "data"
DEFAULT_CONTRIB = 2000.0
DEFAULT_MONTHS = 360

st.set_page_config(page_title="$2,000/mo for 360 months (Real Factors)", layout="centered")
st.title("FV of $2,000 per month — 360 months (Real Returns)")
st.caption("Loads REAL monthly total return factors from one XLSX in the 'data/' folder and computes the ending value.")

# --------- File loading helpers ---------

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
    fac = pd.to_numeric(xl.iloc[start:, 2], errors="coerce") if xl.shape[1] > 2 else pd.Series(dtype=float)
    mask = fac.notna()
    dates = end.where(end.notna(), beg)[mask].dropna()
    fac = fac[mask].loc[dates.index].astype(float).values
    return {"name": name, "dates": pd.DatetimeIndex(dates), "factors": fac}


def load_allocations(folder: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xlsx"):
            rec = load_allocation_xlsx(os.path.join(folder, fn))
            out[rec["name"]] = rec
    return out

# --------- UI ---------

if not os.path.isdir(DATA_FOLDER):
    st.error(f"Data folder '{DATA_FOLDER}' not found. Create it and add .xlsx files (e.g., '0 E.xlsx', '60 E.xlsx').")
    st.stop()

allocs = load_allocations(DATA_FOLDER)
if not allocs:
    st.error("No .xlsx files found in 'data/'.")
    st.stop()

names = sorted(allocs.keys())
choice = st.selectbox("Choose allocation file", names)
rec = allocs[choice]

col1, col2 = st.columns(2)
with col1:
    contrib = st.number_input("Monthly contribution ($)", min_value=0.0, value=DEFAULT_CONTRIB, step=100.0)
with col2:
    months = st.number_input("Number of months", min_value=1, max_value=600, value=DEFAULT_MONTHS, step=12)

if st.button("▶ Compute Ending Value"):
    f = rec["factors"].astype(float)
    if len(f) < months:
        st.error(f"Not enough monthly factors for {months} months. Available: {len(f)}.")
    else:
        # Start-of-month contribution, then grow by factor
        bal = 0.0
        for i in range(int(months)):
            bal += float(contrib)
            bal *= float(f[i])
        total_contrib = float(contrib) * int(months)
        gain = bal - total_contrib
        start_date = str(rec["dates"][0].date())
        end_date = str(rec["dates"][int(months)-1].date())

        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ending Value", f"${bal:,.0f}")
        c2.metric("Total Contributions", f"${total_contrib:,.0f}")
        c3.metric("Gain over Contributions", f"${gain:,.0f}")

        st.caption(f"Window used: {start_date} → {end_date}  (length: {int(months)} months)")

        # Optional: show the first few factors used
        with st.expander("Show first 12 monthly factors used"):
            st.write(pd.Series(f[:min(12, int(months))], name="Monthly factor (real)"))