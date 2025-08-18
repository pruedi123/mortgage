import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------
# Minimal app: compute FV of $2,000/month over 360 months
# using REAL monthly data from one allocation.
# If column 3 is returns, convert to factors first.
# Contribution timing: START of month, then apply factor:
#    balance = (balance + 2000) * factor
# -------------------------------------------------------

DATA_FOLDER = "data"
DEFAULT_CONTRIB = 2000.0
DEFAULT_MONTHS = 360

st.set_page_config(page_title="$2,000/mo for 360 months (Real Data)", layout="centered")
st.title("FV of $2,000 per month — 360 months (Real Data)")
st.caption("Loads one XLSX from 'data/'. If column 3 is returns, they are converted to factors before compounding.")

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
    col3 = pd.to_numeric(xl.iloc[start:, 2], errors="coerce") if xl.shape[1] > 2 else pd.Series(dtype=float)
    mask = col3.notna()
    dates = end.where(end.notna(), beg)[mask].dropna()
    col3 = col3[mask].loc[dates.index].astype(float).values
    return {"name": name, "dates": pd.DatetimeIndex(dates), "col3": col3}

def load_allocations(folder: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith(".xlsx"):
            rec = load_allocation_xlsx(os.path.join(folder, fn))
            out[rec["name"]] = rec
    return out

def returns_to_factors(arr: np.ndarray) -> np.ndarray:
    """
    Convert monthly returns to factors. Auto-detect decimal vs percent:
      - If median around ~1, assume already factors (return as-is).
      - If typical absolute size > 1, treat as percent (e.g., 1.2 -> 1.012).
      - Else treat as decimal (e.g., 0.012 -> 1.012).
    """
    if arr.size == 0:
        return arr
    med = float(np.nanmedian(arr))
    if 0.8 < med < 1.2:
        return arr.astype(float)  # already looks like factors
    abs_med = float(np.nanmedian(np.abs(arr)))
    if abs_med > 1.0:
        # likely percent values
        return (1.0 + arr / 100.0).astype(float)
    else:
        # likely decimal returns
        return (1.0 + arr).astype(float)

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

mode = st.radio("Third column contains:", ["Monthly returns", "Return factors (1+r)"], index=0,
                help="Pick 'Monthly returns' if values look like 0.012 (decimal) or 1.2 (percent) for +1.2%.")

col1, col2 = st.columns(2)
with col1:
    contrib = st.number_input("Monthly contribution ($)", min_value=0.0, value=DEFAULT_CONTRIB, step=100.0)
with col2:
    months = st.number_input("Number of months", min_value=1, max_value=600, value=DEFAULT_MONTHS, step=12)

if st.button("▶ Compute Ending Value"):
    raw = rec["col3"].astype(float)
    factors = returns_to_factors(raw) if mode == "Monthly returns" else raw

    if len(factors) < months:
        st.error(f"Not enough months for {months}. Available: {len(factors)}.")
    else:
        bal = 0.0
        for i in range(int(months)):
            bal += float(contrib)         # start-of-month contribution
            bal *= float(factors[i])      # then apply month's factor

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

        with st.expander("Show first 12 values (raw & interpreted factors)"):
            nshow = min(12, int(months))
            st.write(pd.DataFrame({
                "raw_col3": raw[:nshow],
                "interpreted_factor": factors[:nshow]
            }))