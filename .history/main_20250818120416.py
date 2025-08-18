import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# Compare Pay-Cash vs Borrow (No taxes, Real returns)
# Annual inputs spaced 12 rows apart -> monthly factors = annual_factor**(1/12)
# For each start row s and horizon M months:
#   year k uses index s + 12*k
# Pay Cash: contribute monthly payment while m < term, then grow
# Borrow: invest principal lump for entire horizon (no contributions)
# Output: per-start rows + percentile/probability summary
# =====================================================

DATA_FOLDER = "data"
DEFAULT_MONTHS = 360

st.set_page_config(page_title="Cash vs Borrow — Ending Values & Probabilities", layout="wide")
st.title("Cash vs Borrow — Ending Balances & Probability Summary (Annual inputs, 12-row spacing)")
st.caption("Column 3 is ANNUAL data. For a start row s: year 1 uses s, year 2 uses s+12, etc. Monthly factor = (annual_factor)**(1/12). Contributions are start-of-month.")

# ---------- Mortgage helper ----------

def amortized_payment(principal: float, annual_rate: float, term_months: int) -> float:
    r = annual_rate / 12.0
    if term_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return principal / term_months
    return principal * (r / (1.0 - (1.0 + r) ** (-term_months)))

# ---------- File helpers ----------

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

# ---------- Annual input conversion ----------

def annual_input_to_factor(a: float, mode: str) -> float:
    """Convert ONE ANNUAL input to annual factor (1+R).
    mode: 'Annual returns' (auto-detect decimal vs percent) or 'Annual factors (1+R)'.
    """
    if mode == "Annual factors (1+R)":
        return float(a)
    # Annual returns: detect percent vs decimal by magnitude
    if abs(a) > 1.0:
        return 1.0 + a / 100.0  # percent units
    else:
        return 1.0 + a          # decimal units

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
col1, col2, col3, col4 = st.columns(4)
with col1:
    principal = st.number_input("Loan Principal ($)", min_value=10000.0, value=300000.0, step=10000.0)
with col2:
    apr_pct = st.number_input("APR (%)", min_value=0.0, value=5.0, step=0.25)
with col3:
    term = st.number_input("Term (months)", min_value=12, max_value=600, value=360, step=12)
with col4:
    months = st.number_input("Horizon (months)", min_value=12, max_value=600, value=DEFAULT_MONTHS, step=12)

# Monthly payment (used as contribution in Pay-Cash path while loan active)
mpmt = amortized_payment(float(principal), float(apr_pct) / 100.0, int(term))
st.info(f"Calculated monthly payment: ${mpmt:,.2f}")

# Upfront tax if paying cash — this increases the true amount that would have stayed invested if we borrowed
upfront_tax_pct = st.number_input("Upfront tax rate to free cash (%)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
upfront_tax = float(upfront_tax_pct) / 100.0
if upfront_tax >= 1.0:
    st.error("Upfront tax rate must be < 100%.")
    st.stop()

# Gross withdrawal needed to net the principal after tax: G = P / (1 - t)
# This is the true opportunity cost base we compare against (i.e., what remains invested if you borrow instead of paying cash)
P_gross = float(principal) / (1.0 - upfront_tax) if upfront_tax < 1.0 else float('inf')
st.info(f"Gross withdrawal required to net ${principal:,.0f} at {upfront_tax_pct:.1f}% tax: ${P_gross:,.0f}")

if st.button("▶ Compute Per-Start Rows + Probability Summary"):
    a = rec["annual_col"].astype(float)

    # years needed to cover the horizon
    years_needed = int(np.ceil(int(months) / 12.0))
    max_start = len(a) - 12 * (years_needed - 1) - 1
    if max_start < 0:
        st.error(
            f"Not enough rows for {int(months)} months (needs {years_needed} annual steps). Available rows: {len(a)}."
        )
    else:
        rows = []
        months_int = int(months)
        term_int = int(term)
        P = float(P_gross)  # opportunity cost base: what would have stayed invested if we borrowed
        paymt = float(mpmt)

        for s in range(0, max_start + 1):
            # --- Pay Cash path ---
            bal_cash = 0.0
            # --- Borrow path (lump P invested) ---
            bal_borrow = P

            for m in range(months_int):
                year_idx = m // 12
                a_idx = s + 12 * year_idx
                ann_factor = annual_input_to_factor(float(a[a_idx]), mode)
                month_factor = ann_factor ** (1.0 / 12.0)

                # Pay Cash: add payment while loan active, then grow
                if m < term_int:
                    bal_cash += paymt
                bal_cash *= month_factor

                # Borrow: no contributions; lump grows
                bal_borrow *= month_factor

            start_date = str(rec["dates"][s].date())
            end_row = s + months_int - 1
            end_date = str(rec["dates"][end_row].date()) if end_row < len(rec["dates"]) else start_date

            window_contrib_months = min(months_int, term_int)
            total_contrib = paymt * window_contrib_months

            rows.append({
                "start_date": start_date,
                "end_date": end_date,
                "ending_paycash": bal_cash,
                "ending_borrow": bal_borrow,
                "advantage_paycash_minus_borrow": bal_cash - bal_borrow,
                "monthly_payment": paymt,
                "gross_withdrawal_for_pay_cash": P_gross,
                "contrib_months_in_window": window_contrib_months,
                "total_contributions": total_contrib,
            })

        df = pd.DataFrame(rows)

        # ---- Summary statistics on advantage ----
        adv = df["advantage_paycash_minus_borrow"].values
        prob_paycash_wins = float((adv > 0).mean()) if len(adv) else float("nan")
        p10, p50, p90 = (np.percentile(adv, q) for q in (10, 50, 90)) if len(adv) else (float("nan"),)*3

        st.subheader("Probability & Percentiles (Pay Cash advantage = PayCash − Borrow)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("P(Pay Cash > Borrow)", f"{prob_paycash_wins:.1%}")
        s2.metric("P10 (Advantage)", f"${p10:,.0f}")
        s3.metric("P50 / Median (Advantage)", f"${p50:,.0f}")
        s4.metric("P90 (Advantage)", f"${p90:,.0f}")

        # --- Plain‑English summary for clients ---
        # Advantage = Pay Cash − Borrow. Positive means Pay Cash ends higher; negative means Borrow ends higher.
        median_adv = float(p50)
        prob_pc = float(prob_paycash_wins)

        def _fmt_money(x: float) -> str:
            sign = "-" if x < 0 else ""
            return f"{sign}${abs(x):,.0f}"

        typical_line = (
            f"Typically (median), paying cash ends **{_fmt_money(abs(median_adv))} {'ahead of' if median_adv > 0 else 'behind'}** borrowing."
        )
        prob_line = (
            f"Across all historical start dates tested, paying cash wins **{prob_pc:.1%}** of the time; borrowing wins **{1-prob_pc:.1%}**."
        )

        # Simple recommendation heuristic: go with the option that wins more often; use median to describe typical gap.
        if prob_pc > 0.6 and median_adv > 0:
            rec_line = "**Bottom line:** Paying cash usually comes out ahead in this setup."
        elif prob_pc < 0.4 and median_adv < 0:
            rec_line = "**Bottom line:** Borrowing usually comes out ahead in this setup."
        else:
            rec_line = "**Bottom line:** Results are mixed; either choice can work depending on start date and returns."

        st.markdown(
            """
            ### What this means in plain English
            - **What we’re comparing:** Two portfolios. **Pay Cash** invests the monthly mortgage payment; **Borrow** invests the full loan amount now.
            - **How to read the numbers:** If the advantage is positive, *Pay Cash* finished higher; if negative, *Borrow* finished higher.
            
            {prob_line}
            
            {typical_line}
            
            {rec_line}
            """.format(prob_line=prob_line, typical_line=typical_line, rec_line=rec_line)
        )
        st.caption(
            "We treat the upfront tax as an opportunity cost: if paying cash requires selling to net the principal, we assume the **gross** amount (principal / (1 − tax%)) would have otherwise stayed invested in the Borrow path."
        )

        st.subheader("Per-Start Ending Values")
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index("start_date")[ ["ending_paycash", "ending_borrow"] ])

        st.download_button(
            "Download Per-Start Rows (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cash_vs_borrow_per_start.csv",
            mime="text/csv",
        )

        with st.expander("Peek at first 12 annual inputs (raw)"):
            st.write(pd.Series(a[:12], name="annual_col_raw"))