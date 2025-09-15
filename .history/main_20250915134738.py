import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
#paul
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
st.title("Cash vs Borrow — Ending Balances & Probability Summary")

mode = st.radio(
    "Comparison Mode",
    ["Pay Cash vs Borrow", "Borrow More vs Bigger Down"],
    index=0,
    horizontal=True,
    help="Pay Cash vs Borrow: classic principal vs investing.  Borrow More vs Bigger Down: small down invests the upfront difference; big down invests the monthly payment savings."
)

# ---------- Mortgage helper ----------

def amortized_payment(principal: float, annual_rate: float, term_months: int) -> float:
    r = annual_rate / 12.0
    if term_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return principal / term_months
    return principal * (r / (1.0 - (1.0 + r) ** (-term_months)))

def remaining_principal(principal: float, annual_rate: float, term_months: int, months_elapsed: int) -> float:
    """
    Remaining mortgage balance after k payments with fixed payment schedule.
    annual_rate is a decimal APR (e.g., 0.065 for 6.5%), matching amortized_payment().
    """
    r = annual_rate / 12.0
    if term_months <= 0:
        return 0.0
    k = max(0, min(int(months_elapsed), int(term_months)))
    if abs(r) < 1e-12:
        # Linear paydown if zero interest
        return max(0.0, principal * (1 - k / float(term_months)))
    pmt = amortized_payment(principal, annual_rate, term_months)
    return principal * (1 + r) ** k - pmt * ((1 + r) ** k - 1) / r

# ---------- File helpers ----------

def _parse_table(df: pd.DataFrame) -> Dict[str, Any]:
    name, header_row = None, None
    for r in range(min(6, len(df))):
        v = df.iat[r, 2] if df.shape[1] > 2 else None
        if isinstance(v, str) and v.strip():
            name = v.strip(); header_row = r; break
    if name is None:
        name = "allocation"
        header_row = 0
    start = header_row + 1
    beg = pd.to_datetime(df.iloc[start:, 0], errors="coerce") if df.shape[1] > 0 else pd.Series(dtype="datetime64[ns]")
    end = pd.to_datetime(df.iloc[start:, 1], errors="coerce") if df.shape[1] > 1 else pd.Series(dtype="datetime64[ns]")
    col3 = pd.to_numeric(df.iloc[start:, 2], errors="coerce") if df.shape[1] > 2 else pd.Series(dtype=float)
    mask = col3.notna()
    dates = end.where(end.notna(), beg)[mask].dropna()
    col3 = col3[mask].loc[dates.index].astype(float).values
    return {"name": name, "dates": pd.DatetimeIndex(dates), "annual_col": col3}


def load_allocation_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        try:
            df = pd.read_excel(path, header=None, engine="openpyxl")
        except Exception as e:
            # Show a helpful inline error and re-raise to stop early (so the user fixes requirements)
            st.error("To read .xlsx on Streamlit Cloud, add `openpyxl>=3.1` to `requirements.txt`, or provide `.csv` files instead.")
            raise
        rec = _parse_table(df)
        # Use filename as name if header missing
        base = os.path.basename(path).replace(".xlsx", "")
        if rec["name"] == "allocation":
            rec["name"] = base
        return rec
    elif ext == ".csv":
        df = pd.read_csv(path, header=None)
        rec = _parse_table(df)
        base = os.path.basename(path).replace(".csv", "")
        if rec["name"] == "allocation":
            rec["name"] = base
        return rec
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_allocations(folder: str):
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        lower = fn.lower()
        if lower.endswith(".xlsx") or lower.endswith(".csv"):
            path = os.path.join(folder, fn)
            try:
                rec = load_allocation_file(path)
                out[rec["name"]] = rec
            except Exception:
                # Skip unreadable files but continue; message already shown via st.error for .xlsx w/o openpyxl
                continue
    return out

# ---------- Annual input conversion ----------

def annual_input_to_factor(a: float) -> float:
    """Convert ONE ANNUAL return to annual factor (1+R). Auto‑detect decimal vs percent."""
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
def_index = names.index("100 E") if "100 E" in names else 0
choice = st.selectbox("Choose allocation file", names, index=def_index)
rec = allocs[choice]

st.caption("Note: Third column is assumed to be **annual returns** (decimal or percent). We auto‑detect units.")

st.subheader("Mortgage & Horizon")
if mode == "Pay Cash vs Borrow":
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
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        home_price = st.number_input("Home Price ($)", min_value=10000.0, value=500000.0, step=10000.0)
    with col2:
        apr_pct = st.number_input("APR (%)", min_value=0.0, value=6.50, step=0.05)
    with col3:
        term = st.number_input("Term (months)", min_value=12, max_value=600, value=360, step=12)
    with col4:
        months = st.number_input("Horizon (months)", min_value=12, max_value=600, value=180, step=12)
    col5, col6 = st.columns(2)
    with col5:
        down_small_pct = st.slider("Smaller Down Payment (%)", 0.0, 40.0, 5.0, step=0.5)
    with col6:
        down_big_pct = st.slider("Bigger Down Payment (%)", 0.0, 60.0, 20.0, step=0.5)
    if down_big_pct < down_small_pct:
        st.error("Bigger down must be ≥ smaller down. Adjust sliders.")

if mode == "Pay Cash vs Borrow":
    upfront_tax_pct = st.number_input("Upfront tax rate to free cash (%)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
    upfront_tax = float(upfront_tax_pct) / 100.0
    if upfront_tax >= 1.0:
        st.error("Upfront tax rate must be < 100%.")
        st.stop()
    P_gross = float(principal) / (1.0 - upfront_tax) if upfront_tax < 1.0 else float('inf')
    c1, c2, c3 = st.columns(3)
    c1.metric("Net needed (principal)", f"${principal:,.0f}")
    c2.metric("Upfront tax rate", f"{upfront_tax_pct:.1f}%")
    c3.metric("Gross withdrawal required", f"${P_gross:,.0f}")

if st.button("▶ Compute Per-Start Rows + Probability Summary"):
    a = rec["annual_col"].astype(float)
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

        for s in range(0, max_start + 1):
            # Build per-start monthly path using annual inputs turned into monthly factors
            def month_factor_at(m: int) -> float:
                year_idx = m // 12
                a_idx = s + 12 * year_idx
                ann_factor = annual_input_to_factor(float(a[a_idx]))
                return ann_factor ** (1.0 / 12.0)

            if mode == "Pay Cash vs Borrow":
                # --- Pay Cash path ---
                bal_cash = 0.0
                # --- Borrow path (lump P invested) ---
                P = float(P_gross)  # opportunity cost base
                bal_borrow = P
                paymt = amortized_payment(float(principal), float(apr_pct) / 100.0, term_int)

                for m in range(months_int):
                    mf = month_factor_at(m)
                    if m < term_int:
                        bal_cash += paymt  # start-of-month contribution
                    bal_cash *= mf
                    bal_borrow *= mf

                start_date = str(rec["dates"][s].date())
                end_row = s + months_int - 1
                end_date = str(rec["dates"][end_row].date()) if end_row < len(rec["dates"]) else start_date

                window_contrib_months = min(months_int, term_int)
                total_contrib = paymt * window_contrib_months

                # Net worth comparison: Borrow net = investments - remaining mortgage
                rem_borrow = remaining_principal(principal=float(principal), annual_rate=float(apr_pct) / 100.0, term_months=term_int, months_elapsed=months_int)
                nw_paycash = bal_cash
                nw_borrow = bal_borrow - rem_borrow
                advantage = nw_paycash - nw_borrow

                rows.append({
                    "mode": "Pay Cash vs Borrow",
                    "start_date": start_date,
                    "end_date": end_date,
                    "networth_paycash": nw_paycash,
                    "networth_borrow": nw_borrow,
                    "remaining_principal_borrow": rem_borrow,
                    "advantage_paycash_minus_borrow": advantage,
                    "monthly_payment": paymt,
                    "gross_withdrawal_for_pay_cash": P_gross,
                    "contrib_months_in_window": window_contrib_months,
                    "total_contributions": total_contrib,
                })

            else:
                # --- Borrow More (MB) vs Bigger Down (BD) ---
                price = float(home_price)
                small = float(down_small_pct) / 100.0
                big = float(down_big_pct) / 100.0
                extra_down = max(0.0, price * (big - small))

                loan_MB = price * (1.0 - small)
                loan_BD = price * (1.0 - big)

                pmt_MB = amortized_payment(loan_MB, float(apr_pct) / 100.0, term_int)
                pmt_BD = amortized_payment(loan_BD, float(apr_pct) / 100.0, term_int)
                bd_monthly_invest = max(0.0, pmt_MB - pmt_BD)  # BD invests monthly savings vs MB
                mb_initial_invest = extra_down                       # MB invests upfront difference

                bal_MB = mb_initial_invest
                bal_BD = 0.0

                for m in range(months_int):
                    mf = month_factor_at(m)
                    # BD contributes at start-of-month (to match Pay Cash convention)
                    if m < term_int:
                        bal_BD += bd_monthly_invest
                    bal_BD *= mf
                    # MB: no monthly contributions; only growth
                    bal_MB *= mf

                start_date = str(rec["dates"][s].date())
                end_row = s + months_int - 1
                end_date = str(rec["dates"][end_row].date()) if end_row < len(rec["dates"]) else start_date

                # Remaining principals after horizon
                rem_MB = remaining_principal(principal=loan_MB, annual_rate=float(apr_pct) / 100.0, term_months=term_int, months_elapsed=months_int)
                rem_BD = remaining_principal(principal=loan_BD, annual_rate=float(apr_pct) / 100.0, term_months=term_int, months_elapsed=months_int)

                # Net worths
                nw_MB = bal_MB - rem_MB
                nw_BD = bal_BD - rem_BD
                advantage_mb_minus_bd = nw_MB - nw_BD

                rows.append({
                    "mode": "Borrow More vs Bigger Down",
                    "start_date": start_date,
                    "end_date": end_date,
                    "networth_borrow_more": nw_MB,
                    "networth_bigger_down": nw_BD,
                    "remaining_principal_MB": rem_MB,
                    "remaining_principal_BD": rem_BD,
                    "advantage_borrow_more_minus_bigger_down": advantage_mb_minus_bd,
                    "monthly_payment_MB": pmt_MB,
                    "monthly_payment_BD": pmt_BD,
                    "mb_initial_invest": mb_initial_invest,
                    "bd_monthly_invest": bd_monthly_invest,
                })

        df = pd.DataFrame(rows)

        # ---- Summary + Charts per mode ----
        if mode == "Pay Cash vs Borrow":
            adv = df["advantage_paycash_minus_borrow"].values
            prob_pc_wins = float((adv > 0).mean()) if len(adv) else float("nan")
            p10, p50, p90 = (np.percentile(adv, q) for q in (10, 50, 90)) if len(adv) else (float("nan"),)*3

            st.subheader("Probability & Percentiles (Pay Cash advantage (Net Worth) = PayCash − Borrow; Borrow is net of remaining mortgage)")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Pay Cash wins", f"{prob_pc_wins:.1%}")
            s2.metric("Borrow wins", f"{(1 - prob_pc_wins):.1%}")
            s3.metric("P10 (Advantage)", f"${p10:,.0f}")
            s4.metric("P50 / Median (Advantage)", f"${p50:,.0f}")
            s5.metric("P90 (Advantage)", f"${p90:,.0f}")

            median_adv = float(p50)
            def _fmt_money(x: float) -> str:
                sign = "-" if x < 0 else ""
                return f"{sign}${abs(x):,.0f}"
            typical_line = f"Typically (median), paying cash ends **{_fmt_money(abs(median_adv))} {'ahead of' if median_adv > 0 else 'behind'}** borrowing."
            prob_line = f"Across all historical start dates tested, paying cash wins **{prob_pc_wins:.1%}** of the time; borrowing wins **{1-prob_pc_wins:.1%}**."

            st.markdown(
                """
                ### What this means in plain English
                - **What we’re comparing:** Two portfolios. **Pay Cash** invests the monthly mortgage payment; **Borrow** invests the full loan amount now.
                - **How to read the numbers:** We compare **net worth**. If the advantage is positive, *Pay Cash* finished higher; if negative, *Borrow* finished higher (its investments **minus** the remaining mortgage).
                {prob_line}
                {typical_line}
                """.format(prob_line=prob_line, typical_line=typical_line)
            )
            st.caption(
                "We treat the upfront tax as an opportunity cost: if paying cash requires selling to net the principal, we assume the **gross** amount (principal / (1 − tax%)) would have otherwise stayed invested in the Borrow path. "
                "Comparisons are made on **net worth** at the horizon (Borrow = investments minus remaining mortgage)."
            )

            st.subheader("Per-Start Net Worth")
            st.dataframe(df, use_container_width=True)
            st.line_chart(df.set_index("start_date")[["networth_paycash", "networth_borrow"]])

            st.download_button(
                "Download Per-Start Rows (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="cash_vs_borrow_per_start.csv",
                mime="text/csv",
            )

        else:
            adv = df["advantage_borrow_more_minus_bigger_down"].values
            prob_mb_wins = float((adv > 0).mean()) if len(adv) else float("nan")
            p10, p50, p90 = (np.percentile(adv, q) for q in (10, 50, 90)) if len(adv) else (float("nan"),)*3

            st.subheader("Probability & Percentiles (Borrow More advantage (Net Worth) = MB − BD; each net of remaining mortgage)")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Borrow More wins", f"{prob_mb_wins:.1%}")
            s2.metric("Bigger Down wins", f"{(1 - prob_mb_wins):.1%}")
            s3.metric("P10 (Advantage)", f"${p10:,.0f}")
            s4.metric("P50 / Median (Advantage)", f"${p50:,.0f}")
            s5.metric("P90 (Advantage)", f"${p90:,.0f}")

            def _fmt_money(x: float) -> str:
                sign = "-" if x < 0 else ""
                return f"{sign}${abs(x):,.0f}"
            typical_line = f"Typically (median), borrowing more ends **{_fmt_money(abs(p50))} {'ahead of' if p50 > 0 else 'behind'}** a bigger down payment."
            prob_line = f"Across all historical start dates tested, borrowing more wins **{prob_mb_wins:.1%}** of the time; bigger down wins **{1-prob_mb_wins:.1%}**."

            st.markdown(
                """
                ### What this means in plain English
                - **What we’re comparing:** **Borrow More (smaller down)** invests the **up-front difference** on day 0; **Bigger Down** invests **monthly payment savings** vs Borrow More.
                - **How to read the numbers:** We compare **net worth** for each: investments **minus** remaining mortgage at the horizon.
                {prob_line}
                {typical_line}
                """.format(prob_line=prob_line, typical_line=typical_line)
            )

            st.subheader("Per-Start Net Worth")
            st.dataframe(df, use_container_width=True)
            st.line_chart(df.set_index("start_date")[["networth_borrow_more", "networth_bigger_down"]])

            st.download_button(
                "Download Per-Start Rows (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="borrow_more_vs_bigger_down_per_start.csv",
                mime="text/csv",
            )

with st.expander("Peek at first 12 annual inputs (raw)"):
    try:
        _a_preview = pd.Series(rec["annual_col"].astype(float)[:12], name="annual_col_raw")
    except Exception:
        _a_preview = pd.Series([], name="annual_col_raw")
    st.write(_a_preview)