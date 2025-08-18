if st.button("▶ Compute Ending Values for All Start Periods"):
    raw = rec["col3"].astype(float)
    factors = returns_to_factors(raw) if mode == "Monthly returns" else raw

    max_start = len(factors) - int(months)
    if max_start < 0:
        st.error(f"Not enough months for {int(months)}. Available: {len(factors)}.")
    else:
        rows = []
        contrib_float = float(contrib)
        months_int = int(months)
        for start in range(0, max_start + 1):
            bal = 0.0
            # Loop over this window
            for i in range(months_int):
                bal += contrib_float              # start-of-month contribution
                bal *= float(factors[start + i])  # then apply that month's factor
            start_date = str(rec["dates"][start].date())
            end_date = str(rec["dates"][start + months_int - 1].date())
            total_contrib = contrib_float * months_int
            gain = bal - total_contrib
            rows.append({
                "start_date": start_date,
                "end_date": end_date,
                "ending_value": bal,
                "total_contributions": total_contrib,
                "gain_over_contributions": gain
            })
        df = pd.DataFrame(rows)

        st.subheader("Ending Value per Starting Month")
        st.dataframe(df, use_container_width=True)

        # Optional quick viz of ending values
        st.line_chart(df.set_index("start_date")["ending_value"])

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV of All Start Periods",
            data=csv,
            file_name="ending_values_all_start_periods.csv",
            mime="text/csv",
        )

        # Show first 12 raw & interpreted factors for transparency
        with st.expander("Show first 12 values (raw & interpreted factors)"):
            nshow = min(12, int(months))
            st.write(pd.DataFrame({
                "raw_col3": raw[:nshow],
                "interpreted_factor": factors[:nshow]
            }))