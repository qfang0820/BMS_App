import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

# =========================
# Page setup
# =========================
st.set_page_config(page_title="BMS + Cell Analyzer", layout="wide")

# =========================
# Simple login using Streamlit secrets
# =========================
def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    st.title("BMS Analyzer Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        correct_user = st.secrets.get("APP_USERNAME", "")
        correct_pass = st.secrets.get("APP_PASSWORD", "")

        if username == correct_user and password == correct_pass:
            st.session_state.logged_in = True
            st.success("Login successful. Loading app...")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    if not st.session_state.logged_in:
        st.stop()

login()

st.title("BMS + Cell Analyzer")

# =========================
# Helpers (robust parsing that WILL work with commas/units/text)
# =========================
def read_any_table(uploaded_file) -> pd.DataFrame:
    fname = uploaded_file.name.lower()
    if fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric parser:
    - '3,450' -> 3450
    - '3450mV' -> 3450
    - '3.45 V' -> 3.45
    - 'SOC=503' -> 503
    - keeps +/- sign
    """
    if s is None:
        return pd.Series(dtype="float64")

    s = s.astype(str).str.strip()
    s = s.str.replace("\u00a0", " ", regex=False)   # NBSP
    s = s.str.replace(",", "", regex=False)        # thousands separators

    extracted = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")

def cell_index(name: str) -> int:
    try:
        return int(str(name)[1:])
    except Exception:
        return 0

# =========================
# Downloadable templates
# =========================
st.markdown("### Download sample upload templates")

# ---- BMS template (raw units) ----
bms_template = pd.DataFrame(
    [
        {
            "Time": "2025-11-11 10:00:00",
            "Stack voltage": 3450,      # 345.0 V after /10
            "Stack current": -120,      # -12.0 A after /10
            "SOC": 503,                 # 50.3 % after /10
            "MAX CELL": 3350,           # 3.350 V after /1000
            "MIN CELL": 3320,           # 3.320 V after /1000
            "2nd MAX CELL": 3345,
            "2nd MIN CELL": 3325,
        },
        {
            "Time": "2025-11-11 10:00:10",
            "Stack voltage": 3448,
            "Stack current": -118,
            "SOC": 502,
            "MAX CELL": 3348,
            "MIN CELL": 3318,
            "2nd MAX CELL": 3343,
            "2nd MIN CELL": 3323,
        },
    ]
).to_csv(index=False)

# ---- Cell rack template (Time, Serial number, V1..V396, in mV) ----
cell_cols = ["Time", "Serial number"] + [f"V{i}" for i in range(1, 397)]
base_time_str = "2025-11-11 10:00:00"
base_serial = 5613
row0 = {"Time": base_time_str, "Serial number": base_serial, **{f"V{i}": 3350 for i in range(1, 397)}}
row1 = {"Time": "None", "Serial number": base_serial + 1, **{f"V{i}": 3348 for i in range(1, 397)}}
row2 = {"Time": "None", "Serial number": base_serial + 2, **{f"V{i}": 3345 for i in range(1, 397)}}
cell_template = pd.DataFrame([row0, row1, row2], columns=cell_cols).to_csv(index=False)

# ---- Rack level template (raw units; only a few required) ----
rack_template = pd.DataFrame(
    [
        {
            "Time": "2025-11-05 16:20:50",
            "BCMU ID": 1,
            "Total voltage": 13091,              # 1309.1 V after /10
            "Current": 0,                        # 0.0 A after /10
            "SOC(0.1%)": 926,                    # 92.6 % after /10 (optional)
            "Average voltage": 3312,             # 3.312 V after /1000 (optional)
            "Highest cell voltage": 3315,        # 3.315 V (optional)
            "Highest cell voltage position": 148,
            "Lowest cell voltage": 3310,         # 3.310 V (optional)
            "Lowest cell voltage position": 105,
            "Cell voltage difference": 5,        # 0.005 V (optional)
        },
        {
            "Time": "2025-11-05 16:21:55",
            "BCMU ID": 2,
            "Total voltage": 13100,
            "Current": -15,                      # -1.5 A after /10 (optional)
            "SOC(0.1%)": 920,
            "Average voltage": 3310,
            "Highest cell voltage": 3316,
            "Highest cell voltage position": 102,
            "Lowest cell voltage": 3308,
            "Lowest cell voltage position": 221,
            "Cell voltage difference": 8,
        },
    ]
).to_csv(index=False)

t1, t2, t3 = st.columns(3)
with t1:
    st.download_button("⬇ Download BMS log template (CSV)", bms_template, "bms_template.csv", "text/csv", key="dl_bms")
with t2:
    st.download_button("⬇ Download rack cell data template (CSV)", cell_template, "cell_record_template.csv", "text/csv", key="dl_cell")
with t3:
    st.download_button("⬇ Download rack-level log template (CSV)", rack_template, "rack_level_template.csv", "text/csv", key="dl_rack")

st.caption(
    "- This app supports messy values like `3,450`, `3450mV`, `3.45 V`, `SOC=503`.\n"
    "- Use the checkboxes in Upload Data if your file is already in engineering units."
)

# =========================
# Sidebar: Navigation always visible + Upload collapsible
# =========================
st.sidebar.header("📍 Navigation")

nav_options = ["BMS Overview", "Energy", "Rack Level", "Rack Energy", "Cell Detail"]
selection = st.sidebar.radio(
    "Navigation Tree",
    nav_options,
    label_visibility="collapsed",
    key="nav_tree_selection",
    format_func=lambda x: f"⠀⠀• {x}" if x in ("Energy", "Rack Energy") else x,
)

main_page = None
bms_subpage = None
rack_subpage = None

if selection == "BMS Overview":
    main_page = "BMS"
    bms_subpage = "Overview"
elif selection == "Energy":
    main_page = "BMS"
    bms_subpage = "Energy"
elif selection == "Rack Level":
    main_page = "RACK"
    rack_subpage = "Overview"
elif selection == "Rack Energy":
    main_page = "RACK"
    rack_subpage = "Energy"
elif selection == "Cell Detail":
    main_page = "CELL"

st.sidebar.markdown("---")

with st.sidebar.expander("📁 Upload Data", expanded=False):
    st.markdown("### BMS (Pack-level)")
    bms_file = st.file_uploader(
        "Upload BMS log (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="upload_bms_file",
    )
    bms_eng_units = st.checkbox(
        "BMS already in V / A / % / V (no scaling)",
        value=False,
        key="bms_eng_units",
        help="If checked: Stack voltage/current/SOC and MAX/MIN CELL are already engineering units.",
    )

    st.markdown("---")
    st.markdown("### Rack-level")
    rack_level_file = st.file_uploader(
        "Upload Rack-level log (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="upload_rack_level_file",
    )
    rack_eng_units = st.checkbox(
        "Rack-level already in V / A / % / V (no scaling)",
        value=False,
        key="rack_eng_units",
        help="If checked: Total voltage/current and optional cell voltages are already engineering units.",
    )

    st.markdown("---")
    st.caption("Rack cell files are uploaded inside **Cell Detail** page (one file per rack).")

# =========================
# BMS data preparation (ROBUST)
# =========================
bms_df = None
bms_error = None
bms_has_2nd_max = False
bms_has_2nd_min = False

if bms_file is not None:
    try:
        df = read_any_table(bms_file)
    except Exception as e:
        bms_error = f"❌ Error reading BMS file: {e}"
    else:
        if df.empty:
            bms_error = "❌ BMS file has no rows of data."
        else:
            df.columns = df.columns.astype(str).str.strip()

            required_cols = ["Time", "Stack voltage", "Stack current", "SOC", "MAX CELL", "MIN CELL"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                bms_error = (
                    "❌ BMS file is missing required columns:\n"
                    + "\n".join(f"- {c}" for c in missing)
                    + "\n\nColumns found:\n"
                    + ", ".join(df.columns.astype(str))
                )
            else:
                bms_has_2nd_max = "2nd MAX CELL" in df.columns
                bms_has_2nd_min = "2nd MIN CELL" in df.columns

                # time
                df["__time__"] = pd.to_datetime(df["Time"], errors="coerce")
                df = df.dropna(subset=["__time__"]).sort_values("__time__")

                # numeric parse (robust)
                df["Stack voltage"] = clean_numeric_series(df["Stack voltage"])
                df["Stack current"] = clean_numeric_series(df["Stack current"])
                df["SOC"] = clean_numeric_series(df["SOC"])
                df["MAX CELL"] = clean_numeric_series(df["MAX CELL"])
                df["MIN CELL"] = clean_numeric_series(df["MIN CELL"])
                if bms_has_2nd_max:
                    df["2nd MAX CELL"] = clean_numeric_series(df["2nd MAX CELL"])
                if bms_has_2nd_min:
                    df["2nd MIN CELL"] = clean_numeric_series(df["2nd MIN CELL"])

                # scale if raw
                if not bms_eng_units:
                    df["Stack voltage"] = df["Stack voltage"] / 10.0
                    df["Stack current"] = df["Stack current"] / 10.0
                    df["SOC"] = df["SOC"] / 10.0
                    df["MAX CELL"] = df["MAX CELL"] / 1000.0
                    df["MIN CELL"] = df["MIN CELL"] / 1000.0
                    if bms_has_2nd_max:
                        df["2nd MAX CELL"] = df["2nd MAX CELL"] / 1000.0
                    if bms_has_2nd_min:
                        df["2nd MIN CELL"] = df["2nd MIN CELL"] / 1000.0

                # keep rows with mandatory numbers
                df = df.dropna(subset=["Stack voltage", "MAX CELL", "MIN CELL"])
                if df.empty:
                    bms_error = (
                        "❌ After parsing, no valid rows remain.\n\n"
                        "Open the diagnostics below to see which column(s) are not parsing."
                    )
                else:
                    df["cell_delta"] = df["MAX CELL"] - df["MIN CELL"]
                    bms_df = df

# =========================
# Rack-level data preparation (ROBUST; only 4 mandatory columns)
# =========================
rack_df = None
rack_error = None

if rack_level_file is not None:
    try:
        rdf = read_any_table(rack_level_file)
    except Exception as e:
        rack_error = f"❌ Error reading rack-level file: {e}"
    else:
        if rdf.empty:
            rack_error = "❌ Rack-level file has no rows of data."
        else:
            rdf.columns = rdf.columns.astype(str).str.strip()

            required_rack = ["Time", "BCMU ID", "Total voltage", "Current"]
            missing_r = [c for c in required_rack if c not in rdf.columns]
            if missing_r:
                rack_error = (
                    "❌ Rack-level file is missing required columns:\n"
                    + "\n".join(f"- {c}" for c in missing_r)
                    + "\n\nColumns found:\n"
                    + ", ".join(rdf.columns.astype(str))
                )
            else:
                rdf["__time__"] = pd.to_datetime(rdf["Time"], errors="coerce")
                rdf = rdf.dropna(subset=["__time__"]).sort_values("__time__")

                rdf["BCMU ID"] = clean_numeric_series(rdf["BCMU ID"])
                rdf["Total voltage"] = clean_numeric_series(rdf["Total voltage"])
                rdf["Current"] = clean_numeric_series(rdf["Current"])

                if not rack_eng_units:
                    rdf["Total voltage"] = rdf["Total voltage"] / 10.0
                    rdf["Current"] = rdf["Current"] / 10.0

                # Optional SOC
                if "SOC" in rdf.columns:
                    rdf["SOC"] = clean_numeric_series(rdf["SOC"])
                elif "SOC(0.1%)" in rdf.columns:
                    rdf["SOC"] = clean_numeric_series(rdf["SOC(0.1%)"])
                    if not rack_eng_units:
                        rdf["SOC"] = rdf["SOC"] / 10.0

                # Optional mV columns -> V if raw
                opt_mv_cols = ["Average voltage", "Highest cell voltage", "Lowest cell voltage", "Cell voltage difference"]
                for col in opt_mv_cols:
                    if col in rdf.columns:
                        rdf[col] = clean_numeric_series(rdf[col])
                        if not rack_eng_units:
                            rdf[col] = rdf[col] / 1000.0

                # Optional position cols
                for col in ["Highest cell voltage position", "Lowest cell voltage position"]:
                    if col in rdf.columns:
                        rdf[col] = clean_numeric_series(rdf[col])

                rdf = rdf.dropna(subset=["BCMU ID", "Total voltage", "Current"])
                if rdf.empty:
                    rack_error = "❌ After parsing, no valid rack-level rows remain."
                else:
                    rack_df = rdf

# ============================================================
# PAGES
# ============================================================

# =========================
# BMS Overview / Energy
# =========================
if main_page == "BMS":
    if bms_subpage == "Overview":
        st.subheader("BMS Pack-Level Overview")

        if bms_df is None:
            if bms_error:
                st.error(bms_error)
            else:
                st.info("Upload a **BMS log** in the sidebar to use this section.")
        else:
            df = bms_df

            st.markdown("### Key Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Pack Voltage Max", f"{df['Stack voltage'].max():.2f} V")
                st.metric("Pack Voltage Min", f"{df['Stack voltage'].min():.2f} V")
            with m2:
                st.metric("Min Cell Voltage (Abs Min)", f"{df['MIN CELL'].min():.3f} V")
                st.metric("Max Cell Voltage (Abs Max)", f"{df['MAX CELL'].max():.3f} V")
            with m3:
                st.metric("Max Charge Current", f"{df['Stack current'].max():.1f} A")
                st.metric("Max Discharge Current", f"{df['Stack current'].min():.1f} A")
            with m4:
                st.metric("SoC Range", f"{df['SOC'].min():.1f}% → {df['SOC'].max():.1f}%")

            st.markdown("---")
            st.markdown("### Trends Over Time")

            st.plotly_chart(
                px.line(df, x="__time__", y="Stack voltage", title="Stack Voltage Over Time")
                .update_layout(xaxis_title="Time", yaxis_title="Voltage (V)"),
                use_container_width=True,
            )

            y_cols = ["MIN CELL", "MAX CELL"]
            legend_names = {"MIN CELL": "MIN CELL", "MAX CELL": "MAX CELL"}
            if bms_has_2nd_min:
                y_cols.append("2nd MIN CELL")
                legend_names["2nd MIN CELL"] = "2nd MIN CELL"
            if bms_has_2nd_max:
                y_cols.append("2nd MAX CELL")
                legend_names["2nd MAX CELL"] = "2nd MAX CELL"

            fig_cells = px.line(df, x="__time__", y=y_cols, title="Cell Voltages Over Time")
            fig_cells.update_layout(xaxis_title="Time", yaxis_title="Cell Voltage (V)")
            fig_cells.for_each_trace(lambda t: t.update(name=legend_names.get(t.name, t.name)))
            st.plotly_chart(fig_cells, use_container_width=True)

            st.plotly_chart(
                px.line(df, x="__time__", y="cell_delta", title="Cell Voltage Delta (MAX - MIN)")
                .update_layout(xaxis_title="Time", yaxis_title="Delta (V)"),
                use_container_width=True,
            )

            st.plotly_chart(
                px.line(df, x="__time__", y="Stack current", title="Stack Current Over Time")
                .update_layout(xaxis_title="Time", yaxis_title="Current (A)"),
                use_container_width=True,
            )

            st.plotly_chart(
                px.line(df, x="__time__", y="SOC", title="State of Charge (SoC) Over Time")
                .update_layout(xaxis_title="Time", yaxis_title="SoC (%)"),
                use_container_width=True,
            )

            with st.expander("🧪 BMS parsing diagnostics", expanded=False):
                st.write("Engineering-units mode:", bool(bms_eng_units))
                st.write("Valid%:")
                st.write(
                    {
                        "Time": float(df["__time__"].notna().mean() * 100.0),
                        "Stack voltage": float(df["Stack voltage"].notna().mean() * 100.0),
                        "Stack current": float(df["Stack current"].notna().mean() * 100.0),
                        "SOC": float(df["SOC"].notna().mean() * 100.0),
                        "MAX CELL": float(df["MAX CELL"].notna().mean() * 100.0),
                        "MIN CELL": float(df["MIN CELL"].notna().mean() * 100.0),
                    }
                )
                st.dataframe(df[["Time", "__time__", "Stack voltage", "Stack current", "SOC", "MAX CELL", "MIN CELL"]].head(30))

    elif bms_subpage == "Energy":
        st.subheader("BMS Energy (from BMS log)")

        if bms_df is None:
            if bms_error:
                st.error(bms_error)
            else:
                st.info("Upload a **BMS log** in the sidebar to compute energy.")
        else:
            df = bms_df.copy()
            t_min, t_max = df["__time__"].min(), df["__time__"].max()

            start_t, end_t = st.select_slider(
                "Select time range",
                options=list(df["__time__"]),
                value=(t_min, t_max),
                key="bms_energy_slider",
            )

            if start_t >= end_t:
                st.warning("Start time must be before end time.")
            else:
                dfe = df[(df["__time__"] >= start_t) & (df["__time__"] <= end_t)].copy()
                if dfe["__time__"].nunique() < 2:
                    st.warning("Not enough points in this time range to compute energy.")
                else:
                    dfe = dfe.sort_values("__time__")
                    dfe["dt_h"] = (dfe["__time__"].shift(-1) - dfe["__time__"]).dt.total_seconds().fillna(0) / 3600.0
                    dfe["power_kW"] = dfe["Stack voltage"] * dfe["Stack current"] / 1000.0
                    dfe["dE_kWh"] = dfe["power_kW"] * dfe["dt_h"]

                    e_out = dfe.loc[dfe["power_kW"] > 0, "dE_kWh"].sum()
                    e_in = -dfe.loc[dfe["power_kW"] < 0, "dE_kWh"].sum()
                    e_net = e_out - e_in

                    a, b, c = st.columns(3)
                    a.metric("Energy OUT (Discharge)", f"{e_out:.2f} kWh")
                    b.metric("Energy IN (Charge)", f"{e_in:.2f} kWh")
                    c.metric("Net Energy (OUT - IN)", f"{e_net:.2f} kWh")

                    st.plotly_chart(
                        px.line(dfe, x="__time__", y="power_kW", title="Power (kW) in Selected Range")
                        .update_layout(xaxis_title="Time", yaxis_title="Power (kW)"),
                        use_container_width=True,
                    )

                    with st.expander("Show calculation table (first 200 rows)"):
                        st.dataframe(
                            dfe[["__time__", "Stack voltage", "Stack current", "power_kW", "dt_h", "dE_kWh"]]
                            .rename(
                                columns={
                                    "__time__": "Time",
                                    "Stack voltage": "Stack Voltage (V)",
                                    "Stack current": "Stack Current (A)",
                                    "power_kW": "Power (kW)",
                                    "dt_h": "Δt (h)",
                                    "dE_kWh": "ΔE (kWh)",
                                }
                            )
                            .head(200)
                        )

# =========================
# Rack Level / Rack Energy
# =========================
elif main_page == "RACK":
    st.subheader("Rack-Level Analysis (BCMU separated)")

    if rack_df is None:
        if rack_error:
            st.error(rack_error)
        else:
            st.info("Upload a **rack-level log** in the sidebar to use this section.")
    else:
        df_all = rack_df.copy()
        bcmu_ids = sorted(df_all["BCMU ID"].dropna().unique())

        if rack_subpage == "Overview":
            for i, bcmu in enumerate(bcmu_ids):
                df = df_all[df_all["BCMU ID"] == bcmu].copy()
                if df.empty:
                    continue
                if i > 0:
                    st.markdown("---")

                st.markdown(f"### BCMU ID {int(bcmu)}")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Voltage Max", f"{df['Total voltage'].max():.2f} V")
                    st.metric("Total Voltage Min", f"{df['Total voltage'].min():.2f} V")
                with c2:
                    st.metric("Current Max", f"{df['Current'].max():.2f} A")
                    st.metric("Current Min", f"{df['Current'].min():.2f} A")
                with c3:
                    st.metric("SOC Range", f"{df['SOC'].min():.1f}% → {df['SOC'].max():.1f}%" if "SOC" in df.columns else "N/A")
                with c4:
                    st.metric("ΔCellV Max", f"{df['Cell voltage difference'].max():.3f} V" if "Cell voltage difference" in df.columns else "N/A")
                    st.metric("ΔCellV Min", f"{df['Cell voltage difference'].min():.3f} V" if "Cell voltage difference" in df.columns else "N/A")

                st.plotly_chart(
                    px.line(df, x="__time__", y="Total voltage", title=f"Total Voltage (BCMU {int(bcmu)})")
                    .update_layout(xaxis_title="Time", yaxis_title="Voltage (V)"),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.line(df, x="__time__", y="Current", title=f"Current (BCMU {int(bcmu)})")
                    .update_layout(xaxis_title="Time", yaxis_title="Current (A)"),
                    use_container_width=True,
                )

                if "Average voltage" in df.columns:
                    st.plotly_chart(
                        px.line(df, x="__time__", y="Average voltage", title=f"Average Cell Voltage (BCMU {int(bcmu)})")
                        .update_layout(xaxis_title="Time", yaxis_title="Voltage (V)"),
                        use_container_width=True,
                    )

                if "Highest cell voltage" in df.columns and "Lowest cell voltage" in df.columns:
                    st.plotly_chart(
                        px.line(
                            df,
                            x="__time__",
                            y=["Highest cell voltage", "Lowest cell voltage"],
                            title=f"Highest/Lowest Cell Voltage (BCMU {int(bcmu)})",
                        ).update_layout(xaxis_title="Time", yaxis_title="Cell Voltage (V)", legend_title="Series"),
                        use_container_width=True,
                    )

                if "Cell voltage difference" in df.columns:
                    st.plotly_chart(
                        px.line(df, x="__time__", y="Cell voltage difference", title=f"Cell Voltage Difference (BCMU {int(bcmu)})")
                        .update_layout(xaxis_title="Time", yaxis_title="Delta (V)"),
                        use_container_width=True,
                    )

        elif rack_subpage == "Energy":
            st.markdown("### Rack Energy (Per BCMU)")

            bcmu_choice_list = ["All BCMU"] + list(bcmu_ids)
            selected_bcmu = st.selectbox(
                "Select BCMU ID for energy analysis",
                bcmu_choice_list,
                format_func=lambda x: "All BCMU" if x == "All BCMU" else f"BCMU {int(x)}",
                key="rack_energy_bcmu_select",
            )

            df_base = df_all.copy() if selected_bcmu == "All BCMU" else df_all[df_all["BCMU ID"] == selected_bcmu].copy()

            if df_base.empty or df_base["__time__"].nunique() < 2:
                st.warning("Not enough data points for energy calculation.")
            else:
                t_min, t_max = df_base["__time__"].min(), df_base["__time__"].max()
                start_t, end_t = st.select_slider(
                    "Select time range",
                    options=list(df_base["__time__"]),
                    value=(t_min, t_max),
                    key="rack_energy_slider",
                )

                if start_t >= end_t:
                    st.warning("Start time must be before end time.")
                else:
                    df_window = df_base[(df_base["__time__"] >= start_t) & (df_base["__time__"] <= end_t)].copy()

                    # SINGLE BCMU
                    if selected_bcmu != "All BCMU":
                        df_b = df_window.sort_values("__time__").copy()
                        if df_b["__time__"].nunique() < 2:
                            st.info("Not enough points in this time window.")
                        else:
                            df_b["dt_h"] = (df_b["__time__"].shift(-1) - df_b["__time__"]).dt.total_seconds().fillna(0) / 3600.0
                            df_b["power_kW"] = df_b["Total voltage"] * df_b["Current"] / 1000.0
                            df_b["dE_kWh"] = df_b["power_kW"] * df_b["dt_h"]

                            e_out = df_b.loc[df_b["power_kW"] > 0, "dE_kWh"].sum()
                            e_in = -df_b.loc[df_b["power_kW"] < 0, "dE_kWh"].sum()
                            e_net = e_out - e_in

                            a, b, c = st.columns(3)
                            a.metric("Energy OUT (Discharge)", f"{e_out:.2f} kWh")
                            b.metric("Energy IN (Charge)", f"{e_in:.2f} kWh")
                            c.metric("Net Energy (OUT - IN)", f"{e_net:.2f} kWh")

                            st.plotly_chart(
                                px.line(df_b, x="__time__", y="power_kW", title=f"Rack Power (kW) - BCMU {int(selected_bcmu)}")
                                .update_layout(xaxis_title="Time", yaxis_title="Power (kW)"),
                                use_container_width=True,
                            )

                            with st.expander("Show energy calculation table (first 200 rows)"):
                                st.dataframe(
                                    df_b[["__time__", "Total voltage", "Current", "power_kW", "dt_h", "dE_kWh"]]
                                    .rename(
                                        columns={
                                            "__time__": "Time",
                                            "Total voltage": "Rack Voltage (V)",
                                            "Current": "Rack Current (A)",
                                            "power_kW": "Power (kW)",
                                            "dt_h": "Δt (h)",
                                            "dE_kWh": "ΔE (kWh)",
                                        }
                                    )
                                    .head(200)
                                )

                    # ALL BCMU (FIXED: compute dt per BCMU)
                    else:
                        bcmu_ids2 = sorted(df_window["BCMU ID"].dropna().unique())
                        summary_rows = []
                        frames = []

                        for bcmu in bcmu_ids2:
                            df_b = df_window[df_window["BCMU ID"] == bcmu].copy().sort_values("__time__")
                            if df_b.empty or df_b["__time__"].nunique() < 2:
                                continue

                            df_b["dt_h"] = (df_b["__time__"].shift(-1) - df_b["__time__"]).dt.total_seconds().fillna(0) / 3600.0
                            df_b["power_kW"] = df_b["Total voltage"] * df_b["Current"] / 1000.0
                            df_b["dE_kWh"] = df_b["power_kW"] * df_b["dt_h"]

                            e_out = df_b.loc[df_b["power_kW"] > 0, "dE_kWh"].sum()
                            e_in = -df_b.loc[df_b["power_kW"] < 0, "dE_kWh"].sum()
                            e_net = e_out - e_in

                            summary_rows.append(
                                {
                                    "BCMU ID": int(bcmu),
                                    "Energy OUT (kWh)": e_out,
                                    "Energy IN (kWh)": e_in,
                                    "Net Energy (kWh)": e_net,
                                }
                            )
                            frames.append(df_b)

                        if not summary_rows:
                            st.info("No BCMU has enough valid points in this window.")
                        else:
                            st.markdown("#### Energy Summary per BCMU in Selected Window")
                            df_summary = pd.DataFrame(summary_rows).sort_values("BCMU ID")
                            st.dataframe(
                                df_summary.style.format(
                                    {"Energy OUT (kWh)": "{:.2f}", "Energy IN (kWh)": "{:.2f}", "Net Energy (kWh)": "{:.2f}"}
                                )
                            )

                            df_all_energy = pd.concat(frames, ignore_index=True)
                            st.plotly_chart(
                                px.line(
                                    df_all_energy,
                                    x="__time__",
                                    y="power_kW",
                                    color="BCMU ID",
                                    title="Rack Power (kW) for All BCMU",
                                ).update_layout(xaxis_title="Time", yaxis_title="Power (kW)", legend_title="BCMU ID"),
                                use_container_width=True,
                            )

                            with st.expander("Show combined energy calculation table (first 200 rows)"):
                                st.dataframe(
                                    df_all_energy[["__time__", "BCMU ID", "Total voltage", "Current", "power_kW", "dt_h", "dE_kWh"]]
                                    .rename(
                                        columns={
                                            "__time__": "Time",
                                            "Total voltage": "Rack Voltage (V)",
                                            "Current": "Rack Current (A)",
                                            "power_kW": "Power (kW)",
                                            "dt_h": "Δt (h)",
                                            "dE_kWh": "ΔE (kWh)",
                                        }
                                    )
                                    .head(200)
                                )

# =========================
# Cell Detail
# =========================
elif main_page == "CELL":
    st.subheader("Cell-Level Detail by Rack")

    st.write(
        "Upload one file **per rack**.\n"
        "- Columns: `Time`, `Serial number`, `V1..Vn` (e.g., V1–V396)\n"
        "- If your cell voltages are stored in **mV**, check the box to divide by 1000.\n"
        "- First valid `Time` row sets base timestamp; others use (Serial number offset in seconds)."
    )

    num_racks = st.number_input("How many racks do you want to upload?", 1, 20, 4, 1, key="num_racks")

    rack_configs = []
    for i in range(num_racks):
        c1, c2 = st.columns(2)
        with c1:
            rack_name = st.text_input(f"Rack {i+1} name", value=f"RACK{i:02d}", key=f"cell_rack_name_{i}")
        with c2:
            rack_file = st.file_uploader(f"File for Rack {i+1}", type=["csv", "xlsx", "xls"], key=f"cell_rack_file_{i}")
        rack_configs.append((rack_name, rack_file))

    scale_mV = st.checkbox("Cell voltage stored in mV → divide by 1000", True, key="cell_scale_mv")

    combined_cells_snap = []
    time_series_list = []

    for rack_name, rack_file in rack_configs:
        if rack_file is None:
            continue

        try:
            df_r = read_any_table(rack_file)
        except Exception as e:
            st.warning(f"Could not read file for rack '{rack_name}': {e}")
            continue

        df_r.columns = df_r.columns.astype(str).str.strip()

        if "Serial number" not in df_r.columns or "Time" not in df_r.columns:
            st.warning(f"Rack '{rack_name}': need 'Time' and 'Serial number'. Skipping.")
            continue

        v_cols = [c for c in df_r.columns if str(c).upper().startswith("V")]
        if not v_cols:
            st.warning(f"Rack '{rack_name}': no V columns. Skipping.")
            continue

        # baseline time: first row where Time is not None/empty
        time_str = df_r["Time"].astype(str).str.strip()
        mask_valid_time = (time_str.str.lower() != "none") & (time_str != "")
        if not mask_valid_time.any():
            st.warning(f"Rack '{rack_name}': no valid base Time found. Skipping.")
            continue

        base_idx = df_r[mask_valid_time].index[0]
        base_time = pd.to_datetime(df_r.loc[base_idx, "Time"], errors="coerce")
        if pd.isna(base_time):
            st.warning(f"Rack '{rack_name}': base Time cannot parse. Skipping.")
            continue

        df_r["Serial number"] = clean_numeric_series(df_r["Serial number"])
        base_serial = df_r.loc[base_idx, "Serial number"]
        if pd.isna(base_serial):
            st.warning(f"Rack '{rack_name}': base Serial number cannot parse. Skipping.")
            continue

        df_r = df_r.dropna(subset=["Serial number"])
        df_r["calculated_time"] = base_time + pd.to_timedelta(df_r["Serial number"] - base_serial, unit="s")
        df_r = df_r.sort_values("calculated_time")

        # parse voltages robustly
        for c in v_cols:
            df_r[c] = clean_numeric_series(df_r[c])

        if scale_mV:
            df_r[v_cols] = df_r[v_cols] / 1000.0

        # snapshot last row
        if not df_r.empty:
            last = df_r.iloc[-1]
            for col in v_cols:
                combined_cells_snap.append(
                    {"Rack": rack_name, "Cell Index": cell_index(col), "Voltage": last[col], "Time": last["calculated_time"]}
                )

        df_r["Rack"] = rack_name
        time_series_list.append(df_r)

    st.markdown("---")

    if not combined_cells_snap:
        st.info("Upload rack cell files to see heatmap / stats.")
    else:
        df_snap = pd.DataFrame(combined_cells_snap).dropna(subset=["Voltage"]).sort_values(["Rack", "Cell Index"])

        st.subheader("Latest Snapshot Heatmap (All Racks)")
        fig_heat = px.density_heatmap(
            df_snap,
            x="Cell Index",
            y="Rack",
            z="Voltage",
            color_continuous_scale="RdYlGn",
            title=f"Cell Voltages at End of Log (Approx Time: {df_snap['Time'].max()})",
        )
        fig_heat.update_layout(xaxis_title="Cell Index", yaxis_title="Rack ID")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Rack Statistics (Snapshot)")
        stats = df_snap.groupby("Rack")["Voltage"].agg(["min", "max", "mean", "count"]).reset_index()
        stats["delta"] = stats["max"] - stats["min"]
        st.dataframe(stats.style.format({"min": "{:.3f}", "max": "{:.3f}", "mean": "{:.3f}", "delta": "{:.3f}"}))

        st.subheader("Delta (Max - Min) per Rack")
        st.plotly_chart(px.bar(stats, x="Rack", y="delta", title="Voltage Imbalance (Delta) by Rack"), use_container_width=True)

    st.subheader("📈 All Cells: V vs Time")
    if not time_series_list:
        st.info("Upload rack files to see V vs time.")
    else:
        df_cells = pd.concat(time_series_list, ignore_index=True)
        v_cols_all = [c for c in df_cells.columns if str(c).upper().startswith("V")]
        if not v_cols_all:
            st.info("No V columns found for plotting.")
        else:
            df_long = df_cells.melt(
                id_vars=["Rack", "calculated_time"],
                value_vars=v_cols_all,
                var_name="CellID",
                value_name="Voltage_V",
            )
            df_long["CellID"] = df_long["CellID"].apply(cell_index)
            df_long = df_long.dropna(subset=["Voltage_V"])

            racks = ["All Racks"] + sorted(df_long["Rack"].unique())
            selected_rack = st.selectbox("Select rack for time-series plot", racks, key="cell_ts_rack")
            df_plot = df_long if selected_rack == "All Racks" else df_long[df_long["Rack"] == selected_rack]

            st.write(
                f"Plotting **{df_plot['CellID'].nunique()} cells**, "
                f"{df_plot.shape[0]:,} points for {selected_rack}."
            )

            fig_ts = px.line(
                df_plot,
                x="calculated_time",
                y="Voltage_V",
                color="CellID",
                line_group="CellID",
                title=f"Cell Voltages vs Time ({selected_rack})",
                render_mode="webgl",
            )
            fig_ts.update_layout(height=700, xaxis_title="Time", yaxis_title="Cell Voltage (V)", legend_title="Cell")
            st.plotly_chart(fig_ts, use_container_width=True)
