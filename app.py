import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from io import StringIO  # for CSV template generation (not strictly required but ok)

# =========================
# Page setup
# =========================
st.set_page_config(page_title="BMS + Cell Analyzer", layout="wide")

# =========================
# Simple login using Streamlit secrets
# =========================
def login():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # If already logged in, just return
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
        else:
            st.error("Invalid username or password.")

    # If not logged in yet, stop the app here
    if not st.session_state.logged_in:
        st.stop()

# Call login before showing the main app
login()

# =========================
# Downloadable templates
# =========================
st.markdown("### Download sample upload templates")

# ---- BMS template (raw units like your real files) ----
bms_template = pd.DataFrame(
    [
        {
            "Time": "2025-11-11 10:00:00",
            "Stack voltage": 3450,      # 345.0 V after /10
            "Stack current": -120,      # -12.0 A after /10 (discharge)
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
        {
            "Time": "2025-11-11 10:00:20",
            "Stack voltage": 3445,
            "Stack current": -115,
            "SOC": 501,
            "MAX CELL": 3347,
            "MIN CELL": 3317,
            "2nd MAX CELL": 3342,
            "2nd MIN CELL": 3322,
        },
    ]
)
bms_template_csv = bms_template.to_csv(index=False)

# ---- Cell rack template (Time, Serial number, V1..V396, in mV) ----
cell_cols = ["Time", "Serial number"] + [f"V{i}" for i in range(1, 397)]
cell_rows = []

base_time_str = "2025-11-11 10:00:00"
base_serial = 5613

row0 = {"Time": base_time_str, "Serial number": base_serial}
for i in range(1, 397):
    row0[f"V{i}"] = 3350
cell_rows.append(row0)

row1 = {"Time": "None", "Serial number": base_serial + 1}
for i in range(1, 397):
    row1[f"V{i}"] = 3348
cell_rows.append(row1)

row2 = {"Time": "None", "Serial number": base_serial + 2}
for i in range(1, 397):
    row2[f"V{i}"] = 3345
cell_rows.append(row2)

cell_template = pd.DataFrame(cell_rows, columns=cell_cols)
cell_template_csv = cell_template.to_csv(index=False)

c_t1, c_t2 = st.columns(2)
with c_t1:
    st.download_button(
        label="⬇ Download BMS log template (CSV)",
        data=bms_template_csv,
        file_name="bms_template.csv",
        mime="text/csv",
        key="download_bms_template",
    )
with c_t2:
    st.download_button(
        label="⬇ Download rack cell data template (CSV)",
        data=cell_template_csv,
        file_name="cell_record_template.csv",
        mime="text/csv",
        key="download_cell_template",
    )

st.caption(
    "- BMS template uses the **raw units** your app expects: Stack V /10, Current /10, SOC /10, Cells in mV.\n"
    "- Cell template uses `Time`, `Serial number`, and `V1..V396` in **mV** (e.g., 3350 = 3.350 V)."
)

# =========================
# Sidebar uploads
# =========================
st.sidebar.header("📁 Upload Data")

bms_file = st.sidebar.file_uploader(
    "BMS pack-level log (.csv, .xlsx, .xls)",
    type=["csv", "xlsx", "xls"],
    key="bms_file",
)

cell_file = st.sidebar.file_uploader(
    "Cell-level combined data (.csv, .xlsx, .xls) – optional for single-file mode",
    type=["csv", "xlsx", "xls"],
    key="cell_file",
)

# =========================
# Constants for BMS analysis
# =========================
MIN_CELL_CRIT = 2.50   # V
SOC_CRIT = 5.0         # %
DELTA_WARN = 0.30      # V
IDLE_CURR = 2.0        # A

# =========================
# BMS data preparation
# =========================
bms_df = None
bms_error = None
bms_has_2nd_max = False
bms_has_2nd_min = False

if bms_file is not None:
    fname = bms_file.name.lower()
    try:
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(bms_file)
        else:
            df = pd.read_csv(bms_file)
    except Exception as e:
        bms_error = f"❌ Error reading BMS file: {e}"
    else:
        if df.empty:
            bms_error = "❌ BMS file has no rows of data."
        else:
            df.columns = df.columns.str.strip()

            required_cols = [
                "Time",
                "Stack voltage",
                "Stack current",
                "SOC",
                "MAX CELL",
                "MIN CELL",
            ]
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

                # Scaling (your rules)
                df["Stack voltage"] = pd.to_numeric(df["Stack voltage"], errors="coerce") / 10
                df["Stack current"] = pd.to_numeric(df["Stack current"], errors="coerce") / 10
                df["SOC"] = pd.to_numeric(df["SOC"], errors="coerce") / 10

                df["MAX CELL"] = pd.to_numeric(df["MAX CELL"], errors="coerce") / 1000
                df["MIN CELL"] = pd.to_numeric(df["MIN CELL"], errors="coerce") / 1000

                if bms_has_2nd_max:
                    df["2nd MAX CELL"] = pd.to_numeric(df["2nd MAX CELL"], errors="coerce") / 1000
                if bms_has_2nd_min:
                    df["2nd MIN CELL"] = pd.to_numeric(df["2nd MIN CELL"], errors="coerce") / 1000

                # Time & numeric cleanup
                time_col = "Time"
                pack_v_col = "Stack voltage"
                current_col = "Stack current"
                soc_col = "SOC"
                max_cell_v_col = "MAX CELL"
                min_cell_v_col = "MIN CELL"
                second_max_col = "2nd MAX CELL" if bms_has_2nd_max else None
                second_min_col = "2nd MIN CELL" if bms_has_2nd_min else None

                df["__time__"] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.dropna(subset=["__time__"])
                df = df.sort_values("__time__")

                numeric_cols = [pack_v_col, current_col, soc_col, max_cell_v_col, min_cell_v_col]
                if bms_has_2nd_max:
                    numeric_cols.append(second_max_col)
                if bms_has_2nd_min:
                    numeric_cols.append(second_min_col)

                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna(subset=[pack_v_col, max_cell_v_col, min_cell_v_col])

                if df.empty:
                    bms_error = (
                        "❌ After cleaning BMS data, no valid rows remain with voltage values."
                    )
                else:
                    df["cell_delta"] = df[max_cell_v_col] - df[min_cell_v_col]
                    bms_df = df

# =========================
# Cell combined data (mode 1)
# =========================
cell_df_raw = None
cell_error = None

if cell_file is not None:
    cfname = cell_file.name.lower()
    try:
        if cfname.endswith((".xlsx", ".xls")):
            cell_df_raw = pd.read_excel(cell_file)
        else:
            cell_df_raw = pd.read_csv(cell_file)
    except Exception as e:
        cell_error = f"❌ Error reading cell-level file: {e}"
    else:
        if cell_df_raw.empty:
            cell_error = "❌ Cell-level file has no rows of data."
        else:
            cell_df_raw.columns = cell_df_raw.columns.str.strip()

# =========================
# Tabs
# =========================
tab_bms_overview, tab_bms_energy, tab_cells = st.tabs(
    ["BMS Overview", "BMS Energy", "Cell Detail"]
)

# =================================================================
# BMS OVERVIEW TAB
# =================================================================
with tab_bms_overview:
    st.subheader("BMS Pack-Level Overview")

    if bms_file is None and bms_df is None:
        st.info("Upload a **BMS log** in the sidebar to use this section.")
    elif bms_error is not None:
        st.error(bms_error)
        if bms_df is not None:
            st.dataframe(bms_df.head(50))
    elif bms_df is not None:
        df = bms_df
        pack_v_col = "Stack voltage"
        current_col = "Stack current"
        soc_col = "SOC"
        max_cell_v_col = "MAX CELL"
        min_cell_v_col = "MIN CELL"
        second_max_col = "2nd MAX CELL" if bms_has_2nd_max else None
        second_min_col = "2nd MIN CELL" if bms_has_2nd_min else None

        st.markdown("### Raw BMS Data Preview")
        st.dataframe(df.head(50))
        st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.markdown("### Key Metrics")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Pack Voltage Max", f"{df[pack_v_col].max():.2f} V")
            st.metric("Pack Voltage Min", f"{df[pack_v_col].min():.2f} V")

        with c2:
            st.metric("Min Cell Voltage (Abs Min)", f"{df[min_cell_v_col].min():.3f} V")
            st.metric("Max Cell Voltage (Abs Max)", f"{df[max_cell_v_col].max():.3f} V")

        with c3:
            st.metric("Max Charge Current", f"{df[current_col].max():.1f} A")
            st.metric("Max Discharge Current", f"{df[current_col].min():.1f} A")

        with c4:
            st.metric(
                "SoC Range",
                f"{df[soc_col].min():.1f} % → {df[soc_col].max():.1f} %",
            )

        st.markdown("### Basic Checks")

        issues = []
        min_cell_min = df[min_cell_v_col].min()
        soc_min = df[soc_col].min()
        delta_max = df["cell_delta"].max()

        if min_cell_min < MIN_CELL_CRIT:
            issues.append(f"⚠️ Min cell < {MIN_CELL_CRIT:.2f} V (lowest: {min_cell_min:.3f} V).")

        if soc_min < SOC_CRIT:
            issues.append(f"⚠️ SoC < {SOC_CRIT:.1f}% (lowest: {soc_min:.1f} %).")

        if delta_max > DELTA_WARN:
            issues.append(f"⚠️ Cell delta > {DELTA_WARN:.2f} V (max: {delta_max:.3f} V).")

        if not issues:
            st.success("✅ No basic issues detected with current thresholds.")
        else:
            for msg in issues:
                st.warning(msg)

        st.markdown("---")
        st.markdown("### Trends Over Time")

        # Stack voltage
        fig_pack = px.line(
            df,
            x="__time__",
            y=pack_v_col,
            title="Stack Voltage Over Time",
        )
        fig_pack.update_layout(xaxis_title="Time", yaxis_title="Voltage (V)")
        st.plotly_chart(fig_pack, use_container_width=True)

        # Cell voltages
        y_cols = [min_cell_v_col, max_cell_v_col]
        legend_names = {
            min_cell_v_col: "MIN CELL",
            max_cell_v_col: "MAX CELL",
        }
        if second_min_col is not None:
            y_cols.append(second_min_col)
            legend_names[second_min_col] = "2nd MIN CELL"
        if second_max_col is not None:
            y_cols.append(second_max_col)
            legend_names[second_max_col] = "2nd MAX CELL"

        fig_cells = px.line(
            df,
            x="__time__",
            y=y_cols,
            title="Cell Voltages Over Time",
        )
        fig_cells.update_layout(xaxis_title="Time", yaxis_title="Cell Voltage (V)")
        fig_cells.for_each_trace(lambda t: t.update(name=legend_names.get(t.name, t.name)))
        st.plotly_chart(fig_cells, use_container_width=True)

        # Cell delta
        fig_delta = px.line(
            df,
            x="__time__",
            y="cell_delta",
            title="Cell Voltage Delta (MAX - MIN)",
        )
        fig_delta.update_layout(xaxis_title="Time", yaxis_title="Delta (V)")
        st.plotly_chart(fig_delta, use_container_width=True)

        # Current
        fig_current = px.line(
            df,
            x="__time__",
            y=current_col,
            title="Stack Current Over Time",
        )
        fig_current.update_layout(xaxis_title="Time", yaxis_title="Current (A)")
        st.plotly_chart(fig_current, use_container_width=True)

        # SoC
        fig_soc = px.line(
            df,
            x="__time__",
            y=soc_col,
            title="State of Charge (SoC) Over Time",
        )
        fig_soc.update_layout(xaxis_title="Time", yaxis_title="SoC (%)")
        st.plotly_chart(fig_soc, use_container_width=True)


# =================================================================
# BMS ENERGY TAB
# =================================================================
with tab_bms_energy:
    st.subheader("Energy In / Out (from BMS log)")

    if bms_df is None:
        if bms_error:
            st.error(bms_error)
        else:
            st.info("Upload a **BMS log** in the sidebar to compute energy.")
    else:
        df = bms_df
        pack_v_col = "Stack voltage"
        current_col = "Stack current"

        t_min = df["__time__"].min()
        t_max = df["__time__"].max()

        start_t, end_t = st.select_slider(
            "Select time range",
            options=list(df["__time__"]),
            value=(t_min, t_max),
        )

        if start_t >= end_t:
            st.warning("Start time must be before end time.")
        else:
            df_energy = df[(df["__time__"] >= start_t) & (df["__time__"] <= end_t)].copy()

            if len(df_energy) < 2:
                st.warning("Not enough points in this time range to compute energy.")
            else:
                df_energy["dt_h"] = df_energy["__time__"].shift(-1) - df_energy["__time__"]
                df_energy["dt_h"] = df_energy["dt_h"].dt.total_seconds().fillna(0) / 3600.0

                df_energy["power_kW"] = df_energy[pack_v_col] * df_energy[current_col] / 1000.0
                df_energy["dE_kWh"] = df_energy["power_kW"] * df_energy["dt_h"]

                energy_out_kWh = df_energy.loc[df_energy["power_kW"] > 0, "dE_kWh"].sum()
                energy_in_kWh = -df_energy.loc[df_energy["power_kW"] < 0, "dE_kWh"].sum()
                net_energy_kWh = energy_out_kWh - energy_in_kWh

                c1, c2, c3 = st.columns(3)
                c1.metric("Energy OUT (Discharge)", f"{energy_out_kWh:.2f} kWh")
                c2.metric("Energy IN (Charge)", f"{energy_in_kWh:.2f} kWh")
                c3.metric("Net Energy (OUT - IN)", f"{net_energy_kWh:.2f} kWh")

                st.caption(
                    f"Time window: {start_t.strftime('%Y-%m-%d %H:%M:%S')} → "
                    f"{end_t.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"({(end_t - start_t).total_seconds()/3600.0:.2f} hours)"
                )

                fig_pwr = px.line(
                    df_energy,
                    x="__time__",
                    y="power_kW",
                    title="Power (kW) in Selected Range",
                )
                fig_pwr.update_layout(xaxis_title="Time", yaxis_title="Power (kW)")
                st.plotly_chart(fig_pwr, use_container_width=True)

                with st.expander("Show calculation table (first 200 rows)"):
                    st.dataframe(
                        df_energy[
                            ["__time__", pack_v_col, current_col, "power_kW", "dt_h", "dE_kWh"]
                        ].rename(
                            columns={
                                "__time__": "Time",
                                pack_v_col: "Stack Voltage (V)",
                                current_col: "Stack Current (A)",
                                "power_kW": "Power (kW)",
                                "dt_h": "Δt (h)",
                                "dE_kWh": "ΔE (kWh)",
                            }
                        ).head(200)
                    )

                # -------------------------------------------------
                # Theoretical container energy from LFP cell specs
                # -------------------------------------------------
                st.markdown("---")
                st.subheader("Theoretical Container Energy (LFP OCV Model)")

                # ---- User inputs ----
                cell_Ah = st.number_input(
                    "Cell capacity (Ah)",
                    min_value=0.1,
                    max_value=2000.0,
                    value=280.0,
                    step=10.0,
                    help="Nameplate Ah rating of one LFP cell",
                    key="energy_cell_Ah",
                )

                cells_per_string = st.number_input(
                    "Cells in series per string",
                    min_value=1,
                    max_value=2000,
                    value=396,
                    step=1,
                    help="For example: 396 cells in series in one rack/string",
                    key="energy_cells_per_string",
                )

                num_strings = st.number_input(
                    "Number of parallel strings / racks",
                    min_value=1,
                    max_value=100,
                    value=4,
                    step=1,
                    help="For example: number of racks tied in parallel on the DC bus",
                    key="energy_num_strings",
                )

                charge_cutoff = st.number_input(
                    "Charge cutoff voltage per cell (V)",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.6,
                    step=0.01,
                    key="energy_charge_cutoff",
                )

                discharge_cutoff = st.number_input(
                    "Discharge cutoff voltage per cell (V)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.7,
                    step=0.01,
                    key="energy_discharge_cutoff",
                )

                weak_cells = st.number_input(
                    "Number of weak cells to bypass/remove",
                    min_value=0,
                    max_value=1000,
                    value=1,
                    step=1,
                    help="If one or more cells are effectively unusable and must be removed from the energy budget.",
                    key="energy_weak_cells",
                )

                # ---- Approximate LFP OCV curve (SoC in [0,1]) ----
                # Points: (SoC, Voltage [V])
                lfp_curve = [
                    (0.00, 2.50),
                    (0.05, 3.00),
                    (0.10, 3.20),
                    (0.20, 3.25),
                    (0.80, 3.28),
                    (0.90, 3.30),
                    (0.95, 3.35),
                    (1.00, 3.60),
                ]

                def ocv_lfp(soc: float) -> float:
                    """Piecewise-linear OCV(soc) for LFP."""
                    if soc <= 0.0:
                        return lfp_curve[0][1]
                    if soc >= 1.0:
                        return lfp_curve[-1][1]
                    for (s1, v1), (s2, v2) in zip(lfp_curve, lfp_curve[1:]):
                        if s1 <= soc <= s2:
                            t = (soc - s1) / (s2 - s1)
                            return v1 + t * (v2 - v1)
                    return lfp_curve[-1][1]

                def find_soc_for_voltage(v_target: float, steps: int = 2000):
                    """
                    Find SOC in [0,1] where OCV(SOC) ≈ v_target by scanning and interpolating.
                    Returns None if not found.
                    """
                    s_prev = 0.0
                    v_prev = ocv_lfp(s_prev)
                    for i in range(1, steps + 1):
                        s = i / steps
                        v = ocv_lfp(s)
                        if (v_prev - v_target) * (v - v_target) <= 0:
                            if v == v_prev:
                                return s
                            t = (v_target - v_prev) / (v - v_prev)
                            s_cross = s_prev + t * (s - s_prev)
                            return max(0.0, min(1.0, s_cross))
                        s_prev, v_prev = s, v
                    return None

                def integrate_ocv(soc_min: float, soc_max: float, steps: int = 1000) -> float:
                    """
                    Numerically integrate OCV(soc) d(soc) from soc_min to soc_max
                    using midpoint rule. Returns ∫ V dSOC (unit: V * fraction_of_SOC).
                    """
                    if soc_max <= soc_min:
                        return 0.0
                    ds = (soc_max - soc_min) / steps
                    total = 0.0
                    for i in range(steps):
                        s_mid = soc_min + (i + 0.5) * ds
                        total += ocv_lfp(s_mid)
                    return total * ds

                # ---- Map voltage cutoffs to SOC window using OCV curve ----
                if charge_cutoff <= discharge_cutoff:
                    st.warning("Charge cutoff must be higher than discharge cutoff.")
                elif cell_Ah <= 0 or cells_per_string <= 0 or num_strings <= 0:
                    st.warning("Please enter positive values for cell Ah, cell count, and number of strings.")
                else:
                    soc_min = find_soc_for_voltage(discharge_cutoff)
                    soc_max = find_soc_for_voltage(charge_cutoff)

                    if soc_min is None:
                        soc_min = 0.0
                    if soc_max is None:
                        soc_max = 1.0

                    if soc_max <= soc_min:
                        st.warning(
                            "Could not determine a valid SOC window from the specified cutoffs "
                            "using the LFP OCV curve. Using full 0–100% SOC as fallback."
                        )
                        soc_min, soc_max = 0.0, 1.0

                    usable_soc_pct = (soc_max - soc_min) * 100.0
                    integral_VdSOC = integrate_ocv(soc_min, soc_max, steps=1000)
                    avg_voltage = integral_VdSOC / (soc_max - soc_min) if soc_max > soc_min else 0.0

                    # per-cell usable energy in Wh
                    energy_cell_Wh = cell_Ah * integral_VdSOC

                    total_cells = cells_per_string * num_strings
                    effective_cells = max(total_cells - weak_cells, 0)

                    full_energy_kWh = energy_cell_Wh * total_cells / 1000.0
                    weak_energy_kWh = energy_cell_Wh * effective_cells / 1000.0

                    shortfall_kWh = full_energy_kWh - weak_energy_kWh
                    shortfall_pct = (shortfall_kWh / full_energy_kWh * 100.0) if full_energy_kWh > 0 else 0.0

                    st.markdown(
                        f"Estimated usable SOC window from LFP OCV curve: **{usable_soc_pct:.1f}%** "
                        f"({soc_min*100:.1f}% → {soc_max*100:.1f}%).  "
                        f"Average cell voltage over this window ≈ **{avg_voltage:.3f} V**."
                    )

                    cE1, cE2, cE3 = st.columns(3)
                    cE1.metric(
                        "Full theoretical energy (all cells healthy)",
                        f"{full_energy_kWh:.1f} kWh",
                    )
                    cE2.metric(
                        "Energy if weak cell(s) removed",
                        f"{weak_energy_kWh:.1f} kWh",
                    )
                    cE3.metric(
                        "Shortfall vs full design",
                        f"{shortfall_kWh:.1f} kWh ({shortfall_pct:.2f} %)",
                    )

                    st.caption(
                        "Based on an approximate **LFP OCV–SoC curve**, integrated between the SOC levels "
                        "corresponding to your per-cell voltage cutoffs. "
                        "This is still an approximation (real cells differ with temperature, rate, ageing), "
                        "but it is more realistic than a simple linear voltage model."
                    )

# =================================================================
# 🧩 CELL DETAIL TAB
# =================================================================
with tab_cells:
    st.subheader("Cell-Level Detail by Rack")

    st.write(
        "Upload one file **per rack**. Each file should follow this format:\n"
        "- Columns: `Time`, `Serial number`, `V1..Vn` (e.g., V1–V396)\n"
        "- Cell voltages in **mV** (e.g., 3350 = 3.350 V)\n"
        "- First valid `Time` + `Serial number` row is used as the base timestamp; "
        "other rows use `Serial number` as seconds offset."
    )

    st.markdown("### Separate file per rack (each file like `Time, Serial number, V1..V396`)")

    num_racks = st.number_input(
        "How many racks do you want to upload?",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        key="num_racks",
    )

    st.markdown("Upload files for each rack:")

    rack_configs = []
    for i in range(num_racks):
        col1, col2 = st.columns(2)
        with col1:
            rack_name = st.text_input(
                f"Rack {i+1} name",
                value=f"RACK{i:02d}",
                key=f"multi_rack_name_{i}",
            )
        with col2:
            rack_file = st.file_uploader(
                f"File for Rack {i+1}",
                type=["csv", "xlsx", "xls"],
                key=f"multi_rack_file_{i}",
            )
        rack_configs.append((rack_name, rack_file))

    st.markdown("### ⚙ Value Scaling for Cell Voltage")
    scale_mV = st.checkbox(
        "Cell voltage is stored in mV (e.g., 3313 = 3.313 V) → divide by 1000",
        value=True,
        key="multi_scale_mV",
    )

    # Helper to get cell index from 'V1', 'V2', ...
    def cell_index(name: str) -> int:
        try:
            return int(name[1:])
        except Exception:
            return 0

    combined_cells_snap = []
    time_series_list = []

    for rack_name, rack_file in rack_configs:
        if rack_file is None:
            continue

        try:
            fname = rack_file.name.lower()
            if fname.endswith((".xlsx", ".xls")):
                df_r = pd.read_excel(rack_file)
            else:
                df_r = pd.read_csv(rack_file)
        except Exception as e:
            st.warning(f"Could not read file for rack '{rack_name}': {e}")
            continue

        df_r.columns = df_r.columns.str.strip()

        # Must have these columns in your template
        if "Serial number" not in df_r.columns or "Time" not in df_r.columns:
            st.warning(
                f"Rack '{rack_name}': need 'Time' and 'Serial number' columns. "
                "Skipping this file."
            )
            continue

        # Detect V1..Vn columns
        v_cols = [c for c in df_r.columns if c.upper().startswith("V")]
        if not v_cols:
            st.warning(
                f"Rack '{rack_name}' file has no V1/V2/... columns. Skipping this file."
            )
            continue

        v_cols = sorted(v_cols, key=cell_index)

        # -------- SNAPSHOT: last valid row --------
        df_valid = df_r.dropna(subset=v_cols, how="all")
        if df_valid.empty:
            st.warning(
                f"Rack '{rack_name}' file has no valid cell readings in V1..Vn columns."
            )
        else:
            last_row = df_valid.iloc[-1]
            cell_values = last_row[v_cols].astype(float)

            snap_df = pd.DataFrame(
                {
                    "Rack": rack_name,
                    "CellID": [cell_index(c) for c in v_cols],
                    "__cell_v__": cell_values.values,
                }
            )

            if scale_mV:
                snap_df["__cell_v__"] = snap_df["__cell_v__"] / 1000.0

            snap_df["__temp__"] = None
            snap_df = snap_df.dropna(subset=["__cell_v__"])
            if not snap_df.empty:
                combined_cells_snap.append(snap_df)

        # -------- TIME SERIES: build real Time from Time + Serial number --------
        df_r["Serial number"] = pd.to_numeric(df_r["Serial number"], errors="coerce")
        df_r = df_r.dropna(subset=["Serial number"])
        if df_r.empty:
            st.warning(
                f"Rack '{rack_name}': 'Serial number' column has no valid numeric data."
            )
            continue
        df_r = df_r.sort_values("Serial number")

        df_r["Time_str"] = df_r["Time"].astype(str).str.strip()
        mask_valid_time = (df_r["Time_str"].str.lower() != "none") & (df_r["Time_str"] != "")
        valid_time_rows = df_r[mask_valid_time]

        if valid_time_rows.empty:
            st.warning(
                f"Rack '{rack_name}': 'Time' column has no valid timestamps. "
                "Cannot build V vs time; skipping this rack."
            )
            continue

        base_time_str = valid_time_rows["Time_str"].iloc[0]
        base_time = pd.to_datetime(base_time_str, errors="coerce")
        base_serial = valid_time_rows["Serial number"].iloc[0]

        if pd.isna(base_time) or pd.isna(base_serial):
            st.warning(
                f"Rack '{rack_name}': could not parse base Time / Serial number. "
                "Skipping this rack."
            )
            continue

        df_r["Time_calc"] = base_time + pd.to_timedelta(
            df_r["Serial number"] - base_serial, unit="s"
        )

        df_long = df_r.melt(
            id_vars=["Time_calc"],
            value_vars=v_cols,
            var_name="CellID",
            value_name="Voltage_mV",
        )

        df_long.rename(columns={"Time_calc": "Time"}, inplace=True)
        df_long["Rack"] = rack_name
        df_long["CellID"] = df_long["CellID"].str.replace("V", "").astype(int)

        if scale_mV:
            df_long["Voltage_V"] = pd.to_numeric(df_long["Voltage_mV"], errors="coerce") / 1000.0
        else:
            df_long["Voltage_V"] = pd.to_numeric(df_long["Voltage_mV"], errors="coerce")

        df_long = df_long.dropna(subset=["Voltage_V"])
        if not df_long.empty:
            time_series_list.append(df_long)

    # ----- Combined snapshot stats -----
    if not combined_cells_snap:
        st.info("No valid snapshot data found from rack files.")
    else:
        df_cells_all = pd.concat(combined_cells_snap, ignore_index=True)

        st.subheader("Combined Cell Snapshot (Last Valid Row per Rack)")
        st.dataframe(df_cells_all.head(50))
        st.caption(
            f"Total rows: {df_cells_all.shape[0]} across "
            f"{df_cells_all['Rack'].nunique()} racks."
        )

    # ----- Time series: V vs time + full-range per-rack stats -----
    st.subheader("📈 All Cells: V vs Time")

    if not time_series_list:
        st.info(
            "Upload rack files with a valid 'Time' (first row) + 'Serial number' + V1..Vn "
            "to see V vs Time plots and full-range stats."
        )
    else:
        df_cells_time = pd.concat(time_series_list, ignore_index=True)

        st.subheader("Per-Rack Statistics (Full Time Range)")

        grp_ts = df_cells_time.groupby("Rack")

        avg_v = grp_ts["Voltage_V"].mean().rename("Avg_Cell_V")
        delta_v = (grp_ts["Voltage_V"].max() - grp_ts["Voltage_V"].min()).rename("Cell_V_Delta")
        idx_min = grp_ts["Voltage_V"].idxmin()
        idx_max = grp_ts["Voltage_V"].idxmax()

        df_min = (
            df_cells_time.loc[idx_min, ["Rack", "CellID", "Time", "Voltage_V"]]
            .rename(
                columns={
                    "CellID": "Min_Cell_ID",
                    "Time": "Min_Time",
                    "Voltage_V": "Min_Cell_V",
                }
            )
        )
        df_max = (
            df_cells_time.loc[idx_max, ["Rack", "CellID", "Time", "Voltage_V"]]
            .rename(
                columns={
                    "CellID": "Max_Cell_ID",
                    "Time": "Max_Time",
                    "Voltage_V": "Max_Cell_V",
                }
            )
        )

        rack_stats_full = (
            avg_v.to_frame()
            .join(delta_v)
            .reset_index()
            .merge(df_min, on="Rack")
            .merge(df_max, on="Rack")
        )

        rack_stats_full = rack_stats_full[
            [
                "Rack",
                "Min_Cell_ID",
                "Min_Cell_V",
                "Min_Time",
                "Max_Cell_ID",
                "Max_Cell_V",
                "Max_Time",
                "Avg_Cell_V",
                "Cell_V_Delta",
            ]
        ]

        st.dataframe(
            rack_stats_full.style.format(
                {
                    "Min_Cell_V": "{:.3f}",
                    "Max_Cell_V": "{:.3f}",
                    "Avg_Cell_V": "{:.3f}",
                    "Cell_V_Delta": "{:.3f}",
                }
            )
        )
        st.caption(
            "Min / Max are taken over the **entire file** (all timestamps) per rack. "
            "Min/Max time shows when that extreme cell voltage occurred."
        )

        racks = ["All Racks"] + sorted(df_cells_time["Rack"].unique())
        selected_rack = st.selectbox(
            "Select rack for time-series plot",
            racks,
            key="ts_rack_sel",
        )

        df_plot = df_cells_time
        if selected_rack != "All Racks":
            df_plot = df_plot[df_plot["Rack"] == selected_rack]

        st.write(
            f"Plotting **{df_plot['CellID'].nunique()} cells**, "
            f"{df_plot.shape[0]:,} points for {selected_rack}."
        )

        fig_ts = px.line(
            df_plot,
            x="Time",
            y="Voltage_V",
            color="CellID",
            line_group="CellID",
            title=f"Cell Voltages vs Time ({selected_rack})",
            render_mode="webgl",
        )
        fig_ts.update_layout(
            height=700,
            xaxis_title="Time",
            yaxis_title="Cell Voltage (V)",
            legend_title="Cell",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
