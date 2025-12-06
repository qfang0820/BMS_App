import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from io import StringIO

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
        # Defaults to empty string if secrets are not set up
        correct_user = st.secrets.get("APP_USERNAME", "")
        correct_pass = st.secrets.get("APP_PASSWORD", "")

        # If secrets aren't set, this might allow login with empty fields 
        # (Change logic if you want to enforce specific hardcoded values)
        if username == correct_user and password == correct_pass:
            st.session_state.logged_in = True
            st.success("Login successful. Loading app...")
            st.rerun() # Rerun to refresh the page and show the app
        else:
            st.error("Invalid username or password.")

    # If not logged in yet, stop the app here
    if not st.session_state.logged_in:
        st.stop()

# Call login before showing the main app
login()

st.title("BMS + Cell Analyzer")

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
# Sidebar: Navigation + Upload
# =========================
st.sidebar.header("📍 Navigation")

# ── Top-level pages ─────────────────────────────
# Label hidden so it doesn't show "Main section"
main_page = st.sidebar.radio(
    "",
    ["BMS Overview", "Cell Detail"],
    index=0,
    key="main_page",
    label_visibility="collapsed",
)

# Little helper text to look like your tree
if main_page == "BMS Overview":
    # Indented title for sub-pages
    st.sidebar.markdown("&nbsp;&nbsp;&nbsp;**BMS Overview**", unsafe_allow_html=True)
    bms_subpage = st.sidebar.radio(
        "",
        ["Overview", "Energy"],
        index=0,
        key="bms_subpage",
        label_visibility="collapsed",
    )
else:
    bms_subpage = None  # not used on Cell Detail page

st.sidebar.markdown("---")

# ── File upload (collapsible) ───────────────────
with st.sidebar.expander("📁 Upload Data", expanded=False):
    bms_file = st.file_uploader(
        "BMS pack-level log (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="bms_file",
    )

    cell_file = st.file_uploader(
        "Cell-level combined data (.csv, .xlsx, .xls) – (currently unused)",
        type=["csv", "xlsx", "xls"],
        key="cell_file",
    )

    st.caption(
        "Upload BMS logs for pack analysis, and rack-level files on the **Cell Detail** page."
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
# (Unused) cell_file prep – kept for future
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

# =================================================================
# MAIN SECTION: BMS OVERVIEW
# =================================================================
if main_page == "BMS Overview":

    if bms_subpage == "Overview":
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

    elif bms_subpage == "Energy":
        st.subheader("BMS Energy")

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
# MAIN SECTION: CELL DETAIL
# =================================================================
elif main_page == "Cell Detail":
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

        # -----------------------------------------------------------
        # Logic: Reconstruct Time based on Serial Number offset
        # -----------------------------------------------------------
        # We need at least one valid time string to start
        valid_time_idx = df_r["Time"].first_valid_index()
        
        if valid_time_idx is None:
            st.warning(f"Rack '{rack_name}': No valid 'Time' found to establish a baseline.")
            continue

        try:
            base_time = pd.to_datetime(df_r.loc[valid_time_idx, "Time"])
            base_serial = float(df_r.loc[valid_time_idx, "Serial number"])
        except Exception as e:
            st.warning(f"Rack '{rack_name}': Error parsing base time/serial: {e}")
            continue

        # Convert serial to numeric just in case
        df_r["Serial number"] = pd.to_numeric(df_r["Serial number"], errors="coerce").fillna(0)

        # Calculate offset in seconds from the base_serial
        # offset = (Current Serial - Base Serial)
        df_r["time_offset_s"] = df_r["Serial number"] - base_serial
        df_r["calculated_time"] = base_time + pd.to_timedelta(df_r["time_offset_s"], unit="s")

        # Sort by time
        df_r = df_r.sort_values("calculated_time")

        # -----------------------------------------------------------
        # Logic: Scaling and numeric conversion
        # -----------------------------------------------------------
        # Ensure voltage columns are numeric
        for c in v_cols:
            df_r[c] = pd.to_numeric(df_r[c], errors="coerce")

        if scale_mV:
            df_r[v_cols] = df_r[v_cols] / 1000.0

        # -----------------------------------------------------------
        # Logic: Store Snapshot (Last valid row)
        # -----------------------------------------------------------
        if not df_r.empty:
            last_row = df_r.iloc[-1]
            
            # Create a "long" format for this rack's last timestamp
            # We want: RackName, CellIndex, Voltage
            for col in v_cols:
                idx = cell_index(col)
                val = last_row[col]
                combined_cells_snap.append({
                    "Rack": rack_name,
                    "Cell Index": idx,
                    "Voltage": val,
                    "Time": last_row["calculated_time"]
                })

            # Store full dataframe for time-series (optional usage)
            df_r["Rack"] = rack_name
            time_series_list.append(df_r)

    # =========================
    # Visualizations
    # =========================
    st.markdown("---")
    
    if not combined_cells_snap:
        st.info("Upload rack files to see the cell heatmap.")
    else:
        df_snap = pd.DataFrame(combined_cells_snap)
        
        # 1. Heatmap
        st.subheader("Latest Snapshot Heatmap (All Racks)")
        
        # Pivot for heatmap: Index=Rack, Columns=Cell Index, Values=Voltage
        # We sort Racks and Cell Indices to ensure order
        df_snap = df_snap.sort_values(by=["Rack", "Cell Index"])
        
        fig_heat = px.density_heatmap(
            df_snap,
            x="Cell Index",
            y="Rack",
            z="Voltage",
            color_continuous_scale="RdYlGn", # Red-Yellow-Green
            text_auto=False,
            title=f"Cell Voltages at End of Log (Approx Time: {df_snap['Time'].max()})"
        )
        fig_heat.update_layout(
            xaxis_title="Cell Index", 
            yaxis_title="Rack ID"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # 2. Statistics Table
        st.subheader("Rack Statistics (Snapshot)")
        stats = df_snap.groupby("Rack")["Voltage"].agg(['min', 'max', 'mean', 'count']).reset_index()
        stats["delta"] = stats["max"] - stats["min"]
        st.dataframe(stats.style.format({
            "min": "{:.3f}", 
            "max": "{:.3f}", 
            "mean": "{:.3f}", 
            "delta": "{:.3f}"
        }))

        # 3. Min/Max Bar Chart
        st.subheader("Delta (Max - Min) per Rack")
        fig_bar = px.bar(
            stats, 
            x="Rack", 
            y="delta", 
            title="Voltage Imbalance (Delta) by Rack",
            color="delta",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
