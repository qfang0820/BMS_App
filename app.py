import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="BMS Analyzer", layout="wide")
st.title("BMS Analyzer (Template-Compatible)")

SOURCE_REQUIRED_COLUMNS = ["时间", "堆电压(0.1V)", "堆电流(0.1A)", "堆SOC(0.1%)"]
COLUMN_ALIASES = {
    "时间": "Time",
    "堆电压(0.1V)": "Stack voltage",
    "堆电流(0.1A)": "Stack current",
    "堆SOC(0.1%)": "SOC",
    "序号": "Sequence",
    "堆最高电压": "MAX CELL",
    "堆最高电压位置": "MAX CELL POS",
    "堆最低电压": "MIN CELL",
    "堆最低电压位置": "MIN CELL POS",
}

# =========================
# Helpers
# =========================
def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric parser incl scientific notation:
    - '3,450' -> 3450
    - '3450mV' -> 3450
    - '3.45E3' -> 3450
    - '-1.2e2' -> -120
    """
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(",", "", regex=False)
    extracted = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


@st.cache_data(show_spinner=False)
def read_any_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_csv(uploaded_file, engine="python", encoding_errors="ignore")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";", engine="python", encoding_errors="ignore")


def normalize_bms_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {source: target for source, target in COLUMN_ALIASES.items() if source in df.columns}
    return df.rename(columns=aliases)


def compute_energy(
    df: pd.DataFrame,
    time_col: str,
    v_col: str,
    i_col: str,
    max_gap_seconds: float | None = None,
):
    """
    Energy via rectangular integration using dt to next sample.

    power_kW = V * A / 1000
    dE_kWh  = power_kW * dt_h

    Convention:
      power_kW > 0 => discharge energy OUT
      power_kW < 0 => charge energy IN
    """
    dfe = df.sort_values(time_col).copy()
    dfe["dt_s"] = (dfe[time_col].shift(-1) - dfe[time_col]).dt.total_seconds()
    dfe = dfe.dropna(subset=["dt_s"])
    dfe = dfe[dfe["dt_s"] > 0]

    if max_gap_seconds is not None:
        dfe["dt_s"] = dfe["dt_s"].clip(upper=max_gap_seconds)

    dfe["dt_h"] = dfe["dt_s"] / 3600.0
    dfe["power_kW"] = (dfe[v_col] * dfe[i_col]) / 1000.0
    dfe["dE_kWh"] = dfe["power_kW"] * dfe["dt_h"]
    dfe["discharge_kWh"] = dfe["dE_kWh"].clip(lower=0)
    dfe["charge_kWh"] = (-dfe["dE_kWh"]).clip(lower=0)
    dfe["net_energy_kWh"] = dfe["dE_kWh"].cumsum()
    dfe["cum_discharge_kWh"] = dfe["discharge_kWh"].cumsum()
    dfe["cum_charge_kWh"] = dfe["charge_kWh"].cumsum()

    e_out = float(dfe["discharge_kWh"].sum())
    e_in = float(dfe["charge_kWh"].sum())
    e_net = e_out - e_in
    return dfe, e_out, e_in, e_net


# =========================
# Sidebar
# =========================
st.sidebar.header("Upload")
bms_file = st.sidebar.file_uploader("Upload BMS file (.csv/.xlsx)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
bms_eng_units = st.sidebar.checkbox(
    "Already in engineering units (V/A/%/V) - no scaling",
    value=False,
    help="Unchecked means your file is likely 0.1V / 0.1A / 0.1% and cell in mV.",
)

max_gap = st.sidebar.number_input(
    "Energy gap clamp (seconds, 0 = no clamp)",
    min_value=0,
    value=0,
    step=10,
    help="Prevents a comms outage gap from dominating kWh integration.",
)
max_gap_seconds = None if max_gap == 0 else float(max_gap)

page = st.sidebar.radio("Page", ["Overview", "Energy"], index=0)

# =========================
# Load + parse BMS
# =========================
bms_df = None
bms_error = None

if bms_file is not None:
    try:
        df = read_any_table(bms_file)
    except Exception as e:
        bms_error = f"❌ Error reading file: {e}"
    else:
        if df.empty:
            bms_error = "❌ File has no rows."
        else:
            df.columns = df.columns.astype(str).str.strip()

            missing = [c for c in SOURCE_REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                bms_error = (
                    "❌ Missing required columns:\n"
                    + "\n".join(f"- {c}" for c in missing)
                    + "\n\nColumns found:\n"
                    + ", ".join(df.columns)
                )
            else:
                df = normalize_bms_columns(df)

                # parse time (handles '2026/2/27 0:03' etc.)
                df["__time__"] = pd.to_datetime(df["Time"], errors="coerce")
                df = df.dropna(subset=["__time__"]).sort_values("__time__")

                # parse numeric for required
                for col in ["Stack voltage", "Stack current", "SOC"]:
                    df[col] = clean_numeric_series(df[col])

                # parse numeric for optional if they exist
                optional_cols = ["Sequence", "MAX CELL", "MAX CELL POS", "MIN CELL", "MIN CELL POS"]
                for col in optional_cols:
                    if col in df.columns:
                        df[col] = clean_numeric_series(df[col])

                # scaling
                if not bms_eng_units:
                    df["Stack voltage"] = df["Stack voltage"] / 10.0
                    df["Stack current"] = df["Stack current"] / 10.0
                    df["SOC"] = df["SOC"] / 10.0
                    if "MAX CELL" in df.columns:
                        df["MAX CELL"] = df["MAX CELL"] / 1000.0
                    if "MIN CELL" in df.columns:
                        df["MIN CELL"] = df["MIN CELL"] / 1000.0

                # Only drop rows missing the 3 mandatory numeric fields
                df = df.dropna(subset=["Stack voltage", "Stack current", "SOC"])
                if df.empty:
                    bms_error = "❌ After parsing, no valid rows remain."
                else:
                    # Optional delta
                    if "MAX CELL" in df.columns and "MIN CELL" in df.columns:
                        df["cell_delta"] = df["MAX CELL"] - df["MIN CELL"]
                    else:
                        df["cell_delta"] = np.nan
                    bms_df = df

# =========================
# UI
# =========================
if bms_df is None:
    if bms_file is None:
        st.info("Upload your BMS file (with 时间, 堆电压(0.1V), 堆电流(0.1A), 堆SOC(0.1%)).")
    else:
        st.error(bms_error if bms_error else "❌ Failed to load.")
    st.stop()

df = bms_df

if page == "Overview":
    st.subheader("BMS Overview")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Stack V (max)", f"{df['Stack voltage'].max():.2f} V")
        st.metric("Stack V (min)", f"{df['Stack voltage'].min():.2f} V")
    with m2:
        # Cell metrics only if present
        if "MAX CELL" in df.columns and df["MAX CELL"].notna().any():
            st.metric("Max cell (abs)", f"{df['MAX CELL'].max():.3f} V")
        else:
            st.metric("Max cell (abs)", "N/A")
        if "MIN CELL" in df.columns and df["MIN CELL"].notna().any():
            st.metric("Min cell (abs)", f"{df['MIN CELL'].min():.3f} V")
        else:
            st.metric("Min cell (abs)", "N/A")
    with m3:
        st.metric("I (max)", f"{df['Stack current'].max():.1f} A")
        st.metric("I (min)", f"{df['Stack current'].min():.1f} A")
    with m4:
        st.metric("SoC range", f"{df['SOC'].min():.1f}% → {df['SOC'].max():.1f}%")

    st.markdown("---")

    st.plotly_chart(
        px.line(df, x="__time__", y="Stack voltage", title="Stack Voltage").update_layout(
            xaxis_title="Time", yaxis_title="V"
        ),
        use_container_width=True,
    )

    # Optional cell plots
    cell_series = []
    if "MIN CELL" in df.columns and df["MIN CELL"].notna().any():
        cell_series.append("MIN CELL")
    if "MAX CELL" in df.columns and df["MAX CELL"].notna().any():
        cell_series.append("MAX CELL")

    if cell_series:
        st.plotly_chart(
            px.line(df, x="__time__", y=cell_series, title="Cell Voltages (Optional)").update_layout(
                xaxis_title="Time", yaxis_title="V"
            ),
            use_container_width=True,
        )

    if "cell_delta" in df.columns and df["cell_delta"].notna().any():
        st.plotly_chart(
            px.line(df, x="__time__", y="cell_delta", title="Cell Delta (MAX - MIN, Optional)").update_layout(
                xaxis_title="Time", yaxis_title="V"
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        px.line(df, x="__time__", y="Stack current", title="Stack Current").update_layout(
            xaxis_title="Time", yaxis_title="A"
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.line(df, x="__time__", y="SOC", title="SoC").update_layout(
            xaxis_title="Time", yaxis_title="%"
        ),
        use_container_width=True,
    )

    with st.expander("Diagnostics"):
        st.write("Rows:", len(df))
        st.write("Time range:", df["__time__"].min(), "→", df["__time__"].max())
        show_cols = ["Time", "__time__", "Stack voltage", "Stack current", "SOC"]
        for c in ["Sequence", "MAX CELL", "MAX CELL POS", "MIN CELL", "MIN CELL POS", "cell_delta"]:
            if c in df.columns:
                show_cols.append(c)
        st.dataframe(df[show_cols].head(50))

elif page == "Energy":
    st.subheader("Energy (from Stack V/I)")

    df2 = df.sort_values("__time__").copy()
    if df2["__time__"].nunique() < 2:
        st.warning("Not enough points to compute energy.")
        st.stop()

    # Fast index slider (handles duplicate timestamps)
    start_i, end_i = st.slider("Select window (index-based)", 0, len(df2) - 1, (0, len(df2) - 1))
    start_t = df2.iloc[start_i]["__time__"]
    end_t = df2.iloc[end_i]["__time__"]
    st.caption(f"Selected: {start_t} → {end_t}")

    if start_t >= end_t:
        st.warning("Start must be before end.")
        st.stop()

    win = df2[(df2["__time__"] >= start_t) & (df2["__time__"] <= end_t)].copy()
    if win["__time__"].nunique() < 2:
        st.warning("Not enough points inside this window.")
        st.stop()

    calc, e_out, e_in, e_net = compute_energy(
        win, "__time__", "Stack voltage", "Stack current", max_gap_seconds=max_gap_seconds
    )

    if calc.empty:
        st.warning("Not enough valid intervals to compute energy.")
        st.stop()

    duration_h = float(calc["dt_h"].sum())
    avg_power_kw = float(calc["power_kW"].mean())
    peak_discharge_kw = float(calc["power_kW"].clip(lower=0).max())
    peak_charge_kw = float(calc["power_kW"].clip(upper=0).abs().max())

    a, b, c, d, e, f = st.columns(6)
    a.metric("Energy OUT (discharge)", f"{e_out:.2f} kWh")
    b.metric("Energy IN (charge)", f"{e_in:.2f} kWh")
    c.metric("Net (OUT - IN)", f"{e_net:.2f} kWh")
    d.metric("Window duration", f"{duration_h:.2f} h")
    e.metric("Average power", f"{avg_power_kw:.2f} kW")
    f.metric("Peak discharge / charge", f"{peak_discharge_kw:.2f} / {peak_charge_kw:.2f} kW")

    st.plotly_chart(
        px.line(calc, x="__time__", y="power_kW", title="Power (kW)").update_layout(
            xaxis_title="Time", yaxis_title="kW"
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.line(
            calc,
            x="__time__",
            y=["cum_discharge_kWh", "cum_charge_kWh", "net_energy_kWh"],
            title="Cumulative Energy",
        ).update_layout(xaxis_title="Time", yaxis_title="kWh"),
        use_container_width=True,
    )

    with st.expander("Energy calculation table (first 200 rows)"):
        st.dataframe(
            calc[
                [
                    "__time__",
                    "Stack voltage",
                    "Stack current",
                    "power_kW",
                    "dt_h",
                    "dE_kWh",
                    "cum_discharge_kWh",
                    "cum_charge_kWh",
                    "net_energy_kWh",
                ]
            ].head(200)
        )
