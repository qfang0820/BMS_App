import streamlit as st
import streamlit.components.v2 as components
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

ENERGY_SELECTOR_COMPONENT = components.component(
    name="energy_live_selector_v1",
    html="""
    <div class="energy-selector">
      <div class="selection-summary" id="selection-summary">Drag on the chart to inspect data live. Double-click to reset.</div>
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">Energy OUT</div><div class="metric-value" id="metric-out">0.00</div></div>
        <div class="metric-card"><div class="metric-label">Energy IN</div><div class="metric-value" id="metric-in">0.00</div></div>
        <div class="metric-card"><div class="metric-label">Net</div><div class="metric-value" id="metric-net">0.00</div></div>
        <div class="metric-card"><div class="metric-label">Window duration</div><div class="metric-value" id="metric-duration">0.00 h</div></div>
        <div class="metric-card"><div class="metric-label">Average power</div><div class="metric-value" id="metric-avg-power">0.00</div></div>
        <div class="metric-card"><div class="metric-label">Peak discharge / charge</div><div class="metric-value" id="metric-peak">0.00 / 0.00</div></div>
      </div>
      <div class="chart-wrap"><div id="energy-chart"></div></div>
      <div class="table-wrap">
        <div class="table-title" id="selection-title">Selected Data</div>
        <div class="table-scroll">
          <table>
            <thead id="selection-head"></thead>
            <tbody id="selection-body"></tbody>
          </table>
        </div>
      </div>
    </div>
    """,
    css="""
    .energy-selector {
      width: 100%;
      height: 100%;
      display: grid;
      gap: 0.75rem;
      color: var(--st-text-color);
      font-family: var(--st-font);
    }
    .selection-summary {
      font-size: 0.95rem;
      color: var(--st-secondary-text-color);
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 0.75rem;
    }
    .metric-card {
      border: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
      border-radius: 0.75rem;
      padding: 0.85rem 0.9rem;
      background: color-mix(in srgb, var(--st-bg-color) 92%, black 8%);
      min-height: 92px;
    }
    .metric-label {
      font-size: 0.82rem;
      color: var(--st-secondary-text-color);
      margin-bottom: 0.4rem;
    }
    .metric-value {
      font-size: 1.6rem;
      line-height: 1.15;
      font-weight: 600;
      letter-spacing: -0.02em;
    }
    .chart-wrap {
      height: 380px;
      border: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
      border-radius: 0.75rem;
      overflow: hidden;
      background: color-mix(in srgb, var(--st-bg-color) 88%, black 12%);
    }
    #energy-chart {
      width: 100%;
      height: 100%;
    }
    .table-wrap {
      border: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
      border-radius: 0.75rem;
      overflow: hidden;
      background: color-mix(in srgb, var(--st-bg-color) 92%, black 8%);
    }
    .table-title {
      padding: 0.75rem 0.9rem;
      font-weight: 600;
      border-bottom: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
    }
    .table-scroll {
      max-height: 250px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.55rem 0.7rem;
      border-bottom: 1px solid color-mix(in srgb, var(--st-border-color) 50%, transparent);
      text-align: left;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: color-mix(in srgb, var(--st-bg-color) 94%, black 6%);
      z-index: 1;
    }
    .muted {
      color: var(--st-secondary-text-color);
    }
    @media (max-width: 1200px) {
      .metrics-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    }
    @media (max-width: 720px) {
      .metrics-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .metric-value {
        font-size: 1.25rem;
      }
    }
    """,
    js="""
    const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    let plotlyPromise = null;

    function loadPlotly() {
      if (window.Plotly) {
        return Promise.resolve(window.Plotly);
      }
      if (!plotlyPromise) {
        plotlyPromise = new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = PLOTLY_URL;
          script.onload = () => resolve(window.Plotly);
          script.onerror = reject;
          document.head.appendChild(script);
        });
      }
      return plotlyPromise;
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") {
        return "";
      }
      if (typeof value === "number" && Number.isFinite(value)) {
        return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(3);
      }
      return String(value);
    }

    function uniqueSortedIndices(points) {
      const indices = [];
      for (const point of points || []) {
        if (Number.isInteger(point.pointNumber)) {
          indices.push(point.pointNumber);
        }
      }
      return [...new Set(indices)].sort((a, b) => a - b);
    }

    export default function(component) {
      const { parentElement, data, setStateValue } = component;
      const chartDiv = parentElement.querySelector("#energy-chart");
      const summaryEl = parentElement.querySelector("#selection-summary");
      const titleEl = parentElement.querySelector("#selection-title");
      const headEl = parentElement.querySelector("#selection-head");
      const bodyEl = parentElement.querySelector("#selection-body");
      const metricOut = parentElement.querySelector("#metric-out");
      const metricIn = parentElement.querySelector("#metric-in");
      const metricNet = parentElement.querySelector("#metric-net");
      const metricDuration = parentElement.querySelector("#metric-duration");
      const metricAvgPower = parentElement.querySelector("#metric-avg-power");
      const metricPeak = parentElement.querySelector("#metric-peak");
      const rows = data?.rows || [];
      const columns = data?.table_columns || [];
      const selectedIndicesFromPython = data?.selected_indices || [];
      const maxGapSeconds = data?.max_gap_seconds ?? null;
      const powerScale = data?.power_scale ?? 1;
      const powerUnit = data?.power_unit || "kW";
      const energyScale = data?.energy_scale ?? 1;
      const energyUnit = data?.energy_unit || "kWh";
      let destroyed = false;

      function formatMetricValue(value, scale, unit) {
        return `${(value / scale).toFixed(2)} ${unit}`;
      }

      function setMetricText(element, text) {
        if (element) {
          element.textContent = text;
        }
      }

      function renderMetrics(indices) {
        if (!rows.length) {
          setMetricText(metricOut, formatMetricValue(0, energyScale, energyUnit));
          setMetricText(metricIn, formatMetricValue(0, energyScale, energyUnit));
          setMetricText(metricNet, formatMetricValue(0, energyScale, energyUnit));
          setMetricText(metricDuration, "0.00 h");
          setMetricText(metricAvgPower, formatMetricValue(0, powerScale, powerUnit));
          setMetricText(metricPeak, `${(0).toFixed(2)} / ${(0).toFixed(2)} ${powerUnit}`);
          return;
        }

        const safeIndices = [...new Set(indices)].filter((index) => index >= 0 && index < rows.length).sort((a, b) => a - b);
        const startIndex = safeIndices.length >= 2 ? safeIndices[0] : 0;
        const endIndex = safeIndices.length >= 2 ? safeIndices[safeIndices.length - 1] : rows.length - 1;
        const windowRows = rows.slice(startIndex, endIndex + 1);

        if (windowRows.length < 2) {
          setMetricText(metricOut, formatMetricValue(0, energyScale, energyUnit));
          setMetricText(metricIn, formatMetricValue(0, energyScale, energyUnit));
          setMetricText(metricNet, formatMetricValue(0, energyScale, energyUnit));
          setMetricText(metricDuration, "0.00 h");
          setMetricText(metricAvgPower, formatMetricValue(0, powerScale, powerUnit));
          setMetricText(metricPeak, `${(0).toFixed(2)} / ${(0).toFixed(2)} ${powerUnit}`);
          return;
        }

        let eOut = 0;
        let eIn = 0;
        let durationH = 0;
        const powers = [];
        for (let i = 0; i < windowRows.length - 1; i += 1) {
          const current = windowRows[i];
          const next = windowRows[i + 1];
          let dtS = (next.time_ms - current.time_ms) / 1000;
          if (!Number.isFinite(dtS) || dtS <= 0) {
            continue;
          }
          if (maxGapSeconds !== null && maxGapSeconds > 0) {
            dtS = Math.min(dtS, maxGapSeconds);
          }
          const dtH = dtS / 3600;
          const power = Number(current.preview_power_kW) || 0;
          const dE = power * dtH;
          durationH += dtH;
          powers.push(power);
          if (dE >= 0) {
            eOut += dE;
          } else {
            eIn += -dE;
          }
        }

        const eNet = eOut - eIn;
        const avgPower = powers.length ? powers.reduce((sum, value) => sum + value, 0) / powers.length : 0;
        const peakDischarge = powers.length ? Math.max(...powers.map((value) => Math.max(value, 0))) : 0;
        const peakCharge = powers.length ? Math.max(...powers.map((value) => Math.max(-value, 0))) : 0;

        setMetricText(metricOut, formatMetricValue(eOut, energyScale, energyUnit));
        setMetricText(metricIn, formatMetricValue(eIn, energyScale, energyUnit));
        setMetricText(metricNet, formatMetricValue(eNet, energyScale, energyUnit));
        setMetricText(metricDuration, `${durationH.toFixed(2)} h`);
        setMetricText(metricAvgPower, formatMetricValue(avgPower, powerScale, powerUnit));
        setMetricText(
          metricPeak,
          `${(peakDischarge / powerScale).toFixed(2)} / ${(peakCharge / powerScale).toFixed(2)} ${powerUnit}`
        );
      }

      function renderTable(indices, liveMode) {
        const safeIndices = [...new Set(indices)].filter((index) => index >= 0 && index < rows.length).sort((a, b) => a - b);
        const startIndex = safeIndices.length >= 2 ? safeIndices[0] : 0;
        const endIndex = safeIndices.length >= 2 ? safeIndices[safeIndices.length - 1] : rows.length - 1;
        const displayRows = rows.slice(startIndex, endIndex + 1);
        const maxRows = 150;
        const visibleRows = displayRows.slice(0, maxRows);

        headEl.innerHTML = "";
        bodyEl.innerHTML = "";
        renderMetrics(safeIndices);

        const headRow = document.createElement("tr");
        for (const column of columns) {
          const th = document.createElement("th");
          th.textContent = column.label;
          headRow.appendChild(th);
        }
        headEl.appendChild(headRow);

        if (!visibleRows.length) {
          const row = document.createElement("tr");
          const td = document.createElement("td");
          td.colSpan = Math.max(columns.length, 1);
          td.className = "muted";
          td.textContent = "No points selected yet. Drag on the chart to preview rows live.";
          row.appendChild(td);
          bodyEl.appendChild(row);
          titleEl.textContent = "Selected Data";
          summaryEl.textContent = "Drag on the chart to inspect data live. Double-click to reset.";
          return;
        }

        for (const rowData of visibleRows) {
          const tr = document.createElement("tr");
          for (const column of columns) {
            const td = document.createElement("td");
            td.textContent = formatValue(rowData[column.key]);
            tr.appendChild(td);
          }
          bodyEl.appendChild(tr);
        }

        const firstRow = displayRows[0];
        const lastRow = displayRows[displayRows.length - 1];
        const suffix = displayRows.length > maxRows ? `, showing first ${maxRows}` : "";
        titleEl.textContent = `${safeIndices.length >= 2 ? "Selected Window Data" : "Full Window Data"} (${displayRows.length} points${suffix})`;
        summaryEl.textContent = `${liveMode ? "Selecting" : safeIndices.length >= 2 ? "Selected" : "Full"} window: ${firstRow.Time} → ${lastRow.Time}`;
      }

      function publishSelection(indices) {
        setStateValue("selected_indices", indices);
      }

      loadPlotly()
        .then((Plotly) => {
          if (destroyed) {
            return;
          }

          const trace = {
            x: rows.map((row) => row.Time),
            y: rows.map((row) => row.preview_power_display),
            mode: "lines+markers",
            type: "scatter",
            marker: {
              size: 5,
              color: "#7cc6ff",
            },
            line: {
              color: "#7cc6ff",
              width: 2,
            },
            selectedpoints: selectedIndicesFromPython.length ? selectedIndicesFromPython : null,
            selected: {
              marker: {
                color: "#ff9f43",
                size: 6,
              },
            },
            unselected: {
              marker: {
                opacity: 0.35,
              },
              line: {
                opacity: 0.45,
              },
            },
            hovertemplate: `Time: %{x}<br>Power: %{y:.2f} ${powerUnit}<extra></extra>`,
          };

          const layout = {
            title: { text: "Select Energy Window", font: { color: "#ffffff" } },
            dragmode: "select",
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            margin: { l: 50, r: 20, t: 50, b: 50 },
            xaxis: {
              title: { text: "Time" },
              color: "#d5d9e0",
              gridcolor: "rgba(120, 130, 150, 0.22)",
            },
            yaxis: {
              title: { text: powerUnit },
              color: "#d5d9e0",
              gridcolor: "rgba(120, 130, 150, 0.22)",
            },
          };

          const config = {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d"],
          };

          Plotly.react(chartDiv, [trace], layout, config);
          renderTable(selectedIndicesFromPython, false);

          if (typeof chartDiv.removeAllListeners === "function") {
            chartDiv.removeAllListeners("plotly_selecting");
            chartDiv.removeAllListeners("plotly_selected");
            chartDiv.removeAllListeners("plotly_deselect");
            chartDiv.removeAllListeners("plotly_doubleclick");
          }

          chartDiv.on("plotly_selecting", (eventData) => {
            renderTable(uniqueSortedIndices(eventData?.points), true);
          });

          chartDiv.on("plotly_selected", (eventData) => {
            const indices = uniqueSortedIndices(eventData?.points);
            renderTable(indices, false);
            publishSelection(indices);
          });

          chartDiv.on("plotly_deselect", () => {
            renderTable([], false);
            publishSelection([]);
          });

          chartDiv.on("plotly_doubleclick", () => {
            renderTable([], false);
            publishSelection([]);
          });
        })
        .catch((error) => {
          summaryEl.textContent = `Failed to load interactive selector: ${error}`;
        });

      return () => {
        destroyed = true;
      };
    }
    """,
)

OVERVIEW_SELECTOR_COMPONENT = components.component(
    name="overview_live_selector_v1",
    html="""
    <div class="overview-selector">
      <div class="selection-summary" id="selection-summary">Drag on the chart to inspect data live.</div>
      <div class="chart-wrap"><div id="overview-chart"></div></div>
      <div class="table-wrap">
        <div class="table-title" id="selection-title">Selected Data</div>
        <div class="table-scroll">
          <table>
            <thead id="selection-head"></thead>
            <tbody id="selection-body"></tbody>
          </table>
        </div>
      </div>
    </div>
    """,
    css="""
    .overview-selector {
      width: 100%;
      display: grid;
      gap: 0.75rem;
      color: var(--st-text-color);
      font-family: var(--st-font);
    }
    .selection-summary {
      font-size: 0.95rem;
      color: var(--st-secondary-text-color);
    }
    .chart-wrap {
      height: 380px;
      border: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
      border-radius: 0.75rem;
      overflow: hidden;
      background: color-mix(in srgb, var(--st-bg-color) 88%, black 12%);
    }
    #overview-chart {
      width: 100%;
      height: 100%;
    }
    .table-wrap {
      border: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
      border-radius: 0.75rem;
      overflow: hidden;
      background: color-mix(in srgb, var(--st-bg-color) 92%, black 8%);
    }
    .table-title {
      padding: 0.75rem 0.9rem;
      font-weight: 600;
      border-bottom: 1px solid color-mix(in srgb, var(--st-border-color) 70%, transparent);
    }
    .table-scroll {
      max-height: 250px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 0.55rem 0.7rem;
      border-bottom: 1px solid color-mix(in srgb, var(--st-border-color) 50%, transparent);
      text-align: left;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: color-mix(in srgb, var(--st-bg-color) 94%, black 6%);
      z-index: 1;
    }
    .muted {
      color: var(--st-secondary-text-color);
    }
    """,
    js="""
    const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    let plotlyPromise = null;

    function loadPlotly() {
      if (window.Plotly) {
        return Promise.resolve(window.Plotly);
      }
      if (!plotlyPromise) {
        plotlyPromise = new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = PLOTLY_URL;
          script.onload = () => resolve(window.Plotly);
          script.onerror = reject;
          document.head.appendChild(script);
        });
      }
      return plotlyPromise;
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") {
        return "";
      }
      if (typeof value === "number" && Number.isFinite(value)) {
        return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(3);
      }
      return String(value);
    }

    function uniqueSortedIndices(points) {
      const indices = [];
      for (const point of points || []) {
        if (Number.isInteger(point.pointNumber)) {
          indices.push(point.pointNumber);
        }
      }
      return [...new Set(indices)].sort((a, b) => a - b);
    }

    export default function(component) {
      const { parentElement, data } = component;
      const chartDiv = parentElement.querySelector("#overview-chart");
      const summaryEl = parentElement.querySelector("#selection-summary");
      const titleEl = parentElement.querySelector("#selection-title");
      const headEl = parentElement.querySelector("#selection-head");
      const bodyEl = parentElement.querySelector("#selection-body");
      const rows = data?.rows || [];
      const columns = data?.table_columns || [];
      const traces = data?.traces || [];
      let destroyed = false;

      function renderTable(indices, liveMode) {
        const safeIndices = [...new Set(indices)].filter((index) => index >= 0 && index < rows.length).sort((a, b) => a - b);
        const startIndex = safeIndices.length >= 2 ? safeIndices[0] : 0;
        const endIndex = safeIndices.length >= 2 ? safeIndices[safeIndices.length - 1] : rows.length - 1;
        const displayRows = rows.slice(startIndex, endIndex + 1);
        const maxRows = 120;
        const visibleRows = displayRows.slice(0, maxRows);

        headEl.innerHTML = "";
        bodyEl.innerHTML = "";

        const headRow = document.createElement("tr");
        for (const column of columns) {
          const th = document.createElement("th");
          th.textContent = column.label;
          headRow.appendChild(th);
        }
        headEl.appendChild(headRow);

        if (!visibleRows.length) {
          const row = document.createElement("tr");
          const td = document.createElement("td");
          td.colSpan = Math.max(columns.length, 1);
          td.className = "muted";
          td.textContent = "No points selected yet. Drag on the chart to preview rows live.";
          row.appendChild(td);
          bodyEl.appendChild(row);
          titleEl.textContent = "Selected Data";
          summaryEl.textContent = "Drag on the chart to inspect data live.";
          return;
        }

        for (const rowData of visibleRows) {
          const tr = document.createElement("tr");
          for (const column of columns) {
            const td = document.createElement("td");
            td.textContent = formatValue(rowData[column.key]);
            tr.appendChild(td);
          }
          bodyEl.appendChild(tr);
        }

        const firstRow = displayRows[0];
        const lastRow = displayRows[displayRows.length - 1];
        const suffix = displayRows.length > maxRows ? `, showing first ${maxRows}` : "";
        titleEl.textContent = `${safeIndices.length >= 2 ? "Selected Window Data" : "Full Window Data"} (${displayRows.length} points${suffix})`;
        summaryEl.textContent = `${liveMode ? "Selecting" : safeIndices.length >= 2 ? "Selected" : "Full"} window: ${firstRow.Time} → ${lastRow.Time}`;
      }

      loadPlotly()
        .then((Plotly) => {
          if (destroyed) {
            return;
          }

          const plotTraces = traces.map((trace) => ({
            x: rows.map((row) => row.Time),
            y: rows.map((row) => row[trace.key]),
            name: trace.label,
            mode: "lines+markers",
            type: "scatter",
            marker: { size: 4 },
            line: { width: 2, color: trace.color || undefined },
            hovertemplate: `${trace.label}: %{y}<br>Time: %{x}<extra></extra>`,
          }));

          const layout = {
            title: { text: data?.title || "Live Selection", font: { color: "#ffffff" } },
            dragmode: "select",
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            margin: { l: 50, r: 20, t: 50, b: 50 },
            xaxis: {
              title: { text: "Time" },
              color: "#d5d9e0",
              gridcolor: "rgba(120, 130, 150, 0.22)",
            },
            yaxis: {
              title: { text: data?.yaxis_title || "" },
              color: "#d5d9e0",
              gridcolor: "rgba(120, 130, 150, 0.22)",
            },
          };

          Plotly.react(chartDiv, plotTraces, layout, {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d"],
          });
          renderTable([], false);

          if (typeof chartDiv.removeAllListeners === "function") {
            chartDiv.removeAllListeners("plotly_selecting");
            chartDiv.removeAllListeners("plotly_selected");
            chartDiv.removeAllListeners("plotly_deselect");
            chartDiv.removeAllListeners("plotly_doubleclick");
          }

          chartDiv.on("plotly_selecting", (eventData) => {
            renderTable(uniqueSortedIndices(eventData?.points), true);
          });
          chartDiv.on("plotly_selected", (eventData) => {
            renderTable(uniqueSortedIndices(eventData?.points), false);
          });
          chartDiv.on("plotly_deselect", () => {
            renderTable([], false);
          });
          chartDiv.on("plotly_doubleclick", () => {
            renderTable([], false);
          });
        })
        .catch((error) => {
          summaryEl.textContent = `Failed to load interactive selector: ${error}`;
        });

      return () => {
        destroyed = true;
      };
    }
    """,
)

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


def get_component_state_value(key: str, field: str, default):
    state = st.session_state.get(key)
    if state is None:
        return default
    if hasattr(state, field):
        value = getattr(state, field)
        return default if value is None else value
    if isinstance(state, dict):
        value = state.get(field)
        return default if value is None else value
    return default


def choose_display_unit(max_abs_value: float, base_unit: str, large_unit: str, threshold: float = 1000.0):
    if max_abs_value >= threshold:
        return threshold, large_unit
    return 1.0, base_unit


def render_live_overview_chart(
    source_df: pd.DataFrame,
    *,
    title: str,
    y_columns: list[str],
    yaxis_title: str,
    key: str,
) -> None:
    table_columns = [{"key": "Time", "label": "Time"}]
    traces = []
    palette = ["#7cc6ff", "#ff9f43", "#5ad8a6", "#ff6b8a"]
    for index, column in enumerate(y_columns):
        table_columns.append({"key": column, "label": column})
        traces.append({"key": column, "label": column, "color": palette[index % len(palette)]})

    optional_columns = ["Sequence", "MAX CELL", "MAX CELL POS", "MIN CELL", "MIN CELL POS", "cell_delta"]
    existing_optional_columns = [column for column in optional_columns if column in source_df.columns]
    for column in existing_optional_columns:
        table_columns.append({"key": column, "label": column})

    selector_df = source_df[["__time__", "Time", *y_columns, *existing_optional_columns]].copy()
    selector_df["Time"] = selector_df["__time__"].dt.strftime("%Y-%m-%d %H:%M:%S")
    selector_df = selector_df.drop(columns="__time__")
    selector_df = selector_df.where(pd.notna(selector_df), None)

    OVERVIEW_SELECTOR_COMPONENT(
        data={
            "title": title,
            "yaxis_title": yaxis_title,
            "rows": selector_df.to_dict("records"),
            "table_columns": table_columns,
            "traces": traces,
        },
        key=key,
        width="stretch",
        height="content",
    )


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

    render_live_overview_chart(
        df,
        title="Stack Voltage",
        y_columns=["Stack voltage"],
        yaxis_title="V",
        key="overview_stack_voltage_live",
    )

    # Optional cell plots
    cell_series = []
    if "MIN CELL" in df.columns and df["MIN CELL"].notna().any():
        cell_series.append("MIN CELL")
    if "MAX CELL" in df.columns and df["MAX CELL"].notna().any():
        cell_series.append("MAX CELL")

    if cell_series:
        render_live_overview_chart(
            df,
            title="Cell Voltages (Optional)",
            y_columns=cell_series,
            yaxis_title="V",
            key="overview_cell_voltage_live",
        )

    if "cell_delta" in df.columns and df["cell_delta"].notna().any():
        render_live_overview_chart(
            df,
            title="Cell Delta (MAX - MIN, Optional)",
            y_columns=["cell_delta"],
            yaxis_title="V",
            key="overview_cell_delta_live",
        )

    render_live_overview_chart(
        df,
        title="Stack Current",
        y_columns=["Stack current"],
        yaxis_title="A",
        key="overview_stack_current_live",
    )

    render_live_overview_chart(
        df,
        title="SoC",
        y_columns=["SOC"],
        yaxis_title="%",
        key="overview_soc_live",
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

    df2["preview_power_kW"] = (df2["Stack voltage"] * df2["Stack current"]) / 1000.0
    power_scale, power_unit = choose_display_unit(float(df2["preview_power_kW"].abs().max()), "kW", "MW")
    df2["preview_power_display"] = df2["preview_power_kW"] / power_scale

    energy_slider_key = "energy_window_slider"
    energy_slider_pending_key = "energy_window_slider_pending"
    pending_window = st.session_state.pop(energy_slider_pending_key, None)
    current_window = pending_window if pending_window is not None else st.session_state.get(
        energy_slider_key, (0, len(df2) - 1)
    )
    start_default = max(0, min(int(current_window[0]), len(df2) - 1))
    end_default = max(start_default, min(int(current_window[1]), len(df2) - 1))
    st.session_state[energy_slider_key] = (start_default, end_default)

    # Fast index slider (handles duplicate timestamps)
    start_i, end_i = st.slider(
        "Select window (index-based)",
        0,
        len(df2) - 1,
        value=st.session_state[energy_slider_key],
        key=energy_slider_key,
    )
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

    energy_abs_max = max(
        e_out,
        e_in,
        abs(e_net),
        float(calc["cum_discharge_kWh"].abs().max()),
        float(calc["cum_charge_kWh"].abs().max()),
        float(calc["net_energy_kWh"].abs().max()),
    )
    energy_scale, energy_unit = choose_display_unit(energy_abs_max, "kWh", "MWh")
    duration_h = float(calc["dt_h"].sum())
    avg_power_kw = float(calc["power_kW"].mean())
    peak_discharge_kw = float(calc["power_kW"].clip(lower=0).max())
    peak_charge_kw = float(calc["power_kW"].clip(upper=0).abs().max())

    a, b, c, d, e, f = st.columns(6)
    a.metric("Energy OUT (discharge)", f"{e_out / energy_scale:.2f} {energy_unit}")
    b.metric("Energy IN (charge)", f"{e_in / energy_scale:.2f} {energy_unit}")
    c.metric("Net (OUT - IN)", f"{e_net / energy_scale:.2f} {energy_unit}")
    d.metric("Window duration", f"{duration_h:.2f} h")
    e.metric("Average power", f"{avg_power_kw / power_scale:.2f} {power_unit}")
    f.metric(
        "Peak discharge / charge",
        f"{peak_discharge_kw / power_scale:.2f} / {peak_charge_kw / power_scale:.2f} {power_unit}",
    )

    calc["power_display"] = calc["power_kW"] / power_scale
    calc["cum_discharge_display"] = calc["cum_discharge_kWh"] / energy_scale
    calc["cum_charge_display"] = calc["cum_charge_kWh"] / energy_scale
    calc["net_energy_display"] = calc["net_energy_kWh"] / energy_scale

    power_fig = px.line(df2, x="__time__", y="preview_power_display", title=f"Power ({power_unit})").update_layout(
        xaxis_title="Time",
        yaxis_title=power_unit,
        dragmode="select",
        selectdirection="h",
    )
    power_fig.update_traces(mode="lines+markers", marker={"size": 5})
    power_min = float(df2["preview_power_display"].min())
    power_max = float(df2["preview_power_display"].max())
    if power_min == power_max:
        power_min -= 1.0
        power_max += 1.0
    power_fig.add_selection(
        x0=start_t,
        x1=end_t,
        y0=power_min,
        y1=power_max,
        line=dict(color="#ff9f43", width=2, dash="solid"),
    )

    power_selection = st.plotly_chart(
        power_fig,
        use_container_width=True,
        key="energy_power_selector_chart",
        on_select="rerun",
        selection_mode="box",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
    )

    selected_points = list(power_selection.selection.points) if power_selection else []
    selection_signature = tuple(
        point.get("point_index") for point in selected_points if point.get("point_index") is not None
    )
    last_applied_signature = st.session_state.get("energy_power_last_applied_signature")

    if len(selection_signature) >= 2 and selection_signature != last_applied_signature:
        graph_indices = sorted(set(int(point_index) for point_index in selection_signature))
        graph_window = (graph_indices[0], graph_indices[-1])
        st.session_state["energy_power_last_applied_signature"] = selection_signature
        if graph_window != st.session_state[energy_slider_key]:
            st.session_state[energy_slider_pending_key] = graph_window
            st.rerun()

    if not selection_signature and last_applied_signature:
        st.session_state["energy_power_last_applied_signature"] = ()
        full_window = (0, len(df2) - 1)
        if st.session_state[energy_slider_key] != full_window:
            st.session_state[energy_slider_pending_key] = full_window
            st.rerun()

    st.caption("The power chart and the slider share the same start/end time window.")

    st.plotly_chart(
        px.line(
            calc,
            x="__time__",
            y=["cum_discharge_display", "cum_charge_display", "net_energy_display"],
            title=f"Cumulative Energy ({energy_unit})",
        ).update_layout(
            xaxis_title="Time",
            yaxis_title=energy_unit,
            legend_title_text="Series",
        ),
        use_container_width=True,
    )

    with st.expander("Energy calculation table (raw kW/kWh, first 200 rows)"):
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
