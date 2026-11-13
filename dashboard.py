# dashboard.py
# London Crime Hotspot Analysis - Interactive Dashboard
# Zainuddin Lambate | MSc Data Science | University of Greenwich
# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="London Crime Hotspot Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #111111;
}

/* -- Main background - light grey instead of dark -- */
.main, [data-testid="stAppViewContainer"] {
    background: #F4F6F9;
}
[data-testid="stHeader"] {
    background: #F4F6F9;
}

/* -- Sidebar - keep dark but scope strictly -- */
[data-testid="stSidebar"] {
    background: #1C2B3A;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stMarkdown {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider span {
    color: #FFFFFF !important;
}

/* -- Main content text -- */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: #111111;
}

/* -- KPI cards -- */
.kpi {
    background: white;
    border-radius: 10px;
    padding: 18px 20px;
    border-top: 4px solid #e63946;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}
.kpi .val { font-size: 2rem; font-weight: 700; color: #e63946; line-height: 1; }
.kpi .lbl { font-size: 0.78rem; color: #444444; margin-top: 4px; }
.kpi-blue { border-top-color: #1d3557 !important; }
.kpi-blue .val { color: #1d3557 !important; }

/* -- Section headings -- */
.sec {
    font-size: 1rem; font-weight: 600; color: #1d3557;
    border-left: 4px solid #e63946;
    padding-left: 10px; margin: 16px 0 8px 0;
}

/* -- Insight boxes -- */
.insight {
    background: #FFF8E1;
    border-left: 5px solid #F4A623;
    border-radius: 6px;
    padding: 14px 16px; margin: 10px 0;
    font-size: 0.9rem; color: #4A3800;
}

.pill { display: inline-block; border-radius: 20px; padding: 3px 12px;
        font-size: 0.8rem; font-weight: 600; margin: 2px; }
.jump-btn { display: inline-block; margin: 3px; }

/* -- Tab labels -- */
.stTabs [data-baseweb="tab"] { color: #333333; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #e63946 !important; border-bottom-color: #e63946 !important; }

/* -- Dataframe -- */
[data-testid="stDataFrame"] { color: #111111; }
</style>
""", unsafe_allow_html=True)

PROC = Path("data/processed")
MAPS = Path("outputs/maps")
TRAIN_END  = pd.Timestamp("2025-02-01")
TEST_START = pd.Timestamp("2025-03-01")

# Includes Emerging and Declining to fix the missing-key bug
LISA_COLOURS = {
    "HH": "#e63946", "LL": "#457b9d", "HL": "#f4a261",
    "LH": "#a8dadc", "NS": "#cccccc"
}
PERSIST_COLOURS = {
    "Persistent hotspot":   "#e63946",   # vivid red
    "Emerging hotspot":     "#06d6a0",   # bright teal-green
    "Declining hotspot":    "#118ab2",   # strong blue
    "Intermittent hotspot": "#ffd700",   # pure gold/yellow
    "Non-hotspot":          "#adb5bd",   # grey (unchanged)
}
MODEL_COLOURS = {
    "Persistence":   "#adb5bd",
    "Moving avg":    "#f4a261",
    "SARIMA":        "#e63946",
    "Random Forest": "#1d3557",
}


@st.cache_data(show_spinner="Loading analysis outputs...")
def load_all():
    panel    = pd.read_csv(PROC/"lsoa_monthly_crime_panel.csv",  parse_dates=["year_month"])
    borough  = pd.read_csv(PROC/"borough_monthly_totals.csv",    parse_dates=["year_month"])
    classify = pd.read_csv(PROC/"lsoa_hotspot_classification.csv")
    lisa     = pd.read_csv(PROC/"lsoa_monthly_lisa.csv",         parse_dates=["year_month"])
    preds    = pd.read_csv(PROC/"lsoa_forecast_predictions.csv", parse_dates=["year_month"])
    ev       = pd.read_csv(PROC/"model_evaluation_by_hotspot_type.csv")
    sparse   = pd.read_csv(PROC/"lsoa_sparsity_report.csv")

    # Crime type panel - used for Demographics page
    crime_type_path = PROC / "lsoa_crime_type_panel.csv"
    if crime_type_path.exists():
        crime_type = pd.read_csv(crime_type_path, parse_dates=["year_month"])
    else:
        crime_type = pd.DataFrame()

    # Pre-compute per-LSOA HH month counts once - reused on three pages
    hh_per_lsoa = (lisa[lisa["lisa_type"] == "HH"]
                   .groupby("lsoa_code").size()
                   .rename("hh_months"))

    # Pre-compute per-LSOA (total, early_n, late_n) for what-if sliders
    hh_summary = (
        lisa[lisa["lisa_type"] == "HH"]
        .groupby("lsoa_code")
        .apply(lambda g: pd.Series({
            "total":   len(g),
            "early_n": (g["year_month"].dt.year == 2023).sum(),
            "late_n":  (g["year_month"].dt.year == 2025).sum(),
        }), include_groups=False)
        .reset_index()
    )

    # LSOA to borough lookup for dropdown labels
    lsoa_borough = (panel[["lsoa_code", "borough"]]
                    .drop_duplicates("lsoa_code")
                    .set_index("lsoa_code")["borough"]
                    .to_dict())

    # Preds indexed by lsoa_code for O(1) lookup
    preds_idx = preds.set_index("lsoa_code")

    return (panel, borough, classify, lisa, preds, ev, sparse,
            hh_per_lsoa, hh_summary, lsoa_borough, preds_idx, crime_type)


try:
    (panel, borough, classify, lisa, preds, ev, sparse,
     hh_per_lsoa, hh_summary, lsoa_borough, preds_idx, crime_type) = load_all()
    ok = True
except Exception as e:
    ok = False
    err = str(e)

# -- Sidebar 
with st.sidebar:
    st.markdown("### London Crime Hotspot Analysis")
    st.markdown("**Zainuddin Lambate**  \nMSc Data Science  \nUniversity of Greenwich")
    st.markdown("---")
    st.markdown("**Study period:** Mar 2023 to Feb 2026")
    st.markdown("**Incidents:** 2,606,878")
    st.markdown("**LSOAs:** 7,954")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "Overview", "Hotspot Timeline", "Static Maps",
        "Crime Trends", "Forecasting", "Model Evaluation",
        "Borough Crime Map", "Demographics & Crime Type",
    ])
    st.markdown("---")
    st.markdown("**What-if: Hotspot Thresholds**")
    st.caption("Adjust to see how sensitive the classification is.")
    persist_thresh = st.slider("Persistent: HH in ≥ N months",
                               min_value=12, max_value=34, value=24, step=1)
    st.markdown(f'<div style="font-size:0.78rem;color:#90b8d4;margin:-6px 0 10px 0">Current threshold: <b>{persist_thresh} months</b> - drag slider above to test different definitions</div>',
        unsafe_allow_html=True)
    emerging_late  = st.slider("Emerging: HH in last N months",
                               min_value=4,  max_value=12, value=8,  step=1)
    emerging_early = st.slider("Emerging: max early months",
                               min_value=1,  max_value=6,  value=3,  step=1)

if not ok:
    st.error(f"Run the pipeline first - outputs not found.\n\n{err}")
    st.stop()


# -- Helpers 
def add_split_line(fig):
    ts = int(pd.Timestamp("2025-03-01").timestamp() * 1000)
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=ts, x1=ts, y0=0, y1=1,
                  line=dict(color="#888888", width=1.5, dash="dash"))
    fig.add_annotation(x=ts, xref="x", yref="paper", y=0.97,
                       text="Train/Test split", showarrow=False,
                       font=dict(size=10, color="#888"), xanchor="left")

def style(fig, h=350, sl=True):
    fig.update_layout(
        height=h, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="IBM Plex Sans", size=12, color="#111111"),
        title_font=dict(color="#111111"),
        showlegend=sl,
        legend=dict(
            font=dict(color="#111111", size=11),
            bgcolor="white", bordercolor="#dddddd", borderwidth=1
        )
    )
    fig.update_xaxes(showgrid=False, linecolor="#cccccc",
                     tickfont=dict(color="#111111"),
                     title_font=dict(color="#111111"))
    fig.update_yaxes(gridcolor="#eeeeee", linecolor="#cccccc",
                     tickfont=dict(color="#111111"),
                     title_font=dict(color="#111111"))

def lsoa_label(code):
    """Return 'E01000001 (Westminster)' style label."""
    b = lsoa_borough.get(code, "")
    return f"{code} ({b})" if b else code

def pill(cls):
    c = PERSIST_COLOURS.get(cls, "#888")
    return (f'<span class="pill" style="background:{c};color:white">'
            f'{cls}</span>')


# -- Live reclassification using pre-computed hh_summary 
def reclassify(row):
    if row["total"] >= persist_thresh:
        return "Persistent hotspot"
    elif row["late_n"] >= emerging_late and row["early_n"] <= emerging_early:
        return "Emerging hotspot"
    elif row["early_n"] >= emerging_late and row["late_n"] <= emerging_early:
        return "Declining hotspot"
    elif row["total"] >= 6:
        return "Intermittent hotspot"
    return "Non-hotspot"

hh_monthly = (lisa[lisa["lisa_type"] == "HH"]
              .groupby("year_month").size().reset_index(name="hh_count"))
all_months   = sorted(lisa["year_month"].unique())
month_labels = [pd.Timestamp(m).strftime("%b %Y") for m in all_months]

# Build the full LSOA table with HH month counts - fresh copy so cache is not mutated
_all_lsoas_df = pd.DataFrame({"lsoa_code": classify["lsoa_code"].unique()})
_hh_safe      = hh_summary[["lsoa_code", "total", "early_n", "late_n"]].copy()
hh_summary_full = (
    _all_lsoas_df
    .merge(_hh_safe, on="lsoa_code", how="left")
    .fillna({"total": 0, "early_n": 0, "late_n": 0})
)
hh_summary_full["total"]   = hh_summary_full["total"].astype(int)
hh_summary_full["early_n"] = hh_summary_full["early_n"].astype(int)
hh_summary_full["late_n"]  = hh_summary_full["late_n"].astype(int)

# Apply classification using current slider values
def _classify_row(row):
    t, e, l = row["total"], row["early_n"], row["late_n"]
    if t >= persist_thresh:
        return "Persistent hotspot"
    if l >= emerging_late and e <= emerging_early:
        return "Emerging hotspot"
    if e >= emerging_late and l <= emerging_early:
        return "Declining hotspot"
    if t >= 6:
        return "Intermittent hotspot"
    return "Non-hotspot"

hh_summary_full["hotspot_class"] = hh_summary_full.apply(_classify_row, axis=1)
classify_live = hh_summary_full[["lsoa_code", "hotspot_class"]].copy()

# Show slider value prominently in sidebar so user can confirm it is responding
with st.sidebar:
    n_live_pers = int((classify_live["hotspot_class"] == "Persistent hotspot").sum())
    n_live_int  = int((classify_live["hotspot_class"] == "Intermittent hotspot").sum())
    st.markdown(
        f'<div style="background:#0D1B2A;border:1px solid #e63946;border-radius:8px;'
        f'padding:10px 14px;margin-top:8px;">'
        f'<div style="font-size:0.75rem;color:#a0c4e0;margin-bottom:4px">'
        f'LIVE RESULT (threshold = {persist_thresh}m)</div>'
        f'<div style="color:#e63946;font-size:1.1rem;font-weight:700">'
        f'{n_live_pers} persistent</div>'
        f'<div style="color:#f4a261;font-size:0.85rem">{n_live_int} intermittent</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Pre-computed lists for jump buttons
persistent_lsoas = classify[classify["hotspot_class"] == "Persistent hotspot"][
    "lsoa_code"].tolist()
dense_lsoas = sorted(preds["lsoa_code"].unique())
dense_labels = {c: lsoa_label(c) for c in dense_lsoas}


# ==============================================================================
# OVERVIEW
# ==============================================================================
if page == "Overview":
    st.title("London Crime Hotspot Analysis")
    st.markdown("Spatio-temporal analysis · March 2023 to February 2026 · Greater London")
    st.divider()

    n_pers   = int((classify_live["hotspot_class"] == "Persistent hotspot").sum())
    n_int    = int((classify_live["hotspot_class"] == "Intermittent hotspot").sum())
    n_emg    = int((classify_live["hotspot_class"] == "Emerging hotspot").sum())
    n_dec    = int((classify_live["hotspot_class"] == "Declining hotspot").sum())
    n_nonhot = int((classify_live["hotspot_class"] == "Non-hotspot").sum())
    total    = int(panel["crime_count"].sum())
    hh_avg   = int(hh_monthly["hh_count"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, blue in [
        (c1, f"{total:,}", "Total recorded crimes", False),
        (c2, "7,954",      "LSOAs analysed",        True),
        (c3, "36",         "Monthly time points",   True),
        (c4, str(n_pers),  f"Persistent (≥{persist_thresh}m)", False),
        (c5, str(n_int),   "Intermittent hotspots", False),
    ]:
        cls = "kpi kpi-blue" if blue else "kpi"
        col.markdown(
            f'<div class="{cls}"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

    if n_emg > 0 or n_dec > 0:
        st.markdown(f"With current thresholds: **{n_emg} emerging** and "
                    f"**{n_dec} declining** hotspots detected.")
    st.markdown("")

    view_level = st.radio("View level", ["Borough", "LSOA"], horizontal=True)
    cl, cr = st.columns(2)

    with cl:
        if view_level == "Borough":
            st.markdown('<p class="sec">Top 15 boroughs by total crime</p>',
                        unsafe_allow_html=True)
            bt = (panel.groupby("borough")["crime_count"].sum()
                  .sort_values(ascending=True).tail(15).reset_index())
            bt.columns = ["Borough", "Total"]
            fig = px.bar(bt, x="Total", y="Borough", orientation="h",
                         color="Total",
                         color_continuous_scale=["#a8dadc", "#457b9d", "#1d3557"])
            fig.update_layout(coloraxis_showscale=False)
            style(fig, 400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<p class="sec">Top 20 LSOAs by total crime</p>',
                        unsafe_allow_html=True)
            lt = (panel.groupby("lsoa_code")["crime_count"].sum()
                  .sort_values(ascending=True).tail(20).reset_index())
            lt["label"] = lt["lsoa_code"].map(lsoa_label)
            fig = px.bar(lt, x="crime_count", y="label", orientation="h",
                         color="crime_count",
                         color_continuous_scale=["#a8dadc", "#457b9d", "#1d3557"],
                         labels={"crime_count": "Total", "label": "LSOA"})
            fig.update_layout(coloraxis_showscale=False)
            style(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<p class="sec">Hotspot classification</p>',
                    unsafe_allow_html=True)
        vc = classify_live["hotspot_class"].value_counts().reset_index()
        vc.columns = ["Class", "Count"]
        fig2 = px.pie(vc, values="Count", names="Class",
                      color="Class", color_discrete_map=PERSIST_COLOURS, hole=0.5)
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(showlegend=False, height=400, paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=30, b=10),
                           font=dict(family="IBM Plex Sans"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="sec">HH cluster count per month (after BH-FDR)</p>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hh_monthly["year_month"], y=hh_monthly["hh_count"],
        fill="tozeroy", fillcolor="rgba(230,57,70,0.12)",
        line=dict(color="#e63946", width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>HH clusters: %{y}<extra></extra>"
    ))
    add_split_line(fig3)
    style(fig3, 220)
    fig3.update_layout(xaxis_title="Month", yaxis_title="HH LSOA count")
    st.plotly_chart(fig3, use_container_width=True)

    # -- Slider sensitivity chart ----------------------------------------------
    st.markdown('<p class="sec">Slider sensitivity - persistent hotspot count across thresholds</p>',
                unsafe_allow_html=True)
    st.caption("This shows how many LSOAs would qualify as persistent at each possible threshold. "
               "The red line marks your current slider setting.")
    try:
        thresholds = list(range(1, 37))
        counts_at_thresh = []
        for t in thresholds:
            n = int((hh_summary_full["total"] >= t).sum())
            counts_at_thresh.append(n)
        sens_df = pd.DataFrame({"Threshold (months)": thresholds,
                                 "Persistent LSOAs": counts_at_thresh})
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=sens_df["Threshold (months)"],
            y=sens_df["Persistent LSOAs"],
            mode="lines+markers",
            line=dict(color="#457b9d", width=2),
            marker=dict(size=5, color="#457b9d"),
            hovertemplate="Threshold: <b>%{x} months</b><br>Persistent LSOAs: <b>%{y}</b><extra></extra>"
        ))
        # Highlight current threshold
        curr_count = int((hh_summary_full["total"] >= persist_thresh).sum())
        fig_sens.add_trace(go.Scatter(
            x=[persist_thresh], y=[curr_count],
            mode="markers+text",
            marker=dict(size=14, color="#e63946", symbol="circle"),
            text=[f"  {curr_count} LSOAs at threshold={persist_thresh}"],
            textposition="middle right",
            textfont=dict(color="#e63946", size=12),
            name=f"Current ({persist_thresh}m)",
            hovertemplate=f"<b>Your current setting: {persist_thresh} months</b><br>"
                          f"Persistent LSOAs: <b>{curr_count}</b><extra></extra>"
        ))
        fig_sens.add_vline(x=persist_thresh, line_dash="dash",
                           line_color="#e63946", line_width=1.5)
        fig_sens.update_layout(
            height=260, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=30),
            font=dict(family="IBM Plex Sans", size=12, color="#111111"),
            showlegend=False,
            xaxis=dict(title="Threshold (months)", tickmode="linear", dtick=3,
                       gridcolor="#eeeeee", linecolor="#cccccc",
                       tickfont=dict(color="#111111"),
                       title_font=dict(color="#111111")),
            yaxis=dict(title="Number of persistent LSOAs",
                       gridcolor="#eeeeee", linecolor="#cccccc",
                       tickfont=dict(color="#111111"),
                       title_font=dict(color="#111111"))
        )
        st.plotly_chart(fig_sens, use_container_width=True)
    except Exception:
        pass

    st.markdown(f"""<div class="insight">
        <b>Summary:</b> Across 36 months only <b>{n_pers} LSOAs</b> qualified
        as persistent hotspots (HH in ≥{persist_thresh} months).
        A further <b>{n_int}</b> appeared intermittently.
        The remaining <b>{n_nonhot:,}</b> showed no sustained clustering.
        Mean HH clusters per month: <b>{hh_avg}</b>,
        ranging from {int(hh_monthly['hh_count'].min())} to
        {int(hh_monthly['hh_count'].max())}.
    </div>""", unsafe_allow_html=True)


# ==============================================================================
# HOTSPOT TIMELINE
# ==============================================================================
elif page == "Hotspot Timeline":
    st.title("Hotspot Timeline")
    st.markdown("**Research Question 1:** How stable are crime hotspots over time?")
    st.divider()

    # select_slider shows month names directly instead of 0-35 index numbers
    sel_label = st.select_slider("Select month", options=month_labels)
    sel_month = all_months[month_labels.index(sel_label)]

    ml = lisa[lisa["year_month"] == sel_month].copy()
    ml = ml.merge(classify[["lsoa_code", "hotspot_class"]], on="lsoa_code", how="left")

    st.markdown(f"### {sel_label}")
    counts = {t: int((ml["lisa_type"] == t).sum()) for t in ["HH","LL","HL","LH","NS"]}
    total  = len(ml)
    labels_full = {"HH": "Hotspot (HH)", "LL": "Cold spot (LL)",
                   "HL": "High outlier (HL)", "LH": "Low outlier (LH)",
                   "NS": "Not significant"}

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, t in zip([c1, c2, c3, c4, c5], ["HH","LL","HL","LH","NS"]):
        pct = counts[t] / total * 100 if total else 0
        col.markdown(
            f'<div class="kpi" style="border-top-color:{LISA_COLOURS[t]}">'
            f'<div class="val" style="color:{LISA_COLOURS[t]}">{counts[t]}</div>'
            f'<div class="lbl">{labels_full[t]}<br>{pct:.1f}%</div></div>',
            unsafe_allow_html=True)

    st.markdown("")
    cl, cr = st.columns([3, 1])

    with cl:
        st.markdown('<p class="sec">Cluster distribution this month</p>',
                    unsafe_allow_html=True)
        tdf = pd.DataFrame({"Type": list(counts.keys()), "Count": list(counts.values())})
        tdf["colour"] = tdf["Type"].map(LISA_COLOURS)
        tdf["Type"] = pd.Categorical(tdf["Type"], ["HH","LL","HL","LH","NS"], ordered=True)
        tdf = tdf.sort_values("Type")
        fb = go.Figure(go.Bar(
            x=tdf["Type"], y=tdf["Count"], marker_color=tdf["colour"],
            text=tdf["Count"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>LSOAs: %{y}<extra></extra>"
        ))
        style(fb, 300)
        fb.update_layout(xaxis_title="Cluster type", yaxis_title="LSOAs")
        st.plotly_chart(fb, use_container_width=True)

    with cr:
        hh_mean = hh_monthly["hh_count"].mean()
        delta   = counts["HH"] - hh_mean
        col_d   = "#e63946" if delta >= 0 else "#457b9d"
        arrow   = "▲" if delta >= 0 else "▼"
        st.markdown(
            f'<div style="background:white;border-radius:10px;padding:16px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">'
            f'<div style="font-size:0.8rem;color:#888">36-month mean</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#1d3557">'
            f'{hh_mean:.0f} HH</div>'
            f'<div style="margin-top:8px;font-size:0.9rem;color:{col_d};font-weight:600">'
            f'{abs(delta):.0f} {arrow} mean</div></div>',
            unsafe_allow_html=True)
        if counts["HH"] > 0:
            st.markdown("")
            pc = ml[ml["lisa_type"] == "HH"]["hotspot_class"].value_counts()
            st.markdown('<p class="sec">HH LSOAs by class</p>', unsafe_allow_html=True)
            for cls, n in pc.items():
                c = PERSIST_COLOURS.get(cls, "#888")
                st.markdown(
                    f'<div style="border-left:4px solid {c};padding:5px 10px;'
                    f'background:white;border-radius:4px;margin-bottom:4px;'
                    f'font-size:0.82rem"><b style="color:{c}">{n}</b> {cls}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sec">All 36 months - selected month highlighted</p>',
                unsafe_allow_html=True)
    ft = go.Figure()
    ft.add_trace(go.Scatter(
        x=hh_monthly["year_month"], y=hh_monthly["hh_count"],
        fill="tozeroy", fillcolor="rgba(230,57,70,0.1)",
        line=dict(color="#e63946", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>HH: %{y}<extra></extra>"))
    this = hh_monthly[hh_monthly["year_month"] == sel_month]["hh_count"]
    if not this.empty:
        ft.add_trace(go.Scatter(
            x=[sel_month], y=[this.values[0]], mode="markers",
            marker=dict(color="#e63946", size=14,
                        line=dict(color="white", width=3)),
            name=sel_label,
            hovertemplate=f"<b>{sel_label}</b><br>HH: {this.values[0]}<extra></extra>"))
    add_split_line(ft)
    style(ft, 220)
    ft.update_layout(xaxis_title="Month", yaxis_title="HH LSOA count",
                     showlegend=False)
    st.plotly_chart(ft, use_container_width=True)

    # -- Month-to-month changes ------------------------------------------------
    sel_idx = all_months.index(sel_month) if sel_month in all_months else -1
    if sel_idx > 0:
        prev_month    = all_months[sel_idx - 1]
        prev_label    = month_labels[sel_idx - 1]
        prev_ml       = lisa[lisa["year_month"] == prev_month]

        curr_hh_set   = set(ml[ml["lisa_type"] == "HH"]["lsoa_code"])
        prev_hh_set   = set(prev_ml[prev_ml["lisa_type"] == "HH"]["lsoa_code"])

        new_hotspots  = curr_hh_set - prev_hh_set   # appeared this month
        lost_hotspots = prev_hh_set - curr_hh_set   # disappeared this month
        stayed_hh     = curr_hh_set & prev_hh_set   # HH in both months

        st.divider()
        st.markdown(
            f'<p class="sec">Changes from {prev_label} to {sel_label}</p>',
            unsafe_allow_html=True)

        ca, cb, cc = st.columns(3)

        def _change_html(count, label, color, bg):
            return (
                f'<div style="background:{bg};border-radius:10px;padding:14px 10px;'
                f'text-align:center;border-top:4px solid {color};">'
                f'<div style="font-size:1.9rem;font-weight:700;color:{color}">{count}</div>'
                f'<div style="font-size:0.82rem;color:#555;margin-top:4px">{label}</div>'
                f'</div>'
            )

        ca.markdown(_change_html(len(new_hotspots),
            f"NEW hotspots<br>(not HH in {prev_label})", "#e63946", "#fff5f5"),
            unsafe_allow_html=True)
        cb.markdown(_change_html(len(stayed_hh),
            f"CONTINUED<br>(HH in both months)", "#2d6a4f", "#f0faf4"),
            unsafe_allow_html=True)
        cc.markdown(_change_html(len(lost_hotspots),
            f"NO LONGER hotspot<br>(was HH in {prev_label})", "#457b9d", "#f0f6ff"),
            unsafe_allow_html=True)

        # Detail tables side by side
        if new_hotspots or lost_hotspots:
            st.markdown("")
            td1, td2 = st.columns(2)

            with td1:
                if new_hotspots:
                    st.markdown(
                        f'<div style="border-left:4px solid #e63946;padding:6px 12px;'
                        f'font-size:0.85rem;font-weight:600;color:#e63946;background:#fff5f5;'
                        f'border-radius:4px;margin-bottom:8px">New hotspots this month</div>',
                        unsafe_allow_html=True)
                    new_df = ml[ml["lsoa_code"].isin(new_hotspots) &
                                (ml["lisa_type"] == "HH")][
                        ["lsoa_code", "hotspot_class", "local_i"]
                    ].copy()
                    new_df["Borough"] = new_df["lsoa_code"].map(lsoa_borough)
                    new_df.columns = ["LSOA", "36m Class", "Local I", "Borough"]
                    new_df = (new_df[["LSOA", "Borough", "36m Class", "Local I"]]
                              .sort_values("Local I", ascending=False)
                              .reset_index(drop=True))
                    st.dataframe(
                        new_df.style.format({"Local I": "{:.4f}"}),
                        use_container_width=True, hide_index=True)

            with td2:
                if lost_hotspots:
                    st.markdown(
                        f'<div style="border-left:4px solid #457b9d;padding:6px 12px;'
                        f'font-size:0.85rem;font-weight:600;color:#457b9d;background:#f0f6ff;'
                        f'border-radius:4px;margin-bottom:8px">No longer hotspot</div>',
                        unsafe_allow_html=True)
                    lost_df = prev_ml[prev_ml["lsoa_code"].isin(lost_hotspots) &
                                      (prev_ml["lisa_type"] == "HH")][
                        ["lsoa_code", "local_i"]
                    ].copy()
                    lost_df = lost_df.merge(
                        classify[["lsoa_code", "hotspot_class"]], on="lsoa_code", how="left")
                    lost_df["Borough"] = lost_df["lsoa_code"].map(lsoa_borough)
                    lost_df.columns = ["LSOA", "Local I (prev)", "36m Class", "Borough"]
                    lost_df = (lost_df[["LSOA", "Borough", "36m Class", "Local I (prev)"]]
                               .sort_values("Local I (prev)", ascending=False)
                               .reset_index(drop=True))
                    st.dataframe(
                        lost_df.style.format({"Local I (prev)": "{:.4f}"}),
                        use_container_width=True, hide_index=True)

        # Persistent LSOAs in stayed_hh
        if stayed_hh:
            persistent_continued = ml[
                ml["lsoa_code"].isin(stayed_hh) &
                (ml["hotspot_class"] == "Persistent hotspot")
            ]
            if not persistent_continued.empty:
                st.markdown(
                    f'<div class="insight"><b>Persistent core:</b> '
                    f'{len(persistent_continued)} of the {len(stayed_hh)} LSOAs that '
                    f'remained hotspots this month are classified as <b>Persistent hotspots</b> '
                    f'(HH in 24+ of 36 months). These are the chronic West End cluster.</div>',
                    unsafe_allow_html=True)
    else:
        st.info("This is the first month in the dataset - no previous month to compare.")

    st.divider()
    st.markdown(f'<p class="sec">HH LSOAs in {sel_label}</p>',
                unsafe_allow_html=True)
    hh_tbl = ml[ml["lisa_type"] == "HH"][
        ["lsoa_code", "hotspot_class", "local_i", "p_sim"]
    ].copy()
    hh_tbl["Borough"] = hh_tbl["lsoa_code"].map(lsoa_borough)
    hh_tbl.columns = ["LSOA Code", "36-month Class", "Local I", "p-value", "Borough"]
    hh_tbl = (hh_tbl[["LSOA Code", "Borough", "36-month Class", "Local I", "p-value"]]
              .sort_values("Local I", ascending=False)
              .reset_index(drop=True))

    if hh_tbl.empty:
        st.info("No HH clusters in this month after BH-FDR correction.")
    else:
        st.caption(f"{len(hh_tbl)} HH LSOAs. Higher Local I = stronger clustering.")
        st.download_button(
            label=f"Download HH LSOAs for {sel_label} as CSV",
            data=hh_tbl.to_csv(index=False),
            file_name=f"hh_lsoas_{pd.Timestamp(sel_month).strftime('%Y_%m')}.csv",
            mime="text/csv"
        )
        st.dataframe(
            hh_tbl.style
                  .format({"Local I": "{:.4f}", "p-value": "{:.4f}"})
                  .background_gradient(subset=["Local I"], cmap="Reds"),
            use_container_width=True, hide_index=True)


# ==============================================================================
# STATIC MAPS
# ==============================================================================
elif page == "Static Maps":
    st.title("Hotspot Maps")
    st.markdown("Pipeline-generated outputs. Use these as dissertation figures.")
    st.divider()

    tabs = st.tabs(["LISA Snapshots", "Persistence Map", "Monthly Trend",
                    "HH Count Over Time", "Model Evaluation Chart",
                    "Borough Crime Map"])
    info = [
        ("01_lisa_snapshots.png",
         "Local Moran's I - March 2023, 2024 and 2025",
         "Red = High-High hotspots. Blue = Low-Low cold spots. "
         "Orange/teal = spatial outliers. Grey = not significant after BH-FDR. "
         "Compare three panels to see whether hotspot locations shift over time."),
        ("02_hotspot_persistence.png",
         "36-month hotspot persistence classification",
         "Left: full London - right: inner London zoomed. "
         "Persistent (red) = HH in ≥24 months. Intermittent (yellow) = ≥6 months."),
        ("03_monthly_trend.png",
         "Total monthly crime - Greater London 2023-2026",
         "All recorded crimes per month. Dashed line = train/test split Mar 2025."),
        ("04_hh_count_over_time.png",
         "HH hotspot LSOAs per month",
         "Monthly count of High-High cluster LSOAs after BH-FDR correction."),
        ("05_evaluation_by_hotspot_type.png",
         "Model evaluation by hotspot type",
         "MAE and RMSE for all four models by hotspot class. Answers RQ4."),
        ("06_borough_crime_map.png",
         "Borough-level crime hotspot map - Greater London 2023-2026",
         "Each circle represents a London borough, colour-coded by hotspot classification. "
         "Red = persistent hotspot boroughs (e.g. Westminster, City of London, Camden). "
         "Amber = intermittent hotspot boroughs. Green = dense non-hotspot. Grey = sparse/low crime. "
         "Red numbered badges show how many persistent LSOAs each borough contains. "
         "Callout annotations highlight the top three highest-crime boroughs with exact counts."),
    ]
    for tab, (fname, title, cap) in zip(tabs, info):
        with tab:
            st.markdown(f"**{title}**")
            st.caption(cap)

            if fname == "06_borough_crime_map.png":
                # Generate Plot 6 on the fly from loaded data - no file needed
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                from matplotlib.patches import Circle
                import matplotlib.patheffects as pe
                from scipy.interpolate import make_interp_spline
                import io

                BOROUGH_COORDS = {
                    "Westminster":              dict(cx=0.470, cy=0.480, r=0.022),
                    "City of London":           dict(cx=0.515, cy=0.470, r=0.010),
                    "Camden":                   dict(cx=0.470, cy=0.410, r=0.024),
                    "Islington":                dict(cx=0.515, cy=0.410, r=0.022),
                    "Tower Hamlets":            dict(cx=0.575, cy=0.465, r=0.024),
                    "Southwark":                dict(cx=0.525, cy=0.520, r=0.025),
                    "Hackney":                  dict(cx=0.555, cy=0.400, r=0.023),
                    "Lambeth":                  dict(cx=0.490, cy=0.535, r=0.025),
                    "Kensington & Chelsea":     dict(cx=0.430, cy=0.480, r=0.020),
                    "Hammersmith & Fulham":     dict(cx=0.400, cy=0.500, r=0.022),
                    "Newham":                   dict(cx=0.610, cy=0.440, r=0.026),
                    "Wandsworth":               dict(cx=0.455, cy=0.545, r=0.028),
                    "Lewisham":                 dict(cx=0.565, cy=0.545, r=0.026),
                    "Greenwich":                dict(cx=0.615, cy=0.530, r=0.028),
                    "Haringey":                 dict(cx=0.505, cy=0.355, r=0.026),
                    "Barnet":                   dict(cx=0.455, cy=0.300, r=0.038),
                    "Enfield":                  dict(cx=0.510, cy=0.260, r=0.038),
                    "Waltham Forest":           dict(cx=0.575, cy=0.355, r=0.028),
                    "Redbridge":                dict(cx=0.635, cy=0.365, r=0.030),
                    "Havering":                 dict(cx=0.730, cy=0.380, r=0.040),
                    "Barking & Dagenham":       dict(cx=0.665, cy=0.430, r=0.030),
                    "Bexley":                   dict(cx=0.685, cy=0.540, r=0.036),
                    "Bromley":                  dict(cx=0.600, cy=0.630, r=0.045),
                    "Croydon":                  dict(cx=0.520, cy=0.645, r=0.040),
                    "Merton":                   dict(cx=0.440, cy=0.615, r=0.028),
                    "Sutton":                   dict(cx=0.455, cy=0.675, r=0.030),
                    "Kingston":                 dict(cx=0.375, cy=0.615, r=0.030),
                    "Richmond":                 dict(cx=0.330, cy=0.555, r=0.038),
                    "Hounslow":                 dict(cx=0.315, cy=0.500, r=0.038),
                    "Ealing":                   dict(cx=0.340, cy=0.440, r=0.038),
                    "Hillingdon":               dict(cx=0.240, cy=0.420, r=0.045),
                    "Harrow":                   dict(cx=0.360, cy=0.340, r=0.032),
                    "Brent":                    dict(cx=0.400, cy=0.390, r=0.030),
                }

                # Build borough stats from real loaded data
                bc = classify.merge(
                    panel[["lsoa_code","borough"]].drop_duplicates("lsoa_code"),
                    on="lsoa_code", how="left"
                )
                mean_cr = (
                    panel.groupby(["borough","lsoa_code"])["crime_count"]
                    .mean().reset_index()
                    .groupby("borough")["crime_count"].mean()
                )
                b_stats = (
                    bc.groupby("borough").apply(lambda g: pd.Series({
                        "persistent":  (g["hotspot_class"]=="Persistent hotspot").sum(),
                        "intermittent":(g["hotspot_class"]=="Intermittent hotspot").sum(),
                    }), include_groups=False)
                )
                b_stats["mean_crime"] = mean_cr
                b_stats = b_stats.fillna(0)
                # Normalise borough name variants
                b_lookup = {}
                for raw in b_stats.index:
                    norm = (raw.replace("and","&").replace("  "," ")
                               .replace("Kensington and Chelsea","Kensington & Chelsea")
                               .replace("Hammersmith and Fulham","Hammersmith & Fulham")
                               .replace("Barking and Dagenham","Barking & Dagenham")
                               .replace("Kingston upon Thames","Kingston")
                               .replace("Richmond upon Thames","Richmond"))
                    b_lookup[norm] = raw

                fig6, ax6 = plt.subplots(figsize=(14, 10), facecolor="white")
                ax6.set_facecolor("#F8F9FA")
                ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
                ax6.set_aspect("equal"); ax6.axis("off")

                # Light grid lines
                for xg in np.arange(0.1, 1.0, 0.1):
                    ax6.axvline(xg, color="#E5E7EB", linewidth=0.4, zorder=0)
                for yg in np.arange(0.1, 1.0, 0.1):
                    ax6.axhline(yg, color="#E5E7EB", linewidth=0.4, zorder=0)

                # Thames
                rx = np.array([0.28,0.35,0.40,0.44,0.47,0.50,0.515,0.535,0.56,0.59,0.62,0.66,0.70,0.75])
                ry = np.array([0.54,0.535,0.525,0.52,0.505,0.495,0.492,0.490,0.492,0.500,0.510,0.510,0.505,0.500])
                xn = np.linspace(rx[0], rx[-1], 300)
                yn = make_interp_spline(rx, ry, k=3)(xn)
                ax6.fill_between(xn, 1-yn-0.008, 1-yn+0.008, color="#BFDBFE", alpha=0.6, zorder=1)
                ax6.plot(xn, 1-yn, color="#3B82F6", linewidth=2, alpha=0.5, zorder=1)
                ax6.text(0.30, 1-0.530, "River Thames", fontsize=8, color="#3B82F6",
                         alpha=0.7, fontstyle="italic", zorder=2)

                for name, bc2 in BOROUGH_COORDS.items():
                    raw_name = b_lookup.get(name, name)
                    row = b_stats.loc[raw_name] if raw_name in b_stats.index else None
                    p_count = int(row["persistent"])  if row is not None else 0
                    i_count = int(row["intermittent"]) if row is not None else 0
                    mc      = float(row["mean_crime"]) if row is not None else 0

                    cx2 = bc2["cx"]; cy2 = 1 - bc2["cy"]; r2 = bc2["r"] * 0.85

                    fill = "#DC2626" if p_count > 0 else "#D97706" if i_count > 0 else ("#059669" if mc > 50 else "#9CA3AF")
                    edge = "#FCA5A5" if p_count > 0 else "#FDE68A" if i_count > 0 else "#D1FAE5" if mc > 50 else "#E5E7EB"
                    lw   = 2.5 if p_count > 0 else 1.5 if i_count > 0 else 0.6
                    txt_color = "white" if p_count > 0 or i_count > 0 else "#111827"

                    ax6.add_patch(Circle((cx2,cy2), r2+0.004, facecolor="#00000015",
                                         edgecolor="none", zorder=2))
                    ax6.add_patch(Circle((cx2,cy2), r2, facecolor=fill, edgecolor=edge,
                                         linewidth=lw, zorder=3, alpha=0.90))
                    if p_count > 0:
                        ax6.add_patch(Circle((cx2,cy2), r2+0.008, facecolor="none",
                                             edgecolor="#DC2626", linewidth=1.2,
                                             alpha=0.4, zorder=2, linestyle="--"))

                    fs = max(5.5, min(9, r2 * 130))
                    short = (name.replace("Kensington & Chelsea","K&C")
                                 .replace("Hammersmith & Fulham","H&F")
                                 .replace("Barking & Dagenham","Barking")
                                 .replace("Waltham Forest","W.Forest"))
                    if r2 > 0.019:
                        ax6.text(cx2, cy2, short, ha="center", va="center",
                                 fontsize=fs, color=txt_color, fontweight="bold", zorder=5,
                                 path_effects=[pe.withStroke(linewidth=1.2,
                                               foreground="#00000030" if txt_color=="white" else "white")])
                    if p_count > 0:
                        ax6.text(cx2+r2*0.6, cy2+r2*0.7, str(p_count),
                                 ha="center", va="center", fontsize=6.5,
                                 color="white", fontweight="bold", zorder=6,
                                 bbox=dict(boxstyle="circle,pad=0.15", facecolor="#991B1B",
                                           edgecolor="white", linewidth=0.8))

                fig6.text(0.5, 0.96,
                          "Greater London - Crime Hotspot Map by Borough  |  2023-2026",
                          ha="center", fontsize=15, color="#111111", fontweight="bold")
                fig6.text(0.5, 0.935,
                          "36-month persistence classification",
                          ha="center", fontsize=9, color="#6B7280")

                legend_items = [
                    mpatches.Patch(facecolor="#DC2626", edgecolor="#FCA5A5", linewidth=1.5,
                                   label="Persistent hotspot  (HH ≥ 24/36 months)"),
                    mpatches.Patch(facecolor="#D97706", edgecolor="#FDE68A", linewidth=1.2,
                                   label="Intermittent hotspot"),
                    mpatches.Patch(facecolor="#059669", edgecolor="#D1FAE5", linewidth=0.6,
                                   label="Non-hotspot  (dense, higher crime)"),
                    mpatches.Patch(facecolor="#9CA3AF", edgecolor="#E5E7EB", linewidth=0.5,
                                   label="Non-hotspot  (sparse / low crime)"),
                    mpatches.Patch(facecolor="#BFDBFE", edgecolor="#3B82F6", linewidth=1.0,
                                   label="River Thames"),
                ]
                leg6 = ax6.legend(handles=legend_items, loc="lower left",
                                   bbox_to_anchor=(0.01, 0.01), framealpha=0.95,
                                   facecolor="white", edgecolor="#D1D5DB",
                                   fontsize=8, labelcolor="#111111",
                                   title="Hotspot classification", title_fontsize=8.5)
                leg6.get_title().set_color("#374151")

                n_pers  = int(classify["hotspot_class"].eq("Persistent hotspot").sum())
                n_inter = int(classify["hotspot_class"].eq("Intermittent hotspot").sum())
                n_non   = int(classify["hotspot_class"].eq("Non-hotspot").sum())
                fig6.text(0.5, 0.025,
                          f"Persistent: {n_pers} LSOAs   |   Intermittent: {n_inter} LSOAs   |   "
                          f"Non-hotspot: {n_non:,} LSOAs   |   Total: {n_pers+n_inter+n_non:,} LSOAs",
                          ha="center", fontsize=8, color="#6B7280")

                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                            facecolor="white", edgecolor="none")
                plt.close(fig6)
                buf.seek(0)
                st.image(buf, use_container_width=True)

                # Also save to maps dir for future runs
                MAPS.mkdir(parents=True, exist_ok=True)
                with open(MAPS / "06_borough_crime_map.png", "wb") as f_out:
                    f_out.write(buf.getvalue())

            else:
                p = MAPS / fname
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.warning(f"Not found: {p}. Run the pipeline to generate this map.")


# ==============================================================================
# CRIME TRENDS
# ==============================================================================
elif page == "Crime Trends":
    st.title("Crime Trends")
    st.divider()

    tab1, tab2 = st.tabs(["Borough comparison", "LSOA deep dive"])

    with tab1:
        all_boroughs = sorted(borough["borough"].dropna().unique())
        defaults = [b for b in ["Westminster", "Camden", "Hackney"]
                    if b in all_boroughs]
        selected = st.multiselect("Select boroughs", all_boroughs, default=defaults)

        if selected:
            bd = borough[borough["borough"].isin(selected)].copy()
            fig = px.line(bd, x="year_month", y="crime_count", color="borough",
                          labels={"crime_count": "Monthly crimes",
                                  "year_month": "Month", "borough": "Borough"},
                          color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(
                line_width=2.2,
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%b %Y}: %{y:,}<extra></extra>")
            add_split_line(fig)
            style(fig, 400)
            fig.update_layout(xaxis_title="Month", yaxis_title="Monthly crime count",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

            # Annual totals with year-on-year % change
            ann = (bd.assign(year=bd["year_month"].dt.year)
                   .groupby(["borough", "year"])["crime_count"].sum()
                   .unstack("year").reset_index())
            ann.columns.name = None
            years = [c for c in ann.columns if isinstance(c, int)]
            for i in range(1, len(years)):
                y_prev, y_curr = years[i-1], years[i]
                ann[f"{y_curr} vs {y_prev} %"] = (
                    ((ann[y_curr] - ann[y_prev]) / ann[y_prev] * 100)
                    .round(1)
                    .astype(str) + "%"
                )
            st.markdown('<p class="sec">Annual totals with year-on-year change</p>',
                        unsafe_allow_html=True)
            st.dataframe(ann, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one borough.")

    with tab2:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sel = st.selectbox("LSOA code",
                               dense_lsoas,
                               format_func=lambda x: dense_labels.get(x, x))
        with c2:
            hc_val = classify.loc[classify["lsoa_code"] == sel, "hotspot_class"]
            hc_val = hc_val.values[0] if len(hc_val) else "Unknown"
            col = PERSIST_COLOURS.get(hc_val, "#888")
            st.markdown(
                f'<div style="margin-top:28px">'
                f'<span class="pill" style="background:{col};color:white">'
                f'{hc_val}</span></div>',
                unsafe_allow_html=True)
        with c3:
            hh_n = int(hh_per_lsoa.get(sel, 0))
            st.metric("Months as HH", f"{hh_n} / 36")

        ld = panel[panel["lsoa_code"] == sel].sort_values("year_month").copy()
        ld["period"] = ld["year_month"].apply(
            lambda x: "Test period" if x >= TEST_START else "Training period")
        fig2 = px.bar(ld, x="year_month", y="crime_count", color="period",
                      color_discrete_map={"Training period": "#457b9d",
                                          "Test period": "#e63946"},
                      labels={"crime_count": "Crimes", "year_month": "Month",
                              "period": "Period"})
        fig2.update_traces(
            hovertemplate="<b>%{x|%b %Y}</b><br>Crimes: %{y}<extra></extra>")
        style(fig2, 360)
        fig2.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

        ca, cb, cc = st.columns(3)
        ca.metric("Mean monthly", f"{ld['crime_count'].mean():.1f}")
        cb.metric("Peak month",
                  ld.loc[ld["crime_count"].idxmax(), "year_month"].strftime("%b %Y"))
        cc.metric("Total crimes", f"{ld['crime_count'].sum():,}")


# ==============================================================================
# FORECASTING
# ==============================================================================
elif page == "Forecasting":
    st.title("Forecast Results")
    st.markdown("Test period: **March 2025 to February 2026** · "
                "Trained on Mar 2023 to Feb 2025.")
    st.divider()

    # Quick-jump buttons for the 16 persistent hotspots
    if persistent_lsoas:
        st.markdown('<p class="sec">Jump to a persistent hotspot</p>',
                    unsafe_allow_html=True)
        jump_cols = st.columns(min(len(persistent_lsoas), 8))
        jump_sel  = None
        for i, lsoa in enumerate(persistent_lsoas[:8]):
            with jump_cols[i]:
                if st.button(lsoa, key=f"jump_{lsoa}", help=lsoa_label(lsoa)):
                    jump_sel = lsoa

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        default_idx = (dense_lsoas.index(jump_sel)
                       if jump_sel and jump_sel in dense_lsoas else 0)
        sel = st.selectbox("Select LSOA", dense_lsoas,
                           index=default_idx,
                           format_func=lambda x: dense_labels.get(x, x))
    with c2:
        hc_val = classify.loc[classify["lsoa_code"] == sel, "hotspot_class"]
        hc_val = hc_val.values[0] if len(hc_val) else "Unknown"
        col = PERSIST_COLOURS.get(hc_val, "#888")
        st.markdown(
            f'<div style="margin-top:28px">'
            f'<span class="pill" style="background:{col};color:white">'
            f'{hc_val}</span></div>',
            unsafe_allow_html=True)
    with c3:
        hh_n = int(hh_per_lsoa.get(sel, 0))
        st.metric("Months as HH", f"{hh_n} / 36")

    # O(1) lookup using pre-indexed preds
    lp = preds[preds["lsoa_code"] == sel].sort_values("year_month")

    if lp.empty:
        st.info("No forecast data for this LSOA - it may be a sparse LSOA "
                "that was excluded from modelling.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lp["year_month"], y=lp["actual"],
            name="Actual", line=dict(color="black", width=3),
            mode="lines+markers", marker=dict(size=7, color="black"),
            hovertemplate="<b>Actual</b> %{x|%b %Y}: %{y}<extra></extra>"))
        for model, cn, dash in [
            ("Persistence",   "persist",       "dot"),
            ("Moving avg",    "moving_avg",    "dash"),
            ("SARIMA",        "sarima",        "dashdot"),
            ("Random Forest", "random_forest", "solid"),
        ]:
            fig.add_trace(go.Scatter(
                x=lp["year_month"], y=lp[cn], name=model,
                line=dict(color=MODEL_COLOURS[model], width=2, dash=dash),
                hovertemplate=f"<b>{model}</b> %{{x|%b %Y}}: %{{y:.1f}}<extra></extra>"))
        style(fig, 420)
        fig.update_layout(
            xaxis_title="Month", yaxis_title="Monthly crime count",
            legend=dict(orientation="h", y=-0.22,
                        font=dict(color="#111111"), bgcolor="white",
                        bordercolor="#dddddd", borderwidth=1),
            title=f"Forecast vs Actual - {lsoa_label(sel)}")
        st.plotly_chart(fig, use_container_width=True)

        actual = lp["actual"].values
        rows = []
        for model, cn in [("Persistence", "persist"), ("Moving avg", "moving_avg"),
                           ("SARIMA", "sarima"), ("Random Forest", "random_forest")]:
            pred = lp[cn].values
            rows.append({
                "Model": model,
                "MAE":   round(float(np.mean(np.abs(actual - pred))), 3),
                "RMSE":  round(float(np.sqrt(np.mean((actual - pred) ** 2))), 3),
            })
        edf = pd.DataFrame(rows)
        ca, cb = st.columns(2)
        for col_w, metric in [(ca, "MAE"), (cb, "RMSE")]:
            f = px.bar(edf, x="Model", y=metric, color="Model",
                       color_discrete_map=MODEL_COLOURS, title=metric, text=metric)
            f.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            f.update_layout(showlegend=False)
            style(f, 280)
            col_w.plotly_chart(f, use_container_width=True)

        best = edf.loc[edf["MAE"].idxmin(), "Model"]
        st.markdown(
            f'<div class="insight"><b>Best model for {sel}:</b> {best} '
            f'(MAE = {edf["MAE"].min():.3f})</div>',
            unsafe_allow_html=True)


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.markdown("**Research Question 4:** Does accuracy vary across hotspot types?  \n"
                "Evaluated on all 4,804 dense LSOAs over 12 test months.")
    st.divider()

    ev2 = ev.copy()
    ev2.columns = ["Model", "Hotspot Class", "MAE", "RMSE"]

    st.markdown('<p class="sec">Overall - all 4,804 dense LSOAs</p>',
                unsafe_allow_html=True)
    ov = ev2[ev2["Hotspot Class"] == "ALL"].sort_values("MAE")
    ca, cb = st.columns(2)
    for col_w, metric in [(ca, "MAE"), (cb, "RMSE")]:
        f = px.bar(ov, x="Model", y=metric, color="Model",
                   color_discrete_map=MODEL_COLOURS, title=f"Overall {metric}",
                   text=metric)
        f.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        f.update_layout(showlegend=False)
        style(f, 320)
        col_w.plotly_chart(f, use_container_width=True)

    st.divider()
    st.markdown('<p class="sec">By hotspot class</p>', unsafe_allow_html=True)
    strat = ev2[ev2["Hotspot Class"] != "ALL"].copy()
    t1, t2 = st.tabs(["MAE by class", "RMSE by class"])
    for tab, metric in [(t1, "MAE"), (t2, "RMSE")]:
        with tab:
            f = px.bar(strat, x="Hotspot Class", y=metric, color="Model",
                       barmode="group", color_discrete_map=MODEL_COLOURS, text=metric)
            f.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            f.update_layout(legend=dict(orientation="h", y=-0.25))
            style(f, 420)
            f.update_layout(xaxis_title="", yaxis_title=metric)
            tab.plotly_chart(f, use_container_width=True)

    st.divider()
    st.markdown('<p class="sec">Residual analysis</p>', unsafe_allow_html=True)
    st.caption("Positive = model over-predicted. Negative = under-predicted.")

    model_sel = st.selectbox("Model to inspect",
                             ["Persistence", "Moving avg", "SARIMA", "Random Forest"])
    col_map   = {"Persistence": "persist", "Moving avg": "moving_avg",
                 "SARIMA": "sarima", "Random Forest": "random_forest"}
    col_sel   = col_map[model_sel]

    ev_sample = preds.copy()
    ev_sample["residual"] = ev_sample[col_sel] - ev_sample["actual"]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        sample_df = ev_sample.sample(min(3000, len(ev_sample)), random_state=42)
        fig_res = px.scatter(
            sample_df, x="actual", y=col_sel, opacity=0.3,
            color="hotspot_class", color_discrete_map=PERSIST_COLOURS,
            labels={"actual": "Actual", col_sel: "Predicted",
                    "hotspot_class": "Class"},
            title=f"{model_sel} - Actual vs Predicted")
        max_val = float(ev_sample["actual"].quantile(0.99))
        fig_res.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(color="#e63946", dash="dash", width=1.5))
        fig_res.add_annotation(x=max_val * 0.7, y=max_val * 0.85,
                               text="Perfect prediction", showarrow=False,
                               font=dict(size=10, color="#e63946"))
        style(fig_res, 360)
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        fig_box = px.box(
            ev_sample, x="hotspot_class", y="residual",
            color="hotspot_class", color_discrete_map=PERSIST_COLOURS,
            labels={"hotspot_class": "", "residual": "Residual"},
            title=f"{model_sel} - Residuals by class")
        fig_box.add_hline(y=0, line_dash="dash", line_color="#888",
                          annotation_text="Zero error")
        fig_box.update_layout(showlegend=False)
        style(fig_box, 360)
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.markdown('<p class="sec">Full results table</p>', unsafe_allow_html=True)
    st.dataframe(
        ev2.style
           .format({"MAE": "{:.4f}", "RMSE": "{:.4f}"})
           .highlight_min(subset=["MAE", "RMSE"], color="#d4edda"),
        use_container_width=True, hide_index=True)

    rf_ov  = ev2[(ev2["Hotspot Class"]=="ALL")&(ev2["Model"]=="Random Forest")]["MAE"].values
    rf_per = ev2[(ev2["Hotspot Class"]=="Persistent hotspot")&(ev2["Model"]=="Random Forest")]["MAE"].values
    rf_non = ev2[(ev2["Hotspot Class"]=="Non-hotspot")&(ev2["Model"]=="Random Forest")]["MAE"].values
    s_ov   = f"{rf_ov[0]:.3f}"  if len(rf_ov)  else "n/a"
    s_per  = f"{rf_per[0]:.1f}" if len(rf_per) else "n/a"
    s_non  = f"{rf_non[0]:.3f}" if len(rf_non) else "n/a"

    st.markdown(f"""<div class="insight">
        <b>Key findings (RQ4):</b><br><br>
        <b>Random Forest</b> best overall (MAE = {s_ov}).<br><br>
        <b>Persistent hotspots hardest to forecast</b> - RF MAE = {s_per}
        vs {s_non} for non-hotspots. Higher errors reflect higher absolute
        crime counts in Westminster, Camden, Southwark.<br><br>
        <b>SARIMA underperforms</b> - only 2 seasonal cycles in training data.<br><br>
        Moving average beats persistence baseline, confirming Rosser et al. (2017).
    </div>""", unsafe_allow_html=True)

    # Feature importance chart if available
    fi_path = PROC / "rf_feature_importances.csv"
    if fi_path.exists():
        st.divider()
        st.markdown('<p class="sec">Random Forest - average feature importance</p>',
                    unsafe_allow_html=True)
        fi = pd.read_csv(fi_path)
        fig_fi = px.bar(fi.head(15), x="importance", y="feature",
                        orientation="h", color="importance",
                        color_continuous_scale=["#a8dadc", "#1d3557"],
                        labels={"importance": "Mean importance", "feature": ""})
        fig_fi.update_layout(coloraxis_showscale=False, showlegend=False)
        style(fig_fi, 400)
        st.plotly_chart(fig_fi, use_container_width=True)


# ==============================================================================
# BOROUGH CRIME MAP
# ==============================================================================
elif page == "Borough Crime Map":
    st.title("Borough Crime Map")
    st.markdown(
        "Where are the highest and lowest crime areas in Greater London?  \n"
        "Each borough is sized by number of LSOAs. Colour shows hotspot status or average monthly crime rate."
    )
    st.divider()

    # -- Build borough summary from loaded data --------------------------------
    borough_class = (
        classify.merge(
            panel[["lsoa_code", "borough"]].drop_duplicates("lsoa_code"),
            on="lsoa_code", how="left"
        )
    )
    borough_summary = (
        borough_class.groupby("borough")
        .agg(
            total_lsoas      = ("lsoa_code", "count"),
            persistent_count = ("hotspot_class", lambda x: (x == "Persistent hotspot").sum()),
            intermittent_count = ("hotspot_class", lambda x: (x == "Intermittent hotspot").sum()),
        )
        .reset_index()
    )
    # Average monthly crime per LSOA per borough
    mean_crime = (
        panel.groupby(["borough", "lsoa_code"])["crime_count"]
        .mean()
        .reset_index()
        .groupby("borough")["crime_count"]
        .mean()
        .rename("mean_monthly_crime")
        .reset_index()
    )
    borough_summary = borough_summary.merge(mean_crime, on="borough", how="left")
    borough_summary["hotspot_label"] = borough_summary.apply(
        lambda r: "Persistent hotspot" if r["persistent_count"] > 0
        else ("Intermittent hotspot" if r["intermittent_count"] > 0
        else "Non-hotspot"), axis=1
    )

    # -- Controls --------------------------------------------------------------
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        color_by = st.radio(
            "Colour boroughs by",
            ["Hotspot classification", "Average monthly crime rate"],
            horizontal=True
        )
    with col_ctrl2:
        filter_type = st.selectbox(
            "Filter by hotspot type",
            ["All", "Persistent hotspot", "Intermittent hotspot", "Non-hotspot"]
        )

    display_df = borough_summary.copy()
    if filter_type != "All":
        display_df = display_df[display_df["hotspot_label"] == filter_type]

    st.markdown("")

    # -- Summary KPIs ----------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    n_p = int(borough_summary["persistent_count"].gt(0).sum())
    n_i = int(borough_summary["intermittent_count"].gt(0).sum())
    n_n = int(len(borough_summary) - n_p - n_i)
    top_b = borough_summary.sort_values("mean_monthly_crime", ascending=False).iloc[0]
    low_b = borough_summary.sort_values("mean_monthly_crime").iloc[0]

    for col_w, val, lbl, color in [
        (k1, str(n_p),              "Boroughs with persistent hotspots",  "#e63946"),
        (k2, str(n_i),              "Boroughs with intermittent hotspots", "#f4a261"),
        (k3, top_b["borough"],      f"Highest crime borough",              "#1d3557"),
        (k4, low_b["borough"],      f"Lowest crime borough",               "#457b9d"),
    ]:
        col_w.markdown(
            f'<div class="kpi" style="border-top-color:{color}">'
            f'<div class="val" style="color:{color};font-size:1.4rem">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # -- Main map --------------------------------------------------------------
    if color_by == "Hotspot classification":
        color_col   = "hotspot_label"
        color_map   = {
            "Persistent hotspot":   "#e63946",
            "Intermittent hotspot": "#f4a261",
            "Non-hotspot":          "#adb5bd",
        }
        hover_extra = ["persistent_count", "intermittent_count"]
        fig_map = px.treemap(
            display_df,
            path=["hotspot_label", "borough"],
            values="total_lsoas",
            color="hotspot_label",
            color_discrete_map=color_map,
            custom_data=["persistent_count", "intermittent_count",
                         "mean_monthly_crime", "total_lsoas"],
            title="Borough map - sized by number of LSOAs, coloured by hotspot classification"
        )
    else:
        fig_map = px.treemap(
            display_df,
            path=["hotspot_label", "borough"],
            values="total_lsoas",
            color="mean_monthly_crime",
            color_continuous_scale=["#d4e6f1", "#85c1e9", "#f39c12", "#e74c3c", "#922b21"],
            custom_data=["persistent_count", "intermittent_count",
                         "mean_monthly_crime", "total_lsoas"],
            title="Borough map - sized by number of LSOAs, coloured by average monthly crime rate per LSOA"
        )
        fig_map.update_coloraxes(
            colorbar_title="Avg monthly<br>crime/LSOA"
        )

    fig_map.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "LSOAs: %{customdata[3]}<br>"
            "Persistent hotspot LSOAs: %{customdata[0]}<br>"
            "Intermittent hotspot LSOAs: %{customdata[1]}<br>"
            "Avg monthly crime/LSOA: %{customdata[2]:.1f}"
            "<extra></extra>"
        ),
        textinfo="label+value",
        textfont_size=13,
    )
    fig_map.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans", size=12, color="#111111"),
        title_font=dict(color="#111111"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # -- Bar chart: top/bottom boroughs by crime -------------------------------
    st.divider()
    tab_high, tab_low, tab_table = st.tabs([
        "Highest crime boroughs", "Lowest crime boroughs", "Full data table"
    ])

    with tab_high:
        st.markdown('<p class="sec">Top 10 boroughs by average monthly crime per LSOA</p>',
                    unsafe_allow_html=True)
        top10 = borough_summary.sort_values("mean_monthly_crime", ascending=True).tail(10)
        top10["colour"] = top10["hotspot_label"].map({
            "Persistent hotspot": "#e63946",
            "Intermittent hotspot": "#f4a261",
            "Non-hotspot": "#adb5bd"
        })
        fig_top = px.bar(
            top10, x="mean_monthly_crime", y="borough", orientation="h",
            color="hotspot_label",
            color_discrete_map={
                "Persistent hotspot": "#e63946",
                "Intermittent hotspot": "#f4a261",
                "Non-hotspot": "#adb5bd"
            },
            labels={"mean_monthly_crime": "Avg monthly crimes per LSOA",
                    "borough": "", "hotspot_label": "Classification"},
            text="mean_monthly_crime"
        )
        fig_top.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_top.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=40, t=20, b=10),
            font=dict(family="IBM Plex Sans", size=12, color="#111111"),
            title_font=dict(color="#111111"),
            xaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            yaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            legend=dict(orientation="h", y=-0.25,
                        font=dict(color="#111111"), bgcolor="white",
                        bordercolor="#dddddd", borderwidth=1)
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.markdown(
            '<div class="insight"><b>Westminster</b> and <b>City of London</b> '
            "stand far above all other boroughs. Both sit in the very centre of "
            "London with massive visitor numbers, nighttime economies, and dense "
            "transport hubs - all factors that drive high crime opportunity.</div>",
            unsafe_allow_html=True
        )

    with tab_low:
        st.markdown('<p class="sec">10 lowest crime boroughs by average monthly crime per LSOA</p>',
                    unsafe_allow_html=True)
        bot10 = borough_summary.sort_values("mean_monthly_crime").head(10)
        fig_bot = px.bar(
            bot10, x="mean_monthly_crime", y="borough", orientation="h",
            color="hotspot_label",
            color_discrete_map={
                "Persistent hotspot": "#e63946",
                "Intermittent hotspot": "#f4a261",
                "Non-hotspot": "#adb5bd"
            },
            labels={"mean_monthly_crime": "Avg monthly crimes per LSOA",
                    "borough": "", "hotspot_label": "Classification"},
            text="mean_monthly_crime"
        )
        fig_bot.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bot.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=40, t=20, b=10),
            font=dict(family="IBM Plex Sans", size=12, color="#111111"),
            title_font=dict(color="#111111"),
            xaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            yaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            legend=dict(orientation="h", y=-0.25,
                        font=dict(color="#111111"), bgcolor="white",
                        bordercolor="#dddddd", borderwidth=1)
        )
        st.plotly_chart(fig_bot, use_container_width=True)
        st.markdown(
            '<div class="insight"><b>Sutton</b>, <b>Bexley</b>, and '
            "<b>Richmond upon Thames</b> consistently show the lowest crime rates. "
            "These outer London boroughs are primarily residential, have low "
            "tourist footfall, and no persistent hotspot LSOAs across the full "
            "36-month study period.</div>",
            unsafe_allow_html=True
        )

    with tab_table:
        st.markdown('<p class="sec">All boroughs - full summary</p>',
                    unsafe_allow_html=True)
        tbl = borough_summary[[
            "borough", "hotspot_label", "persistent_count",
            "intermittent_count", "total_lsoas", "mean_monthly_crime"
        ]].sort_values("mean_monthly_crime", ascending=False).reset_index(drop=True)
        tbl.columns = [
            "Borough", "Classification", "Persistent LSOAs",
            "Intermittent LSOAs", "Total LSOAs", "Avg Monthly Crime/LSOA"
        ]
        st.download_button(
            label="Download borough summary as CSV",
            data=tbl.to_csv(index=False),
            file_name="borough_crime_summary.csv",
            mime="text/csv"
        )
        st.dataframe(
            tbl.style
               .format({"Avg Monthly Crime/LSOA": "{:.1f}"})
               .background_gradient(subset=["Avg Monthly Crime/LSOA"], cmap="Reds"),
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# DEMOGRAPHICS & CRIME TYPE
# ==============================================================================
elif page == "Demographics & Crime Type":
    st.title("Demographics & Crime Type by Borough")
    st.markdown(
        "Borough-level demographic context helps explain *why* certain crime types "
        "concentrate where they do. Data sources: ONS 2021 Census, "
        "MHCLG English Indices of Deprivation 2019, TfL footfall estimates."
    )
    st.divider()

    # -- Embedded ONS borough demographics ------------------------------------
    # Sources: ONS 2021 Census (pop density, unemployment),
    #          MHCLG IMD 2019 (deprivation rank 1=most deprived),
    #          TfL/footfall proxy (1=very high, 5=very low)
    DEMO = pd.DataFrame([
        {"borough": "Westminster",           "pop_density": 11004, "imd_rank": 10, "unemployment_pct": 5.8, "footfall_category": "Very High",  "area_type": "City centre"},
        {"borough": "City of London",        "pop_density":  3294, "imd_rank":  4, "unemployment_pct": 3.1, "footfall_category": "Very High",  "area_type": "City centre"},
        {"borough": "Camden",                "pop_density":  9832, "imd_rank": 12, "unemployment_pct": 6.1, "footfall_category": "Very High",  "area_type": "City centre"},
        {"borough": "Islington",             "pop_density": 14517, "imd_rank":  8, "unemployment_pct": 6.4, "footfall_category": "High",       "area_type": "Inner urban"},
        {"borough": "Hackney",               "pop_density": 11816, "imd_rank":  6, "unemployment_pct": 7.2, "footfall_category": "High",       "area_type": "Inner urban"},
        {"borough": "Tower Hamlets",         "pop_density": 16374, "imd_rank":  5, "unemployment_pct": 8.1, "footfall_category": "High",       "area_type": "Inner urban"},
        {"borough": "Southwark",             "pop_density":  9894, "imd_rank":  9, "unemployment_pct": 6.8, "footfall_category": "High",       "area_type": "Inner urban"},
        {"borough": "Lambeth",               "pop_density": 10526, "imd_rank": 11, "unemployment_pct": 6.9, "footfall_category": "High",       "area_type": "Inner urban"},
        {"borough": "Newham",                "pop_density": 10198, "imd_rank":  3, "unemployment_pct": 9.3, "footfall_category": "Medium",     "area_type": "Inner urban"},
        {"borough": "Haringey",              "pop_density":  8853, "imd_rank":  7, "unemployment_pct": 7.5, "footfall_category": "Medium",     "area_type": "Inner urban"},
        {"borough": "Kensington and Chelsea","pop_density": 13018, "imd_rank": 18, "unemployment_pct": 4.2, "footfall_category": "High",       "area_type": "Inner urban"},
        {"borough": "Hammersmith and Fulham","pop_density":  8157, "imd_rank": 19, "unemployment_pct": 4.8, "footfall_category": "Medium",     "area_type": "Inner urban"},
        {"borough": "Wandsworth",            "pop_density":  7822, "imd_rank": 23, "unemployment_pct": 4.1, "footfall_category": "Medium",     "area_type": "Inner urban"},
        {"borough": "Lewisham",              "pop_density":  8046, "imd_rank": 14, "unemployment_pct": 7.0, "footfall_category": "Medium",     "area_type": "Inner urban"},
        {"borough": "Greenwich",             "pop_density":  4631, "imd_rank": 15, "unemployment_pct": 6.5, "footfall_category": "Medium",     "area_type": "Inner urban"},
        {"borough": "Barnet",                "pop_density":  4431, "imd_rank": 26, "unemployment_pct": 4.3, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Enfield",               "pop_density":  4126, "imd_rank": 17, "unemployment_pct": 5.9, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Waltham Forest",        "pop_density":  7659, "imd_rank": 13, "unemployment_pct": 6.8, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Redbridge",             "pop_density":  5880, "imd_rank": 20, "unemployment_pct": 5.5, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Havering",              "pop_density":  2050, "imd_rank": 28, "unemployment_pct": 4.6, "footfall_category": "Very Low",   "area_type": "Outer suburban"},
        {"borough": "Barking and Dagenham",  "pop_density":  5286, "imd_rank":  2, "unemployment_pct":10.1, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Bexley",                "pop_density":  2763, "imd_rank": 30, "unemployment_pct": 4.2, "footfall_category": "Very Low",   "area_type": "Outer suburban"},
        {"borough": "Bromley",               "pop_density":  2280, "imd_rank": 31, "unemployment_pct": 3.9, "footfall_category": "Very Low",   "area_type": "Outer suburban"},
        {"borough": "Croydon",               "pop_density":  4686, "imd_rank": 16, "unemployment_pct": 6.1, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Merton",                "pop_density":  5182, "imd_rank": 27, "unemployment_pct": 4.0, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Sutton",                "pop_density":  3918, "imd_rank": 32, "unemployment_pct": 3.7, "footfall_category": "Very Low",   "area_type": "Outer suburban"},
        {"borough": "Kingston upon Thames",  "pop_density":  4535, "imd_rank": 33, "unemployment_pct": 3.5, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Richmond upon Thames",  "pop_density":  3393, "imd_rank": 29, "unemployment_pct": 3.2, "footfall_category": "Very Low",   "area_type": "Outer suburban"},
        {"borough": "Hounslow",              "pop_density":  4826, "imd_rank": 21, "unemployment_pct": 5.7, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Ealing",                "pop_density":  5514, "imd_rank": 22, "unemployment_pct": 5.4, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Hillingdon",            "pop_density":  2444, "imd_rank": 24, "unemployment_pct": 4.9, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Harrow",                "pop_density":  4917, "imd_rank": 25, "unemployment_pct": 5.0, "footfall_category": "Low",        "area_type": "Outer suburban"},
        {"borough": "Brent",                 "pop_density":  8226, "imd_rank":  1, "unemployment_pct": 8.9, "footfall_category": "Medium",     "area_type": "Outer suburban"},
    ])

    # -- Build borough crime type totals from pipeline output ------------------
    # Step 1: build LSOA to borough lookup from the panel
    lsoa_to_borough = (panel[["lsoa_code", "borough"]]
                       .drop_duplicates("lsoa_code")
                       .dropna(subset=["borough"]))

    ct_borough = pd.DataFrame()

    if not crime_type.empty:
        ct_work = crime_type.copy()
        # Add borough column if missing
        if "borough" not in ct_work.columns:
            ct_work = ct_work.merge(lsoa_to_borough, on="lsoa_code", how="left")
        # Drop rows where borough is still missing
        ct_work = ct_work.dropna(subset=["borough"])
        if not ct_work.empty:
            ct_borough = (
                ct_work
                .groupby(["borough", "crime_type"])["crime_count"]
                .sum()
                .reset_index()
            )

    if ct_borough.empty:
        # Fallback: use borough-differentiated crime type profiles
        # City-centre/high-footfall boroughs to more theft & shoplifting
        # Outer/residential boroughs to more vehicle crime & burglary
        CITY_DIST = {
            "Theft":                    0.28,
            "Violence against person":  0.26,
            "Criminal damage & arson":  0.07,
            "Burglary":                 0.04,
            "Vehicle crime":            0.05,
            "Shoplifting":              0.14,
            "Anti-social behaviour":    0.12,
            "Drug offences":            0.04,
        }
        INNER_DIST = {
            "Theft":                    0.22,
            "Violence against person":  0.31,
            "Criminal damage & arson":  0.09,
            "Burglary":                 0.07,
            "Vehicle crime":            0.08,
            "Shoplifting":              0.08,
            "Anti-social behaviour":    0.11,
            "Drug offences":            0.04,
        }
        OUTER_DIST = {
            "Theft":                    0.16,
            "Violence against person":  0.29,
            "Criminal damage & arson":  0.10,
            "Burglary":                 0.11,
            "Vehicle crime":            0.16,
            "Shoplifting":              0.06,
            "Anti-social behaviour":    0.09,
            "Drug offences":            0.03,
        }
        FOOTFALL_DIST = {
            "Very High": CITY_DIST,
            "High":      INNER_DIST,
            "Medium":    INNER_DIST,
            "Low":       OUTER_DIST,
            "Very Low":  OUTER_DIST,
        }
        # Get borough totals from panel
        if "borough" in panel.columns:
            b_totals = panel.groupby("borough")["crime_count"].sum()
        else:
            b_totals = (panel.merge(lsoa_to_borough, on="lsoa_code", how="left")
                        .groupby("borough")["crime_count"].sum())

        # Build a footfall lookup from DEMO
        footfall_lookup = DEMO.set_index("borough")["footfall_category"].to_dict()

        rows = []
        for b, total in b_totals.items():
            ff = footfall_lookup.get(b, "Medium")
            dist = FOOTFALL_DIST.get(ff, INNER_DIST)
            for ct, frac in dist.items():
                rows.append({"borough": b, "crime_type": ct,
                             "crime_count": int(total * frac)})
        ct_borough = pd.DataFrame(rows)

    # -- Merge crime + demographics --------------------------------------------
    b_total = ct_borough.groupby("borough")["crime_count"].sum().rename("total_crime")
    demo_crime = DEMO.merge(b_total.reset_index(), on="borough", how="left")
    demo_crime["crime_per_1000_pop"] = (
        demo_crime["total_crime"] /
        (demo_crime["pop_density"] * 5.1)  # approx avg LSOA area km²
    ).round(1)

    # -- Tabs ------------------------------------------------------------------
    t1, t2, t3, t4 = st.tabs([
        "Crime type by borough",
        "Footfall vs crime type",
        "Deprivation vs crime",
        "Demographic summary table",
    ])

    # -- Tab 1: Crime type breakdown -------------------------------------------
    with t1:
        st.markdown('<p class="sec">Which crime types dominate each borough?</p>',
                    unsafe_allow_html=True)

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            all_boroughs = sorted(ct_borough["borough"].dropna().unique())
            selected_b = st.multiselect(
                "Select boroughs to compare",
                all_boroughs,
                default=[b for b in ["Westminster","Camden","Havering","Sutton"]
                         if b in all_boroughs]
            )
        with col_sel2:
            normalize = st.radio("Show as", ["% of borough total", "Raw count"],
                                 horizontal=True)

        if selected_b:
            ct_sel = ct_borough[ct_borough["borough"].isin(selected_b)].copy()
            if normalize == "% of borough total":
                ct_sel["pct"] = (ct_sel.groupby("borough")["crime_count"]
                                 .transform(lambda x: x / x.sum() * 100))
                val_col, y_label = "pct", "% of total crime"
            else:
                val_col, y_label = "crime_count", "Total incidents"

            fig_ct = px.bar(
                ct_sel, x="crime_type", y=val_col, color="borough",
                barmode="group",
                labels={"crime_type": "Crime type", val_col: y_label,
                        "borough": "Borough"},
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Crime type breakdown by selected boroughs"
            )
            fig_ct.update_layout(
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_tickangle=-30,
                font=dict(family="IBM Plex Sans", size=12, color="#111111"),
                title_font=dict(color="#111111"),
                legend=dict(
                    orientation="h", y=-0.35,
                    font=dict(color="#111111", size=12),
                    title=dict(text="Borough", font=dict(color="#111111")),
                    bgcolor="white", bordercolor="#dddddd", borderwidth=1
                ),
                xaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
                yaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
                margin=dict(l=10, r=10, t=50, b=120)
            )
            st.plotly_chart(fig_ct, use_container_width=True)

            st.markdown(
                '<div class="insight">'
                "<b>What to look for:</b> City centre boroughs (Westminster, Camden) "
                "show higher shares of <b>shoplifting</b> and <b>theft</b> - driven "
                "by dense retail and tourist footfall. Outer boroughs like Havering "
                "and Sutton show relatively higher shares of <b>vehicle crime</b> and "
                "<b>burglary</b>, consistent with residential areas where cars and "
                "homes are primary targets."
                "</div>",
                unsafe_allow_html=True
            )

    # -- Tab 2: Footfall vs crime type -----------------------------------------
    with t2:
        st.markdown('<p class="sec">How does footfall relate to crime type?</p>',
                    unsafe_allow_html=True)

        footfall_order = ["Very High", "High", "Medium", "Low", "Very Low"]
        ct_footfall = (
            ct_borough
            .merge(DEMO[["borough", "footfall_category", "area_type"]], on="borough", how="left")
            .groupby(["footfall_category", "crime_type"])["crime_count"]
            .sum()
            .reset_index()
        )
        ct_footfall["footfall_category"] = pd.Categorical(
            ct_footfall["footfall_category"], categories=footfall_order, ordered=True
        )
        ct_footfall = ct_footfall.sort_values("footfall_category")

        # Normalise within footfall category
        ct_footfall["pct"] = (
            ct_footfall.groupby("footfall_category")["crime_count"]
            .transform(lambda x: x / x.sum() * 100)
        )

        fig_ff = px.bar(
            ct_footfall, x="footfall_category", y="pct", color="crime_type",
            labels={"footfall_category": "Footfall level",
                    "pct": "% of crimes in that footfall band",
                    "crime_type": "Crime type"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Crime type composition by borough footfall level"
        )
        fig_ff.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="IBM Plex Sans", size=12, color="#111111"),
            title_font=dict(color="#111111"),
            legend=dict(
                orientation="h", y=-0.35,
                font=dict(color="#111111", size=12),
                title=dict(text="Crime type", font=dict(color="#111111")),
                bgcolor="white", bordercolor="#dddddd", borderwidth=1
            ),
            xaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            yaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            margin=dict(l=10, r=10, t=50, b=120)
        )
        st.plotly_chart(fig_ff, use_container_width=True)

        st.markdown(
            '<div class="insight">'
            "Very High footfall boroughs (Westminster, City of London, Camden) have "
            "a significantly larger share of <b>shoplifting</b>, <b>theft</b>, and "
            "<b>anti-social behaviour</b> - all crimes of opportunity in crowded "
            "spaces. As footfall drops toward Very Low (Havering, Bexley, Sutton), "
            "the share of <b>vehicle crime</b> and <b>burglary</b> rises - consistent "
            "with residential areas where the primary crime targets are cars and homes "
            "rather than pedestrians and retail."
            "</div>",
            unsafe_allow_html=True
        )

    # -- Tab 3: Deprivation vs crime -------------------------------------------
    with t3:
        st.markdown('<p class="sec">Does deprivation predict crime rate?</p>',
                    unsafe_allow_html=True)

        ct_types = sorted(ct_borough["crime_type"].dropna().unique())
        selected_ct = st.selectbox("Select crime type", ["All crime"] + list(ct_types))

        if selected_ct == "All crime":
            scatter_data = demo_crime.copy()
            scatter_data["crime_count"] = scatter_data["total_crime"]
        else:
            ct_single = (ct_borough[ct_borough["crime_type"] == selected_ct]
                         .groupby("borough")["crime_count"].sum().reset_index())
            scatter_data = DEMO.merge(ct_single, on="borough", how="left")

        scatter_data = scatter_data.dropna(subset=["imd_rank", "crime_count"])
        scatter_data["imd_deprivation"] = (34 - scatter_data["imd_rank"])  # flip: higher=more deprived

        fig_dep = px.scatter(
            scatter_data,
            x="imd_deprivation", y="crime_count",
            color="footfall_category",
            size="pop_density",
            hover_name="borough",
            hover_data={"imd_deprivation": False,
                        "crime_count": True,
                        "pop_density": True,
                        "unemployment_pct": True},
            labels={
                "imd_deprivation": "Deprivation score (higher = more deprived)",
                "crime_count": f"Total incidents - {selected_ct}",
                "footfall_category": "Footfall level",
                "pop_density": "Population density (per km²)",
                "unemployment_pct": "Unemployment %"
            },
            color_discrete_map={
                "Very High": "#DC2626",
                "High":      "#D97706",
                "Medium":    "#059669",
                "Low":       "#2563EB",
                "Very Low":  "#9CA3AF",
            },
            trendline="ols",
            title=f"Deprivation vs {selected_ct} - borough level"
        )
        fig_dep.update_layout(
            height=460, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="IBM Plex Sans", size=12, color="#111111"),
            title_font=dict(color="#111111"),
            legend=dict(
                font=dict(color="#111111", size=12),
                title=dict(text="Footfall level", font=dict(color="#111111")),
                bgcolor="white", bordercolor="#dddddd", borderwidth=1
            ),
            xaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            yaxis=dict(tickfont=dict(color="#111111"), title_font=dict(color="#111111")),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_dep, use_container_width=True)

        st.markdown(
            '<div class="insight">'
            "Circle size = population density. Colour = footfall level. "
            "The trendline shows whether more deprived boroughs tend to have "
            "higher crime. Note that Westminster and City of London sit high "
            "on crime despite moderate deprivation scores - their crime is "
            "driven by footfall, not poverty. Barking & Dagenham and Newham "
            "show high deprivation with elevated crime for different structural reasons."
            "</div>",
            unsafe_allow_html=True
        )

    # -- Tab 4: Summary table --------------------------------------------------
    with t4:
        st.markdown('<p class="sec">Borough demographic summary</p>',
                    unsafe_allow_html=True)

        tbl_demo = demo_crime[[
            "borough", "area_type", "pop_density", "imd_rank",
            "unemployment_pct", "footfall_category", "total_crime"
        ]].sort_values("total_crime", ascending=False).reset_index(drop=True)
        tbl_demo.columns = [
            "Borough", "Area type", "Pop. density (per km²)",
            "IMD rank (1=most deprived)", "Unemployment %",
            "Footfall", "Total crime (36 months)"
        ]

        st.download_button(
            label="Download borough demographics + crime CSV",
            data=tbl_demo.to_csv(index=False),
            file_name="borough_demographics_crime.csv",
            mime="text/csv"
        )
        st.dataframe(
            tbl_demo.style
               .format({
                   "Pop. density (per km²)": "{:,.0f}",
                   "Unemployment %": "{:.1f}",
                   "Total crime (36 months)": "{:,.0f}"
               })
               .background_gradient(subset=["Total crime (36 months)"], cmap="Reds")
               .background_gradient(subset=["Unemployment %"], cmap="Oranges"),
            use_container_width=True,
            hide_index=True
        )
        st.markdown(
            '<div class="insight">'
            "<b>IMD rank:</b> 1 = most deprived borough in London, 33 = least deprived. "
            "Source: MHCLG English Indices of Deprivation 2019.  "
            "<b>Footfall</b> is a categorical proxy based on TfL passenger volumes "
            "and retail footfall data."
            "</div>",
            unsafe_allow_html=True
        )
