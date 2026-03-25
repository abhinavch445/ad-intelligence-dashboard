"""
app.py  —  Ad Intelligence Dashboard  (Fixed + Upgraded)
Run:  streamlit run dashboard/app.py
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.feature_engineering import (
    load_and_merge, encode_features, encode_single,
    CATEGORIES, CITIES, get_category_stats, get_city_stats
)
from utils.simulation import (
    simulate_funnel, simulate_over_cpc_range,
    simulate_over_ctr_range, sensitivity_analysis, build_roas_heatmap
)
from utils.optimization import (
    optimize_cpc, get_cpc_recommendation, roas_label, generate_insights,
    build_category_city_heatmap
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ad Intelligence", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background:#f0f4f8; }
.block-container { padding:1.5rem 2.5rem 2rem 2.5rem; max-width:1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background:#ffffff;
    border-right:1px solid #e2e8f0;
    box-shadow:2px 0 8px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] * { color:#1e293b !important; }
[data-testid="stSidebar"] .stSelectbox>div>div {
    background:#f8fafc !important;
    border:1px solid #cbd5e1 !important;
    border-radius:6px !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:12px !important;
    padding:18px 20px !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.06) !important;
    transition:box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    box-shadow:0 4px 16px rgba(37,99,235,0.10) !important;
}
div[data-testid="metric-container"] * { color:#1e293b !important; }
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size:10px !important; text-transform:uppercase !important;
    letter-spacing:1.2px !important; color:#94a3b8 !important; font-weight:600 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size:24px !important; font-weight:700 !important; color:#0f172a !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#ffffff !important; border-radius:10px !important;
    padding:4px !important; gap:2px !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    color:#64748b !important; border-radius:7px !important;
    font-size:13px !important; font-weight:500 !important; padding:7px 18px !important;
}
.stTabs [aria-selected="true"] {
    background:#2563eb !important; color:#ffffff !important;
    font-weight:600 !important; box-shadow:0 2px 8px rgba(37,99,235,0.3) !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    background:#f8fafc !important; border:1px solid #cbd5e1 !important;
    border-radius:6px !important; color:#1e293b !important;
    font-size:13px !important;
}

/* Section headings */
h1 { color:#0f172a !important; font-weight:700 !important; letter-spacing:-0.5px !important; }
h2,h3,h4 { color:#1e293b !important; font-weight:600 !important; }
p, caption, .stMarkdown p { color:#475569 !important; }
hr { border-color:#e2e8f0 !important; }
label { color:#334155 !important; font-weight:500 !important; }

/* Insight boxes — border-left style, forced dark text */
.ins-good {
    background:#f0fdf4; border-left:4px solid #16a34a;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0; }
.ins-good p { color:#14532d !important; margin:2px 0 !important; }

.ins-warn {
    background:#fffbeb; border-left:4px solid #d97706;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0; }
.ins-warn p { color:#78350f !important; margin:2px 0 !important; }

.ins-bad {
    background:#fef2f2; border-left:4px solid #ef4444;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0; }
.ins-bad p { color:#7f1d1d !important; margin:2px 0 !important; }

/* Section card wrapper */
.section-card {
    background:#ffffff; border-radius:12px; padding:20px 24px;
    border:1px solid #e2e8f0; margin-bottom:16px;
    box-shadow:0 1px 4px rgba(0,0,0,0.05);
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE    = "#2563eb"
PURPLE  = "#7c3aed"
GREEN   = "#16a34a"
RED     = "#ef4444"
AMBER   = "#d97706"
TEAL    = "#0891b2"
COLORS  = ["#2563eb","#7c3aed","#0891b2","#16a34a","#d97706","#dc2626","#059669","#9333ea"]
CSCALE  = [[0,"#dbeafe"],[0.5,"#3b82f6"],[1,"#1e3a8a"]]


def chart(fig, title="", h=300, show_legend=False):
    """Apply sharp, consistent styling to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=13, color="#1e293b", family="Inter"),
            x=0, y=0.98
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbfc",
        height=h,
        margin=dict(t=50 if title else 20, b=40, l=50, r=20),
        font=dict(color="#334155", size=11, family="Inter"),
        xaxis=dict(
            gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False,
            tickfont=dict(color="#64748b", size=10), showgrid=True,
            mirror=True, linewidth=1
        ),
        yaxis=dict(
            gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False,
            tickfont=dict(color="#64748b", size=10), showgrid=True,
            mirror=True, linewidth=1
        ),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
            font=dict(color="#334155", size=10)
        ),
    )
    return fig


# ── Load models ───────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")


@st.cache_resource
def load_models():
    files = ["order_model.pkl","revenue_model.pkl","bid_model.pkl",
             "encoders.pkl","feature_importance.pkl","data_stats.pkl"]
    if any(not os.path.exists(os.path.join(MODELS_DIR, f)) for f in files):
        return None
    def load(n):
        with open(os.path.join(MODELS_DIR, n), "rb") as f: return pickle.load(f)
    return (load("order_model.pkl"), load("revenue_model.pkl"), load("bid_model.pkl"),
            load("encoders.pkl"), load("feature_importance.pkl"), load("data_stats.pkl"))


@st.cache_data
def load_dataset():
    camp = os.path.join(DATA_DIR, "campaign_data.csv")
    sell = os.path.join(DATA_DIR, "final_dataset.xlsx")
    if not os.path.exists(camp): return None
    df = load_and_merge(camp, sell)
    df, _, _ = encode_features(df)
    return df


# ── Pre-aggregate heavy data to avoid JSON limit ──────────────────────────────
@st.cache_data
def precompute_aggregates(_df):
    """Pre-aggregate everything that would otherwise send 79k rows to Plotly."""
    # Histogram bins per category
    hist_data = {}
    for cat in CATEGORIES:
        sub = _df[_df["Category"] == cat]["ROAS"].dropna()
        counts, bins = np.histogram(sub, bins=35, range=(0, sub.quantile(0.99)))
        hist_data[cat] = {"counts": counts.tolist(), "bins": bins.tolist()}

    # Box plot: sample 500 per category
    box_sample = (_df.groupby("Category")
                  .apply(lambda x: x.sample(min(500, len(x)), random_state=42))
                  .reset_index(drop=True)[["Category","city","ROAS","CTR","CVR","CPC","AOV"]])

    # K-Means on full data (cached)
    feats = ["CTR","CVR","CPC","ROAS","AOV"]
    X = _df[feats].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df_c = _df.loc[X.index].copy()
    df_c["Cluster"] = labels
    roas_by_cluster = df_c.groupby("Cluster")["ROAS"].mean().sort_values(ascending=False)
    names = ["🏆 High Performers","📈 Growth Potential","⚠️ Needs Attention","🔴 Underperforming"]
    name_map = {idx: names[rank] for rank, idx in enumerate(roas_by_cluster.index)}
    df_c["Segment"] = df_c["Cluster"].map(name_map)

    # Cluster sample for scatter (800 points max)
    cluster_sample = (df_c.groupby("Segment")
                      .apply(lambda x: x.sample(min(200, len(x)), random_state=42))
                      .reset_index(drop=True)[["Segment","CTR","CVR","ROAS","CPC"]])

    # Cluster summary
    cluster_summary = (df_c.groupby("Segment")
                       .agg(Campaigns=("campaign_id","count"),
                            Avg_ROAS=("ROAS","mean"), Avg_CTR=("CTR","mean"),
                            Avg_CVR=("CVR","mean"),  Avg_CPC=("CPC","mean"),
                            Avg_Revenue=("revenue","mean"))
                       .reset_index().round(2))

    # Category × City pivot
    pivot = _df.pivot_table(values="ROAS", index="Category",
                            columns="city", aggfunc="mean").round(2)

    # Category stats
    cat_stats = get_category_stats(_df)
    city_stats = get_city_stats(_df)

    return dict(
        hist_data=hist_data, box_sample=box_sample,
        cluster_sample=cluster_sample, cluster_summary=cluster_summary,
        pivot=pivot, cat_stats=cat_stats, city_stats=city_stats
    )


models_bundle = load_models()
df = load_dataset()

if models_bundle is None or df is None:
    st.error("⚠️ Models or data not found.")
    st.info("1. Copy data into `data/`\n2. Run `python training/train_models.py`\n3. Reload")
    st.stop()

order_model, rev_model, bid_model, (cat_enc, city_enc), feat_imp, data_info = models_bundle
stats   = data_info["stats"]
metrics = data_info["metrics"]
ord_std = data_info["order_residual_std"]
rev_std = data_info["rev_residual_std"]

with st.spinner("Loading dashboard..."):
    agg = precompute_aggregates(df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Ad Intelligence")
    st.caption("ML-Powered Campaign Optimizer")
    st.divider()

    # Quick personas
    st.markdown("**⚡ Quick Personas**")
    persona = st.selectbox("Load Preset", [
        "Custom",
        "🏆 Premium Electronics · Mumbai",
        "👗 Fashion · Delhi",
        "💄 Beauty · Bengaluru",
        "🏠 Home & Kitchen · Pune",
        "📱 Mobiles · Hyderabad",
    ])
    PERSONAS = {
        "🏆 Premium Electronics · Mumbai": ("Electronics","Mumbai",  100_000,4.5,1.2,28,35000),
        "👗 Fashion · Delhi":              ("Fashion",    "Delhi",    60_000, 5.0,0.8,14, 2500),
        "💄 Beauty · Bengaluru":           ("Beauty",     "Bengaluru",50_000, 4.2,1.5,10, 1800),
        "🏠 Home & Kitchen · Pune":        ("Home & Kitchen","Pune",  40_000, 3.2,0.7,12, 4500),
        "📱 Mobiles · Hyderabad":          ("Mobiles",   "Hyderabad",80_000, 3.8,0.9,22,18000),
    }
    if persona in PERSONAS:
        p_cat,p_city,p_impr,p_ctr,p_cvr,p_cpc,p_aov = PERSONAS[persona]
    else:
        p_cat,p_city = CATEGORIES[0],CITIES[0]
        p_impr,p_ctr,p_cvr,p_cpc,p_aov = (stats["med_impr"],stats["med_ctr"],
                                            stats["med_cvr"],stats["med_cpc"],stats["med_aov"])

    st.divider()
    st.markdown("**🎯 Campaign Inputs**")
    sel_cat  = st.selectbox("Category", CATEGORIES,
                            index=CATEGORIES.index(p_cat) if p_cat in CATEGORIES else 0)
    sel_city = st.selectbox("City", CITIES,
                            index=CITIES.index(p_city) if p_city in CITIES else 0)

    st.divider()
    st.markdown("**🎛 What-If Parameters**")
    st.caption("Use slider or type a value directly")

    def param_input(label, min_v, max_v, default, step, fmt="int"):
        col_s, col_n = st.columns([3, 1])
        with col_s:
            slider_val = st.slider(label, min_v, max_v, default, step,
                                   key=f"slider_{label}", label_visibility="visible")
        with col_n:
            if fmt == "float":
                num_val = st.number_input("", min_value=float(min_v), max_value=float(max_v),
                                          value=float(slider_val), step=float(step),
                                          key=f"num_{label}", label_visibility="collapsed",
                                          format="%.1f")
            else:
                num_val = st.number_input("", min_value=int(min_v), max_value=int(max_v),
                                          value=int(slider_val), step=int(step),
                                          key=f"num_{label}", label_visibility="collapsed")
        return num_val

    impr = param_input("Impressions", 10_000, 200_000, int(p_impr), 5_000)
    ctr  = param_input("CTR (%)", 1.0, 8.0, float(p_ctr), 0.1, "float")
    cvr  = param_input("CVR (%)", 0.5, 6.0, float(p_cvr), 0.1, "float")
    cpc  = param_input("CPC (₹)", stats["min_cpc"], stats["max_cpc"], int(p_cpc), 1)
    aov  = param_input("AOV (Rs.)", 500, 50_000, int(p_aov), 100)

    st.divider()
    budget = st.number_input("💰 Daily Budget (₹)", 500, 500_000, 20_000, 500)
    st.divider()
    st.caption(f"Order R²: **{metrics['order']['r2']}**")
    st.caption(f"Revenue R²: **{metrics['revenue']['r2']}**")


# ── Core simulation ───────────────────────────────────────────────────────────
cat_val, city_val = encode_single(sel_cat, sel_city, cat_enc, city_enc)

result  = simulate_funnel(impr, ctr, cvr, cpc, aov, order_model, rev_model,
                          cat_val, city_val, stats["med_rating"], stats["med_disc"])
clicks  = result["clicks"]
orders  = result["orders"]
spend   = result["spend"]
revenue = result["revenue"]
roas    = result["roas"]

order_lo = max(0, orders  - 1.96 * ord_std)
order_hi = orders  + 1.96 * ord_std
rev_lo   = max(0, revenue - 1.96 * rev_std)
rev_hi   = revenue + 1.96 * rev_std
roas_lo  = rev_lo  / spend if spend > 0 else 0
roas_hi  = rev_hi  / spend if spend > 0 else 0

seg_df_base = df[(df["Category"]==sel_cat) & (df["city"]==sel_city)]
base_roas   = float(seg_df_base["ROAS"].median() if len(seg_df_base) > 10
                    else df["ROAS"].median())
delta_roas  = roas - base_roas
roas_lbl, roas_color = roas_label(roas)
rec = get_cpc_recommendation(df, sel_cat, sel_city)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Ad Campaign Intelligence")
st.caption(
    f"**{sel_cat}**  ·  **{sel_city}**  ·  "
    f"Budget ₹{budget:,}  ·  Segment baseline ROAS **{base_roas:.2f}x**"
)
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────────
st.markdown("### Predicted Outcomes")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Predicted Clicks",  f"{clicks:,.0f}")
k2.metric("Predicted Orders",  f"{orders:.1f}",
          help=f"95% CI: {order_lo:.1f} – {order_hi:.1f}")
k3.metric("Ad Spend",          f"₹{spend:,.0f}")
k4.metric("Predicted Revenue", f"₹{revenue:,.0f}",
          help=f"95% CI: ₹{rev_lo:,.0f} – ₹{rev_hi:,.0f}")
k5.metric("ROAS",              f"{roas:.2f}x",
          delta=f"{delta_roas:+.2f} vs segment")

c1, c2, c3 = st.columns(3)
c1.caption(f"📊 Orders CI: **{order_lo:.1f} – {order_hi:.1f}**")
c2.caption(f"📊 Revenue CI: **₹{rev_lo:,.0f} – ₹{rev_hi:,.0f}**")
c3.caption(f"📊 ROAS range: **{roas_lo:.2f}x – {roas_hi:.2f}x**")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "  Bid Optimization  ", "  Scenario Sim  ", "  Budget Allocator  ",
    "  Market Heatmap  ",   "  Segments  ",     "  Insights  ", "  Model Info  ",
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — BID OPTIMIZATION
# ═══════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### 🎯 CPC Recommendation")
        st.markdown(f"**Recommended for {sel_cat} · {sel_city}**")
        st.markdown(
            f"<h2 style='color:{BLUE};margin:4px 0;font-family:Inter'>₹{rec['rec_min']} – ₹{rec['rec_max']}</h2>",
            unsafe_allow_html=True)
        st.caption(f"From **{rec['n']:,}** high-ROAS campaigns  ·  Avg ROAS: **{rec['avg_roas']}x**")

        r1, r2, r3 = st.columns(3)
        r1.metric("Avg CPC",    f"₹{rec['avg_cpc']}")
        r2.metric("Median CPC", f"₹{rec['median_cpc']}")
        r3.metric("Top-Q CPC",  f"₹{rec['q75_cpc']}")

        st.markdown("---")
        st.markdown("#### ⚙️ Bid Optimizer")
        st.caption("Finds the CPC that maximises ROAS within your budget")

        with st.spinner("Optimizing..."):
            best, opt_df = optimize_cpc(
                impr, ctr, cvr, aov, budget, order_model, rev_model,
                cat_val, city_val, stats["med_rating"], stats["med_disc"])

        o1, o2, o3 = st.columns(3)
        o1.metric("Optimal CPC",     f"₹{best.get('cpc','—')}")
        o2.metric("Expected ROAS",   f"{best.get('roas',0):.2f}x")
        o3.metric("Expected Orders", f"{best.get('orders',0):.1f}")
        st.caption(
            f"At ₹{best.get('cpc','—')} CPC → "
            f"Spend ₹{best.get('spend',0):,.0f}  ·  Revenue ₹{best.get('revenue',0):,.0f}")

        be_df = opt_df[opt_df["roas"] >= 1.0]
        if not be_df.empty:
            st.info(f"⚖️ Break-even CPC: **₹{be_df['cpc'].max():.0f}** — do not bid above this.")

        st.markdown("---")
        st.markdown("#### 📊 Scenario Comparison")
        s1, s2 = st.columns(2)
        s1.metric("Current ROAS",    f"{roas:.2f}x")
        s1.metric("Current Revenue", f"₹{revenue:,.0f}")
        s2.metric("Optimal ROAS",    f"{best.get('roas',0):.2f}x",
                  delta=f"{best.get('roas',0)-roas:+.2f}")
        s2.metric("Optimal Revenue", f"₹{best.get('revenue',0):,.0f}",
                  delta=f"₹{best.get('revenue',0)-revenue:+,.0f}")

    with col_r:
        st.markdown("#### 📉 ROAS vs CPC Curve")
        cpc_rng = np.linspace(stats["min_cpc"], stats["max_cpc"], 120)
        cpc_df  = simulate_over_cpc_range(
            cpc_rng, impr, ctr, cvr, aov, order_model, rev_model,
            cat_val, city_val, stats["med_rating"], stats["med_disc"])

        cpc_df["roas_lo"] = (cpc_df["revenue"] - rev_std) / cpc_df["spend"].clip(lower=0.01)
        cpc_df["roas_hi"] = (cpc_df["revenue"] + rev_std) / cpc_df["spend"].clip(lower=0.01)

        fig = go.Figure()
        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(cpc_df["cpc"]) + list(cpc_df["cpc"][::-1]),
            y=list(cpc_df["roas_hi"]) + list(cpc_df["roas_lo"][::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip"))
        # Main curve
        fig.add_trace(go.Scatter(
            x=cpc_df["cpc"], y=cpc_df["roas"], mode="lines",
            line=dict(color=BLUE, width=2.5), name="Predicted ROAS"))
        # Markers
        fig.add_vline(x=cpc, line_dash="dot", line_color=AMBER, line_width=2,
                      annotation_text=f" Current Rs.{cpc}",
                      annotation_font=dict(color=AMBER, size=11))
        if best.get("cpc"):
            fig.add_vline(x=best["cpc"], line_dash="dash", line_color=GREEN, line_width=2,
                          annotation_text=f" Optimal Rs.{best['cpc']}",
                          annotation_font=dict(color=GREEN, size=11))
        fig.add_hline(y=4, line_dash="dot", line_color="#94a3b8", line_width=1.2,
                      annotation_text=" Target 4x", annotation_position="right",
                      annotation_font=dict(color="#94a3b8", size=10))
        fig.add_hline(y=1, line_dash="dot", line_color=RED, line_width=1.2,
                      annotation_text=" Break-even", annotation_position="right",
                      annotation_font=dict(color=RED, size=10))
        chart(fig, f"CPC vs ROAS — {sel_cat} · {sel_city}", 400, show_legend=True)
        fig.update_layout(xaxis_title="CPC (Rs.)", yaxis_title="Predicted ROAS")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — SCENARIO SIMULATION
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### What-If Simulation")
    st.caption("All outputs from trained ML models — not formulas.")

    col1, col2 = st.columns(2)

    with col1:
        ctr_rng = np.arange(1.0, 8.1, 0.15)
        ctr_df  = simulate_over_ctr_range(
            ctr_rng, impr, cpc, cvr, aov, order_model, rev_model,
            cat_val, city_val, stats["med_rating"], stats["med_disc"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ctr_df["ctr"], y=ctr_df["roas"], mode="lines",
            line=dict(color=PURPLE, width=2.5),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.07)"))
        fig2.add_trace(go.Scatter(x=[ctr], y=[roas], mode="markers",
            marker=dict(size=11, color=PURPLE, line=dict(color="white", width=2.5)),
            showlegend=False))
        fig2.add_vline(x=ctr, line_dash="dot", line_color=AMBER, line_width=1.8,
                       annotation_text=f" CTR {ctr}%",
                       annotation_font=dict(color=AMBER, size=11))
        chart(fig2, "Predicted ROAS vs CTR", 280)
        fig2.update_layout(xaxis_title="CTR (%)", yaxis_title="ROAS", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cpc_df["cpc"], y=cpc_df["orders"], mode="lines",
            line=dict(color=GREEN, width=2.5),
            fill="tozeroy", fillcolor="rgba(22,163,74,0.07)"))
        fig3.add_vline(x=cpc, line_dash="dot", line_color=AMBER, line_width=1.8,
                       annotation_text=f" CPC Rs.{cpc}",
                       annotation_font=dict(color=AMBER, size=11))
        chart(fig3, "Predicted Orders vs CPC (Bid Response)", 280)
        fig3.update_layout(xaxis_title="CPC (Rs.)", yaxis_title="Predicted Orders", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Sensitivity tornado
    sens = sensitivity_analysis(
        impr, ctr, cvr, cpc, aov, order_model, rev_model,
        cat_val, city_val, stats["med_rating"], stats["med_disc"])
    sens = sens.sort_values("Delta")
    sens["Color"] = sens["Delta"].apply(lambda x: GREEN if x >= 0 else RED)

    fig4 = go.Figure(go.Bar(
        x=sens["Delta"], y=sens["Scenario"], orientation="h",
        marker=dict(color=sens["Color"], opacity=0.88, line=dict(width=0)),
        text=[f"{r:.2f}x" for r in sens["ROAS"]],
        textposition="outside", textfont=dict(size=11, color="#334155")))
    fig4.add_vline(x=0, line_color="#cbd5e1", line_width=1.5)
    chart(fig4, "Sensitivity Analysis — ROAS impact of ±20% change", 300)
    fig4.update_layout(xaxis_title="Δ ROAS", yaxis_title="")
    st.plotly_chart(fig4, use_container_width=True)

    # CVR × AOV Heatmap
    cv_v = np.arange(0.5, 6.1, 0.5)
    av_v = np.arange(500, 10100, 500)
    Z = build_roas_heatmap(ctr, cpc, impr, cv_v, av_v,
                           order_model, rev_model, cat_val, city_val,
                           stats["med_rating"], stats["med_disc"])
    fig5 = go.Figure(go.Heatmap(
        z=Z, x=[f"Rs.{int(a):,}" for a in av_v], y=[f"{c:.1f}%" for c in cv_v],
        colorscale=CSCALE,
        colorbar=dict(title="ROAS", tickfont=dict(size=10, color="#64748b")),
        hovertemplate="CVR: %{y}<br>AOV: %{x}<br>ROAS: %{z:.2f}x<extra></extra>",
        text=np.round(Z, 1), texttemplate="%{text}",
        textfont=dict(size=8, color="#1e293b")))
    chart(fig5, f"ROAS Heatmap — CVR × AOV  (CTR {ctr}%, CPC ₹{cpc})", 310)
    fig5.update_layout(xaxis_title="Average Order Value (Rs.)",
                       yaxis_title="Conversion Rate (%)",
                       paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — BUDGET ALLOCATOR
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Multi-Segment Budget Allocation")
    st.caption("Select segments, set budget → optimizer splits it to maximise blended ROAS.")

    ba1, ba2 = st.columns(2)
    with ba1:
        sel_cats_m  = st.multiselect("Categories", CATEGORIES,
                                     default=["Electronics","Fashion","Beauty"])
    with ba2:
        sel_cities_m = st.multiselect("Cities", CITIES,
                                      default=["Mumbai","Delhi","Bengaluru"])
    total_bgt = st.slider("Total Budget (₹)", 10_000, 500_000, 100_000, 5_000)

    if st.button("▶ Run Budget Optimizer", type="primary"):
        if not sel_cats_m or not sel_cities_m:
            st.warning("Select at least one category and one city.")
        else:
            with st.spinner("Optimizing across segments..."):
                alloc = []
                for cat in sel_cats_m:
                    for city in sel_cities_m:
                        c_v, ci_v = encode_single(cat, city, cat_enc, city_enc)
                        best_r, best_c = -1, stats["med_cpc"]
                        for test_cpc in np.linspace(stats["min_cpc"], stats["max_cpc"], 50):
                            r = simulate_funnel(50_000, stats["med_ctr"], stats["med_cvr"],
                                                test_cpc, stats["med_aov"],
                                                order_model, rev_model, c_v, ci_v,
                                                stats["med_rating"], stats["med_disc"])
                            if r["roas"] > best_r:
                                best_r = r["roas"]
                                best_c = round(test_cpc, 2)
                        alloc.append({"Category":cat,"City":city,
                                      "Opt CPC (₹)":best_c,
                                      "Est ROAS":round(best_r,2),
                                      "Score":round(best_r,3)})
                alloc_df = pd.DataFrame(alloc)
                total_score = alloc_df["Score"].sum()
                alloc_df["Budget (₹)"] = (alloc_df["Score"]/total_score*total_bgt).round(0).astype(int)
                alloc_df["Est Revenue (₹)"] = (alloc_df["Budget (₹)"]*alloc_df["Est ROAS"]).round(0).astype(int)
                alloc_df = alloc_df.drop(columns=["Score"]).sort_values("Est ROAS",ascending=False)

            k1,k2,k3 = st.columns(3)
            k1.metric("Total Budget",      f"₹{total_bgt:,}")
            k2.metric("Est Total Revenue", f"₹{alloc_df['Est Revenue (₹)'].sum():,}")
            k3.metric("Blended ROAS",
                      f"{alloc_df['Est Revenue (₹)'].sum()/total_bgt:.2f}x")

            st.dataframe(
                alloc_df.style
                .format({"Est ROAS":"{:.2f}","Budget (₹)":"₹{:,}","Est Revenue (₹)":"₹{:,}"})
                .background_gradient(subset=["Est ROAS"], cmap="Blues"),
                use_container_width=True, hide_index=True)

            fig_tree = px.treemap(
                alloc_df, path=["Category","City"],
                values="Budget (₹)", color="Est ROAS",
                color_continuous_scale=CSCALE)
            fig_tree.update_layout(paper_bgcolor="#ffffff", height=360,
                                   margin=dict(t=10,b=10,l=10,r=10),
                                   font=dict(color="#1e293b",family="Inter"))
            fig_tree.update_traces(textfont=dict(size=12))
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("Configure segments above, then click **Run Budget Optimizer**.")


# ═══════════════════════════════════════════════════════════
# TAB 4 — MARKET HEATMAP
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Category × City ROAS Heatmap")
    st.caption("Average ROAS across all 79,000 campaigns. Darker = better.")

    pivot = agg["pivot"]
    fig_h = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=CSCALE,
        text=pivot.values.round(2),
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=12, color="#1e293b"),
        colorbar=dict(title="Avg ROAS", tickfont=dict(size=10,color="#64748b")),
        hovertemplate="<b>%{y} × %{x}</b><br>Avg ROAS: %{z:.2f}x<extra></extra>"))
    fig_h.update_layout(
        paper_bgcolor="#ffffff", height=380,
        margin=dict(t=30, b=60, l=150, r=20),
        font=dict(color="#334155", size=11, family="Inter"),
        xaxis=dict(tickfont=dict(size=11,color="#334155"), side="bottom",
                   showgrid=False, linecolor="#e2e8f0"),
        yaxis=dict(tickfont=dict(size=11,color="#334155"), showgrid=False,
                   linecolor="#e2e8f0"))
    st.plotly_chart(fig_h, use_container_width=True)

    flat = pivot.stack().reset_index()
    flat.columns = ["Category","City","Avg ROAS"]
    flat = flat.sort_values("Avg ROAS", ascending=False)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**🏆 Top 5 Segments**")
        st.dataframe(flat.head(5).style.format({"Avg ROAS":"{:.2f}"}),
                     use_container_width=True, hide_index=True)
    with b2:
        st.markdown("**⚠️ Bottom 5 Segments**")
        st.dataframe(flat.tail(5).style.format({"Avg ROAS":"{:.2f}"}),
                     use_container_width=True, hide_index=True)

    st.markdown("#### CTR vs CVR by Category")
    cat_s = agg["cat_stats"]
    fig_sc = px.scatter(cat_s, x="avg_ctr", y="avg_cvr", size="campaigns",
                        color="Category", text="Category",
                        color_discrete_sequence=COLORS, size_max=45)
    fig_sc.update_traces(textposition="top center", textfont=dict(size=11))
    chart(fig_sc, "CTR vs CVR by Category  (bubble = campaign count)", 320, False)
    fig_sc.update_layout(xaxis_title="Avg CTR (%)", yaxis_title="Avg CVR (%)",
                         paper_bgcolor="#ffffff", plot_bgcolor="#fafbfc")
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 5 — CAMPAIGN SEGMENTS (K-Means)
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### K-Means Campaign Segmentation")
    st.caption("4 performance tiers from clustering 79,000 campaigns on CTR, CVR, CPC, ROAS, AOV.")

    cluster_summary = agg["cluster_summary"]
    cluster_sample  = agg["cluster_sample"]

    seg_colors = {
        "🏆 High Performers": GREEN,
        "📈 Growth Potential": BLUE,
        "⚠️ Needs Attention":  AMBER,
        "🔴 Underperforming":  RED,
    }

    cols = st.columns(4)
    for i, row in cluster_summary.iterrows():
        lbl, col = roas_label(row["Avg_ROAS"])
        cols[i % 4].metric(
            row["Segment"], f"ROAS {row['Avg_ROAS']:.2f}x",
            f"{int(row['Campaigns']):,} campaigns")

    sc1, sc2 = st.columns(2)

    with sc1:
        fig_box = go.Figure()
        for seg, color in seg_colors.items():
            sub = cluster_sample[cluster_sample["Segment"] == seg]["ROAS"]
            if len(sub) == 0:
                continue
            fig_box.add_trace(go.Box(
                y=sub, name=seg, marker_color=color,
                boxmean=True, line=dict(width=1.5),
                fillcolor="rgba(37,99,235,0.08)"))
        chart(fig_box, "ROAS Distribution by Segment", 340, True)
        fig_box.update_layout(
            showlegend=True, xaxis_title="", yaxis_title="ROAS",
            paper_bgcolor="#ffffff", plot_bgcolor="#fafbfc",
            xaxis=dict(tickfont=dict(size=9, color="#334155"), showgrid=False))
        st.plotly_chart(fig_box, use_container_width=True)

    with sc2:
        color_map = {seg: color for seg, color in seg_colors.items()}
        fig_sc2 = px.scatter(
            cluster_sample, x="CTR", y="CVR",
            color="Segment", color_discrete_map=color_map,
            opacity=0.65, size_max=6)
        fig_sc2.update_traces(marker=dict(size=5, line=dict(width=0)))
        chart(fig_sc2, "CTR vs CVR Coloured by Segment  (800 sample)", 340, True)
        fig_sc2.update_layout(xaxis_title="CTR (%)", yaxis_title="CVR (%)",
                              paper_bgcolor="#ffffff", plot_bgcolor="#fafbfc")
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown("#### Segment Summary")
    st.dataframe(
        cluster_summary.rename(columns={
            "Avg_ROAS":"Avg ROAS","Avg_CTR":"Avg CTR %","Avg_CVR":"Avg CVR %",
            "Avg_CPC":"Avg CPC (₹)","Avg_Revenue":"Avg Revenue (₹)"})
        .style.format({
            "Avg ROAS":"{:.2f}","Avg CTR %":"{:.2f}","Avg CVR %":"{:.2f}",
            "Avg CPC (₹)":"₹{:.1f}","Avg Revenue (₹)":"₹{:,.0f}"})
        .background_gradient(subset=["Avg ROAS"], cmap="Blues"),
        use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS
# ═══════════════════════════════════════════════════════════
with tab6:
    st.markdown("#### Actionable Insights")
    st.caption(f"Based on your inputs vs benchmarks for **{sel_cat}** in **{sel_city}**")

    insights = generate_insights(roas, ctr, cvr, cpc, rec)
    for kind, title, body in insights:
        css = {"good":"ins-good","warn":"ins-warn","red":"ins-bad"}.get(kind,"ins-warn")
        st.markdown(
            f"<div class='{css}'>"
            f"<p><strong>{title}</strong></p>"
            f"<p style='margin-top:4px'>{body}</p>"
            f"</div>",
            unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROAS Distribution")
        # Use pre-aggregated histogram — no raw data sent to browser
        hist = agg["hist_data"][sel_cat]
        counts = hist["counts"]
        bins   = hist["bins"]
        bin_centers = [(bins[i]+bins[i+1])/2 for i in range(len(counts))]
        fig_hist = go.Figure(go.Bar(
            x=bin_centers, y=counts,
            marker=dict(color=BLUE, opacity=0.75,
                        line=dict(color="#1d4ed8", width=0.5)),
            hovertemplate="ROAS: %{x:.1f}x<br>Campaigns: %{y}<extra></extra>"))
        fig_hist.add_vline(x=roas, line_dash="dash", line_color=RED, line_width=2,
                           annotation_text=f" Yours {roas:.1f}x",
                           annotation_font=dict(color=RED,size=11))
        fig_hist.add_vline(x=base_roas, line_dash="dot", line_color=AMBER, line_width=1.8,
                           annotation_text=f" Median {base_roas:.1f}x",
                           annotation_font=dict(color=AMBER,size=10))
        chart(fig_hist, f"ROAS Distribution — {sel_cat}", 300)
        fig_hist.update_layout(xaxis_title="ROAS", yaxis_title="Campaigns",
                               bargap=0.05, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("#### Category Benchmark")
        cat_s = agg["cat_stats"].sort_values("avg_roas", ascending=True)
        bar_colors = [RED if c == sel_cat else "#93c5fd" for c in cat_s["Category"]]
        fig_cat = go.Figure(go.Bar(
            x=cat_s["avg_roas"], y=cat_s["Category"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=cat_s["avg_roas"].round(2), textposition="outside",
            textfont=dict(size=11, color="#334155"),
            hovertemplate="%{y}: %{x:.2f}x ROAS<extra></extra>"))
        chart(fig_cat, "Avg ROAS by Category  (red = selected)", 300)
        fig_cat.update_layout(xaxis_title="Avg ROAS", yaxis_title="")
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("#### Feature Importance")
    fi1, fi2 = st.columns(2)
    with fi1:
        imp_ord = (pd.DataFrame(list(feat_imp["order"].items()),
                               columns=["Feature","Importance"])
                   .sort_values("Importance", ascending=True))
        fig_fi = go.Figure(go.Bar(
            x=imp_ord["Importance"], y=imp_ord["Feature"], orientation="h",
            marker=dict(color=imp_ord["Importance"], colorscale=CSCALE, showscale=False,
                        line=dict(width=0)),
            text=imp_ord["Importance"].round(3), textposition="outside",
            textfont=dict(size=10, color="#334155")))
        chart(fig_fi, "Order Model — Feature Importance", 300)
        fig_fi.update_layout(xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

    with fi2:
        imp_rev = (pd.DataFrame(list(feat_imp["revenue"].items()),
                               columns=["Feature","Importance"])
                   .sort_values("Importance", ascending=True))
        fig_fi2 = go.Figure(go.Bar(
            x=imp_rev["Importance"], y=imp_rev["Feature"], orientation="h",
            marker=dict(color=imp_rev["Importance"],
                        colorscale=[[0,"#d1fae5"],[1,"#065f46"]], showscale=False,
                        line=dict(width=0)),
            text=imp_rev["Importance"].round(3), textposition="outside",
            textfont=dict(size=10, color="#334155")))
        chart(fig_fi2, "Revenue Model — Feature Importance", 300)
        fig_fi2.update_layout(xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig_fi2, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 7 — MODEL INFO
# ═══════════════════════════════════════════════════════════
with tab7:
    st.markdown("#### Model Performance")
    m1,m2,m3 = st.columns(3)
    m1.metric("Order R²",   f"{metrics['order']['r2']}")
    m2.metric("Revenue R²", f"{metrics['revenue']['r2']}")
    m3.metric("Bid R²",     f"{metrics['bid']['r2']}")
    d1,d2,d3 = st.columns(3)
    d1.metric("Order RMSE",   f"{metrics['order']['rmse']:.2f} orders")
    d2.metric("Revenue RMSE", f"₹{metrics['revenue']['rmse']:,.0f}")
    d3.metric("Bid RMSE",     f"{metrics['bid']['rmse']:.0f} clicks")

    st.divider()
    arch = pd.DataFrame([
        ["Order Model",  "GradientBoosting/XGBoost",
         "CTR, CVR, CPC, AOV, Impressions, Rating, Discount, Category, City, Clicks","Orders"],
        ["Revenue Model","GradientBoosting/XGBoost",
         "Predicted Orders, AOV, Category, Discount, Rating","Revenue"],
        ["Bid Response", "GradientBoosting/XGBoost",
         "CPC, Impressions, Category, City, Rating","Clicks"],
    ], columns=["Model","Algorithm","Input Features","Target"])
    st.dataframe(arch, use_container_width=True, hide_index=True)

    st.divider()
    pc1,pc2 = st.columns(2)
    pc1.metric("Order Residual Std",   f"±{ord_std:.2f} orders")
    pc2.metric("Revenue Residual Std", f"±₹{rev_std:,.0f}")
    st.info(
        "**ROAS is never directly predicted.**  \n"
        "Funnel: **Impressions → Clicks → Orders → Revenue → ROAS**  \n"
        "Orders and Revenue are predicted separately.  \n"
        "ROAS = `Predicted Revenue ÷ Ad Spend` — derived, not modelled.")
