"""
optimization.py  —  Bid optimization, CPC recommendation, budget allocation.
"""
import numpy as np
import pandas as pd
from itertools import product


def optimize_cpc(impressions, ctr, cvr, aov, budget,
                 order_model=None, rev_model=None,
                 cat_enc_val=0, city_enc_val=0,
                 rating=3.0, discount=0.2,
                 cpc_min=2, cpc_max=60, steps=300):
    from utils.simulation import simulate_funnel
    best = {"roas": -1, "cpc": cpc_min}
    rows = []
    for cpc in np.linspace(cpc_min, cpc_max, steps):
        r = simulate_funnel(impressions, ctr, cvr, cpc, aov,
                            order_model, rev_model,
                            cat_enc_val, city_enc_val, rating, discount)
        r["cpc"] = round(cpc, 2)
        rows.append(r)
        if r["spend"] <= budget and r["roas"] > best["roas"]:
            best = r.copy()
    return best, pd.DataFrame(rows)


def get_cpc_recommendation(df, category, city, roas_threshold=4.0):
    mask = pd.Series([True] * len(df), index=df.index)
    if category != "All": mask &= (df["Category"] == category)
    if city      != "All": mask &= (df["city"]     == city)
    subset = df[mask & (df["ROAS"] >= roas_threshold)]
    if len(subset) < 10: subset = df[mask]
    if len(subset) == 0: subset = df.copy()
    return dict(
        n          = len(subset),
        avg_cpc    = round(subset["CPC"].mean(),    1),
        median_cpc = round(subset["CPC"].median(),  1),
        q25_cpc    = round(subset["CPC"].quantile(0.25), 1),
        q75_cpc    = round(subset["CPC"].quantile(0.75), 1),
        avg_roas   = round(subset["ROAS"].mean(),   2),
        avg_ctr    = round(subset["CTR"].mean(),    2),
        avg_cvr    = round(subset["CVR"].mean(),    2),
        rec_min    = round(subset["CPC"].quantile(0.25), 1),
        rec_max    = round(subset["CPC"].quantile(0.75), 1),
    )


def roas_label(roas):
    if roas < 2:   return "Poor",             "#ef4444"
    elif roas < 4: return "Moderate",         "#f59e0b"
    else:          return "High Performance", "#16a34a"


def generate_insights(roas, ctr, cvr, cpc, rec):
    tips = []
    if roas < 2:
        tips.append(("red", "🔴 ROAS is below break-even.",
                     "Reduce CPC or improve CVR to restore profitability."))
    elif roas < 4:
        tips.append(("warn", "🟡 ROAS is moderate.",
                     "Focus on improving CTR or CVR to push above 4x."))
    else:
        tips.append(("good", "🟢 ROAS is strong.",
                     "Consider scaling impressions to capture more volume."))
    if cpc > rec["q75_cpc"]:
        tips.append(("warn", f"💡 Your CPC (₹{cpc}) is above top quartile (₹{rec['q75_cpc']}).",
                     f"Try reducing CPC to ₹{rec['rec_min']}–₹{rec['rec_max']}."))
    if cvr < rec["avg_cvr"]:
        tips.append(("warn", f"📦 CVR ({cvr}%) is below category average ({rec['avg_cvr']}%).",
                     "Improve product images, reviews, or price competitiveness."))
    if ctr < rec["avg_ctr"]:
        tips.append(("warn", f"🎯 CTR ({ctr}%) is below category average ({rec['avg_ctr']}%).",
                     "Test new ad creatives or sharpen your headline copy."))
    return tips


def optimize_budget_allocation(categories, cities, total_budget,
                                impressions_per, ctr_vals, cvr_vals,
                                cpc_vals, aov_vals,
                                order_model, rev_model,
                                cat_enc, city_enc,
                                rating=3.0, discount=0.2, steps=20):
    """
    Given a list of segments (category+city combos), find the budget split
    that maximises total predicted ROAS across all segments.
    Returns a DataFrame with allocation per segment.
    """
    from utils.simulation import simulate_funnel
    from utils.feature_engineering import encode_single

    segments = []
    for cat, city in zip(categories, cities):
        c_val, ci_val = encode_single(cat, city, cat_enc, city_enc)
        best_roas, best_cpc = -1, cpc_vals[0]
        for cpc in np.linspace(min(cpc_vals), max(cpc_vals), steps):
            r = simulate_funnel(impressions_per, ctr_vals[0], cvr_vals[0],
                                cpc, aov_vals[0], order_model, rev_model,
                                c_val, ci_val, rating, discount)
            if r["roas"] > best_roas:
                best_roas = r["roas"]
                best_cpc  = round(cpc, 2)
        segments.append({
            "Category": cat, "City": city,
            "Opt CPC": best_cpc, "Est ROAS": round(best_roas, 2),
            "Priority": round(best_roas / max(best_roas, 1), 3)
        })

    seg_df = pd.DataFrame(segments)
    total_priority = seg_df["Priority"].sum()
    seg_df["Budget (₹)"] = (seg_df["Priority"] / total_priority * total_budget).round(0).astype(int)
    seg_df["Est Revenue"] = (seg_df["Budget (₹)"] * seg_df["Est ROAS"]).round(0).astype(int)
    return seg_df.sort_values("Est ROAS", ascending=False)


def build_category_city_heatmap(df):
    """Return pivot table: rows=Category, cols=City, values=avg ROAS."""
    pivot = df.pivot_table(values="ROAS", index="Category",
                           columns="city", aggfunc="mean").round(2)
    return pivot


def cluster_campaigns(df, n_clusters=4):
    """K-Means cluster campaigns into performance segments."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    feats = ["CTR", "CVR", "CPC", "ROAS", "AOV"]
    X = df[feats].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df2 = df.loc[X.index].copy()
    df2["Cluster"] = labels

    # Label clusters by avg ROAS
    cluster_roas = df2.groupby("Cluster")["ROAS"].mean().sort_values(ascending=False)
    cluster_names = {
        cluster_roas.index[0]: "🏆 High Performers",
        cluster_roas.index[1]: "📈 Growth Potential",
        cluster_roas.index[2]: "⚠️ Needs Attention",
        cluster_roas.index[3]: "🔴 Underperforming",
    }
    df2["Segment"] = df2["Cluster"].map(cluster_names)
    return df2, cluster_names
