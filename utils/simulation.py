"""
simulation.py  —  What-if simulation engine.
All ROAS is DERIVED from predicted orders/revenue, never predicted directly.
"""
import numpy as np
import pandas as pd


def simulate_funnel(impressions, ctr, cvr, cpc, aov,
                    order_model=None, rev_model=None,
                    cat_enc_val=0, city_enc_val=0,
                    rating=3.0, discount=0.2):
    """
    Compute full advertising funnel given inputs.
    Uses ML models if provided, otherwise falls back to formulas.

    Funnel:
        Clicks  = Impressions × CTR
        Spend   = Clicks × CPC
        Orders  = model(features)  or  Clicks × CVR
        Revenue = model(features)  or  Orders × AOV
        ROAS    = Revenue / Spend
    """
    clicks = impressions * (ctr / 100)
    spend  = clicks * cpc

    if order_model is not None:
        X_ord = np.array([[ctr, cvr, cpc, aov, impressions,
                            rating, discount, cat_enc_val, city_enc_val, clicks]])
        orders = max(float(order_model.predict(X_ord)[0]), 0)
    else:
        orders = clicks * (cvr / 100)

    if rev_model is not None:
        X_rev = np.array([[orders, aov, cat_enc_val, discount, rating]])
        revenue = max(float(rev_model.predict(X_rev)[0]), 0)
    else:
        revenue = orders * aov

    roas = revenue / spend if spend > 0 else 0.0

    return dict(clicks=round(clicks), orders=round(orders,2),
                spend=round(spend,2), revenue=round(revenue,2), roas=round(roas,3))


def simulate_over_cpc_range(cpc_values, impressions, ctr, cvr, aov,
                             order_model=None, rev_model=None,
                             cat_enc_val=0, city_enc_val=0,
                             rating=3.0, discount=0.2) -> pd.DataFrame:
    rows = []
    for cpc in cpc_values:
        r = simulate_funnel(impressions, ctr, cvr, cpc, aov,
                            order_model, rev_model, cat_enc_val, city_enc_val,
                            rating, discount)
        r["cpc"] = round(cpc, 2)
        rows.append(r)
    return pd.DataFrame(rows)


def simulate_over_ctr_range(ctr_values, impressions, cpc, cvr, aov,
                             order_model=None, rev_model=None,
                             cat_enc_val=0, city_enc_val=0,
                             rating=3.0, discount=0.2) -> pd.DataFrame:
    rows = []
    for ctr in ctr_values:
        r = simulate_funnel(impressions, ctr, cvr, cpc, aov,
                            order_model, rev_model, cat_enc_val, city_enc_val,
                            rating, discount)
        r["ctr"] = round(ctr, 2)
        rows.append(r)
    return pd.DataFrame(rows)


def sensitivity_analysis(impressions, ctr, cvr, cpc, aov,
                          order_model=None, rev_model=None,
                          cat_enc_val=0, city_enc_val=0,
                          rating=3.0, discount=0.2) -> pd.DataFrame:
    """One-way ±20% sensitivity on CTR, CVR, CPC, AOV."""
    base = simulate_funnel(impressions, ctr, cvr, cpc, aov,
                           order_model, rev_model, cat_enc_val, city_enc_val,
                           rating, discount)
    base_roas = base["roas"]
    rows = []
    params = dict(ctr=ctr, cvr=cvr, cpc=cpc, aov=aov)
    for p, val in params.items():
        for mult, lbl in [(1.2,"+20%"),(0.8,"-20%")]:
            args = dict(impressions=impressions, ctr=ctr, cvr=cvr,
                        cpc=cpc, aov=aov, order_model=order_model,
                        rev_model=rev_model, cat_enc_val=cat_enc_val,
                        city_enc_val=city_enc_val, rating=rating, discount=discount)
            args[p] = val * mult
            r = simulate_funnel(**args)
            rows.append(dict(
                Scenario=f"{p.upper()}  {lbl}",
                ROAS=r["roas"], Delta=round(r["roas"]-base_roas,3),
                Orders=r["orders"], Revenue=r["revenue"]
            ))
    return pd.DataFrame(rows)


def build_roas_heatmap(ctr, cpc, impressions,
                       cvr_range, aov_range,
                       order_model=None, rev_model=None,
                       cat_enc_val=0, city_enc_val=0,
                       rating=3.0, discount=0.2):
    """Return 2D array: rows=CVR, cols=AOV."""
    Z = np.zeros((len(cvr_range), len(aov_range)))
    for i, cv in enumerate(cvr_range):
        for j, av in enumerate(aov_range):
            r = simulate_funnel(impressions, ctr, cv, cpc, av,
                                order_model, rev_model,
                                cat_enc_val, city_enc_val, rating, discount)
            Z[i, j] = r["roas"]
    return Z
