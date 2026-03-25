"""
feature_engineering.py  —  Data loading, cleaning, encoding, feature definitions.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

ORDER_FEATURES   = ["CTR","CVR","CPC","AOV","impressions",
                    "rating","discount_pct","cat_enc","city_enc","clicks"]
REVENUE_FEATURES = ["orders_pred","AOV","cat_enc","discount_pct","rating"]
BID_FEATURES     = ["CPC","impressions","cat_enc","city_enc","rating"]

CATEGORIES = ["Appliances","Beauty","Electronics","Fashion",
              "Home & Kitchen","Mobiles","Sports","Toys"]
CITIES     = ["Ahmedabad","Bengaluru","Chennai","Delhi",
              "Hyderabad","Kolkata","Mumbai","Pune"]


def load_and_merge(campaign_path: str, seller_path: str) -> pd.DataFrame:
    camp = pd.read_csv(campaign_path)
    sell = pd.read_excel(seller_path)
    for c in ["CTR","CVR","CTR HELPER","CVR Helper","efficiency score"]:
        if c in camp.columns:
            camp[c] = (camp[c].astype(str)
                       .str.replace("%","",regex=False).str.strip()
                       .replace("nan",np.nan).astype(float))
    sell.drop(columns=[c for c in sell.columns if c.startswith("Unnamed")],inplace=True)
    df = camp.merge(sell, on="product_id", how="left")
    df["CPC"]             = df["CPC Helper"]
    df["AOV"]             = df[" Price"].fillna(df["retail_price"])
    df["city"]            = df["seller_city"]
    df["sentiment_score"] = df["sentiment socre"].fillna(0.5)
    df["discount_pct"]    = df["discount_pct"].fillna(0.2)
    df["rating"]          = df["rating"].fillna(3.0)
    df["review_count"]    = df["review_count"].fillna(100)
    df = df.dropna(subset=["CTR","CVR","CPC","AOV","orders","revenue"])
    df = df[(df["orders"]>0)&(df["revenue"]>0)&(df["CPC"]>0)].reset_index(drop=True)
    return df


def encode_features(df: pd.DataFrame):
    cat_enc  = LabelEncoder().fit(CATEGORIES)
    city_enc = LabelEncoder().fit(CITIES)
    df = df.copy()
    df["cat_enc"]  = cat_enc.transform(df["Category"].astype(str))
    df["city_enc"] = city_enc.transform(df["city"].astype(str))
    return df, cat_enc, city_enc


def encode_single(category, city, cat_enc, city_enc):
    try:    c  = int(cat_enc.transform([category])[0])
    except: c  = 0
    try:    ci = int(city_enc.transform([city])[0])
    except: ci = 0
    return c, ci


def get_data_stats(df: pd.DataFrame) -> dict:
    return dict(
        med_ctr    = round(float(df["CTR"].median()),1),
        med_cvr    = round(float(df["CVR"].median()),1),
        med_cpc    = int(df["CPC"].median()),
        med_aov    = int(round(df["AOV"].median(),-2)),
        med_impr   = int(round(df["impressions"].median(),-3)),
        med_rating = round(float(df["rating"].median()),1),
        med_disc   = round(float(df["discount_pct"].median()),2),
        min_cpc    = int(df["CPC"].min()),
        max_cpc    = int(df["CPC"].max()),
    )


def get_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("Category")
              .agg(avg_cpc=("CPC","mean"), med_cpc=("CPC","median"),
                   q75_cpc=("CPC",lambda x:x.quantile(0.75)),
                   avg_roas=("ROAS","mean"), avg_ctr=("CTR","mean"),
                   avg_cvr=("CVR","mean"), campaigns=("campaign_id","count"))
              .reset_index().round(2))


def get_city_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("city")
              .agg(avg_cpc=("CPC","mean"), med_cpc=("CPC","median"),
                   avg_roas=("ROAS","mean"), campaigns=("campaign_id","count"))
              .reset_index().round(2))
