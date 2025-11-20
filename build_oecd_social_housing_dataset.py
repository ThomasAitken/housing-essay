import io
import sys
import json
import math
import zipfile
from pathlib import Path

import pandas as pd
import requests

PH42_URL = "https://webfs.oecd.org/Els-com/Affordable_Housing_Database/PH4-2-Social-rental-housing-stock.xlsx"
OUT_CSV = Path("oecd_social_housing_2000on.csv")

# ISO3 helper (minimal map; extend if needed)
ISO3 = {
    "Australia":"AUS","Austria":"AUT","Belgium":"BEL","Canada":"CAN","Chile":"CHL","Colombia":"COL","Czechia":"CZE",
    "Denmark":"DNK","Estonia":"EST","Finland":"FIN","France":"FRA","Germany":"DEU","Greece":"GRC","Hungary":"HUN",
    "Iceland":"ISL","Ireland":"IRL","Israel":"ISR","Italy":"ITA","Japan":"JPN","Korea":"KOR","Latvia":"LVA",
    "Lithuania":"LTU","Luxembourg":"LUX","Mexico":"MEX","Netherlands":"NLD","New Zealand":"NZL","Norway":"NOR",
    "Poland":"POL","Portugal":"PRT","Slovak Republic":"SVK","Slovenia":"SVN","Spain":"ESP","Sweden":"SWE",
    "Switzerland":"CHE","Türkiye":"TUR","United Kingdom (England)":"GBR-ENG","United States":"USA"
}

def main():
    # 1) Download the OECD PH4.2 workbook
    resp = requests.get(PH42_URL, timeout=60)
    resp.raise_for_status()
    data = io.BytesIO(resp.content)

    # 2) Inspect sheets; OECD typically includes a tidy data sheet and an annex
    xls = pd.ExcelFile(data)
    sheets = {name.lower(): name for name in xls.sheet_names}

    # Try common possibilities
    guess_sheets = [
        "ph4.2", "ph4-2", "data", "ph4_2", "country time series", "timeseries", "annex", "ph4.2.a1"
    ]

    def pick_sheet(candidates):
        for c in candidates:
            for key, name in sheets.items():
                if c in key:
                    return name
        # fallback to first sheet
        return xls.sheet_names[0]

    data_sheet = pick_sheet(guess_sheets)
    df_raw = pd.read_excel(xls, sheet_name=data_sheet, header=0)

    # 3) Tidy: find columns that look like Country, Year, Share (%), Absolute stock
    # We try flexible matching to handle future header tweaks.
    def find_col(df, candidates):
        for c in candidates:
            for col in df.columns:
                if str(col).strip().lower() == c:
                    return col
        for col in df.columns:
            lc = str(col).strip().lower()
            if any(c in lc for c in candidates):
                return col
        return None

    country_col = find_col(df_raw, {"country","country name","economy"})
    year_col    = find_col(df_raw, {"year","time","reference year"})
    share_col   = find_col(df_raw, {"share of total dwellings","% of total","social share","share (%)"})
    abs_col     = find_col(df_raw, {"number of social dwellings","absolute","stock (units)","dwellings"})

    if country_col is None or year_col is None or share_col is None:
        raise RuntimeError("Could not detect expected columns in PH4.2 workbook. Inspect column names: "
                           + ", ".join(map(str, df_raw.columns)))

    df = df_raw.rename(columns={
        country_col: "country",
        year_col: "year",
        share_col: "social_share_pct",
        **({abs_col: "social_units_abs"} if abs_col else {})
    })

    # Normalize year (handle strings like "around 2010")
    def to_year(v):
        if pd.isna(v):
            return None
        s = str(v)
        # extract first 4-digit year we find
        import re
        m = re.search(r"(20|19)\d{2}", s)
        return int(m.group(0)) if m else None

    df["year"] = df["year"].map(to_year)
    df = df[~df["year"].isna()]
    df["year"] = df["year"].astype(int)

    # Limit to 2000 onwards (21st century)
    df = df[df["year"] >= 2000].copy()

    # Clean share to numeric
    def to_num(x):
        try:
            return float(str(x).replace("%","").strip())
        except:
            return None
    df["social_share_pct"] = df["social_share_pct"].map(to_num)

    if "social_units_abs" in df.columns:
        df["social_units_abs"] = pd.to_numeric(df["social_units_abs"], errors="coerce")

    # Add ISO3
    df["iso3"] = df["country"].map(ISO3)

    # Compute per-country change metrics
    def add_changes(g):
        g = g.sort_values("year").copy()
        first = g["year"].min()
        last  = g["year"].max()
        g["first_year"] = first
        g["last_year"]  = last

        # anchor near-2000 baseline (first obs >=2000)
        base = g.loc[g["year"] == first, "social_share_pct"].iloc[0]
        latest = g.loc[g["year"] == last, "social_share_pct"].iloc[0]

        g["pp_change_since_2000"] = latest - base
        g["pp_change_first_to_last"] = latest - base
        yrs = max(1, last - first)
        g["annual_pp_change"] = (latest - base) / yrs
        return g

    out = df.groupby("country", group_keys=False).apply(add_changes)

    # Sort and output
    out_cols = ["country","iso3","year","social_share_pct"]
    if "social_units_abs" in out.columns:
        out_cols.append("social_units_abs")
    out_cols += ["first_year","last_year","pp_change_since_2000","pp_change_first_to_last","annual_pp_change"]

    out = out[out_cols].sort_values(["country","year"])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.resolve()} with {len(out)} rows across {out['country'].nunique()} countries.")

if __name__ == "__main__":
    main()
