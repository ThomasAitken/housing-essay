# pip install requests pandas tenacity
import io
from typing import List
import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------- Config --------
COUNTRIES = ["AUS","AUT","DNK","FIN","FRA","DEU","IRL","ITA",
             "JPN","KOR","NZL","NOR","ESP","SWE","GBR","USA"]
START_YEAR = 2000
OUT_CSV = "oecd_households_real_disposable_income_pc_2000_present.csv"

# OECD sector accounts (Expenditure) via DB.NOMICS (stable mirror of OECD Data Explorer)
DB_BASE = "https://api.db.nomics.world/v22"
DATASET = "OECD/DSD_NASEC10%40DF_TABLE14_EXP"

def make_series_key(ctry: str) -> str:
    """
    Annual; Households (S14); counterpart=S1; accounting entry=D (EXP table);
    transaction=B6G (Gross disposable income); instrument/exp=_Z;
    unit=XDC (national currency); valuation=S; price base=L (real, chain-linked);
    transform=N; table=T0800.
    """
    return f"A.{ctry}.S14.S1.D.B6G._Z._Z.XDC.S.L.N.T0800"

@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=0.8, min=1, max=8),
       retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)))
def fetch_dbnomics_series(series_key: str) -> pd.DataFrame:
    url = f"{DB_BASE}/series/{DATASET}/{series_key}?observations=1&format=json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    obs = (js.get("series") or {}).get("observations") or []
    df = pd.DataFrame(obs, columns=["year","value"]).dropna()
    if df.empty:
        return df
    # coerce year dtype here to avoid downstream merge issues
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).astype({"year":"int"})
    df = df[df["year"] >= START_YEAR].reset_index(drop=True)
    return df

def fetch_income_panel(countries: List[str]) -> pd.DataFrame:
    frames, errs = [], []
    for c in countries:
        try:
            df = fetch_dbnomics_series(make_series_key(c))
            if df.empty:
                errs.append({"country": c, "year": None, "income_real_households": None, "error": "no data"})
                continue
            df["country"] = c
            df = df.rename(columns={"value":"income_real_households"})[["country","year","income_real_households"]]
            frames.append(df)
        except Exception as e:
            errs.append({"country": c, "year": None, "income_real_households": None, "error": str(e)})
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["country","year","income_real_households"])
    if errs:
        panel = pd.concat([panel, pd.DataFrame(errs)], ignore_index=True)
    return panel

def fetch_oecd_population(country: str) -> pd.DataFrame:
    # Broad query; we’ll select a reasonable measure if multiple are returned
    url = f"https://stats.oecd.org/SDMX-JSON/data/DP_LIVE/A.{country}.POP.?startTime={START_YEAR}&contentType=csv"
    r = requests.get(url, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise ValueError("DP_LIVE returned empty CSV")
    # Prefer absolute headcount measures if present
    prefer = ["PEOPLE","NUMBER","MLN","MLN_PER","THND","THND_PER"]
    pick = None
    if "MEASURE" in df.columns:
        for m in prefer:
            tmp = df[df["MEASURE"] == m]
            if not tmp.empty:
                pick = tmp
                break
    if pick is None:
        pick = df
    out = pick.rename(columns={"LOCATION":"country","TIME":"year","Value":"population"})[["country","year","population"]]
    # coerce dtypes here (== amendment you asked for)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype("int64")
    out["country"] = out["country"].astype(str).str.upper()
    return out

def fetch_worldbank_population(iso3: str) -> pd.DataFrame:
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/SP.POP.TOTL?date={START_YEAR}:2050&format=json"
    r = requests.get(url, timeout=30); r.raise_for_status()
    js = r.json()
    data = js[1] or []
    rows = [{"country": iso3, "year": int(d["date"]), "population": d["value"]}
            for d in data if d and d.get("value") is not None]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["country"] = out["country"].astype(str).str.upper()
    return out

def build_population_panel(countries: List[str]) -> pd.DataFrame:
    frames = []
    for c in countries:
        try:
            frames.append(fetch_oecd_population(c))
        except Exception:
            try:
                frames.append(fetch_worldbank_population(c))
            except Exception as e2:
                frames.append(pd.DataFrame([{"country": c, "year": None, "population": None, "error": f"population fetch failed: {e2}"}]))
    pop = pd.concat(frames, ignore_index=True)
    # final dtype normalization (== amendment)
    if not pop.empty:
        pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
        pop = pop.dropna(subset=["year"]).copy()
        pop["year"] = pop["year"].astype("int64")
        pop["country"] = pop["country"].astype(str).str.upper()
    return pop

def coerce_year(df: pd.DataFrame, col: str = "year") -> pd.DataFrame:
    """Ensure 'year' is int64 to avoid object<int merge errors."""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col]).copy()
        df[col] = df[col].astype("int64")
    return df

# -------- Run --------
income = fetch_income_panel(COUNTRIES)
# dtype normalization for income (== amendment)
if not income.empty:
    income["country"] = income["country"].astype(str).str.upper()
    income = coerce_year(income, "year")

pop = build_population_panel(COUNTRIES)

# Merge (types now aligned) + compute per-capita
df = (income.merge(pop, on=["country","year"], how="inner")
             .dropna(subset=["income_real_households","population"])
             .sort_values(["country","year"])
             .reset_index(drop=True))

df["income_real_pc"] = df["income_real_households"] / df["population"]

# 2000=100 index per country (avoids multi-column .apply issue)
def base_2000_series(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("year")
    base_row = g.loc[g["year"] == 2000, "income_real_pc"]
    base = base_row.iloc[0] if not base_row.empty else g["income_real_pc"].iloc[0]
    return pd.Series(base, index=g.index, name="base_2000")

df["base_2000"] = df.groupby("country", group_keys=False).apply(base_2000_series)
df["income_real_pc_idx2000"] = (df["income_real_pc"] / df["base_2000"]) * 100.0
df = df.drop(columns=["base_2000"])

# Save
df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
print(df.head(12).to_string(index=False))
