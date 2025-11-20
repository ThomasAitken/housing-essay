"""
OECD Average annual wages (AV_AN_WAGE), constant PPP USD — all OECD, 2000→latest
- Uses the '/all' export, filters by CODED dimensions (robust),
- Applies UNIT_MULT,
- Builds START_YEAR=100 index, computes growth to latest,
- Saves panel + scoreboard CSVs and a PNG.
"""

import io
import requests
import pandas as pd
import matplotlib.pyplot as plt

START_YEAR = 2000


URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ELS.SAE,DSD_EARNINGS@AV_AN_WAGE/"
    f"all?startPeriod={START_YEAR}&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "text/csv",
}

# ---------- Download ----------
resp = requests.get(URL, headers=HEADERS, timeout=5)
resp.raise_for_status()
raw = pd.read_csv(io.StringIO(resp.text))

print("Rows on download:", len(raw))
# raw.to_csv("debug_raw.csv", index=False)  # uncomment if you want to inspect

# Helper to pick a column by code name or fallback to label
def pick(df, code_name, label_name_candidates):
    if code_name in df.columns:
        return code_name
    for cand in ([label_name_candidates] if isinstance(label_name_candidates, str) else label_name_candidates):
        if cand in df.columns:
            return cand
    return None

# ---------- Identify columns (prefer CODE columns; fall back to LABELs) ----------
COLS = {
    "country_code": pick(raw, "REF_AREA", ["Reference area code", "REF_AREA_CODE"]),
    "country": pick(raw, "REF_AREA_LABEL", ["Reference area", "Country"]),
    "freq": pick(raw, "FREQ", ["Frequency"]),
    "price_base": pick(raw, "PRICE_BASE", ["Price base"]),
    "unit_measure": pick(raw, "UNIT_MEASURE", ["Unit of measure", "Unit measure"]),
    "measure": pick(raw, "MEASURE", ["Measure"]),
    "stat": pick(raw, "STAT", ["Statistic"]),
    "time": pick(raw, "TIME_PERIOD", ["Time period"]),
    "value": pick(raw, "OBS_VALUE", ["Observation value", "Value"]),
    "unit_mult": pick(raw, "UNIT_MULT", ["Unit multiplier"]),
}

missing = [k for k, v in COLS.items() if v is None and k in {"country", "time", "value"}]
if missing:
    raise RuntimeError(f"Required columns missing: {missing}. Found columns: {list(raw.columns)}")

df = raw.copy()

# ---------- Filter by codes (robust to label wording) ----------
def keep_code(col, good_codes, also_allow_labels=False):
    if col is None:
        return pd.Series([True] * len(df))  # if column absent, don't filter on it
    s = df[col].astype(str)
    mask = s.isin(good_codes)
    if also_allow_labels:  # optional: also accept human labels that contain a keyword
        low = s.str.lower()
        for kw in ["annual", "constant", "ppp", "wage", "mean"]:
            mask |= low.str.contains(kw, na=False)
    return mask

# Keep: Annual, Constant prices, PPP USD, Wages, Mean
m_freq   = keep_code(COLS["freq"],        {"A"})
m_price  = keep_code(COLS["price_base"],  {"Q"})          # Q = constant prices (per OECD codes)
m_unit   = keep_code(COLS["unit_measure"],{"USD_PPP"})    # PPP USD
m_meas   = keep_code(COLS["measure"],     {"WG"})         # wages
m_stat   = keep_code(COLS["stat"],        {"MEAN"})

mask_all = m_freq & m_price & m_unit & m_meas & m_stat
df = df[mask_all].copy()
print("After dimension filters:", len(df))

# ---------- Apply UNIT_MULT ----------
# OECD unit multipliers are powers of 10 (0=units, 3=thousands, 6=millions, etc.)
val = pd.to_numeric(df[COLS["value"]], errors="coerce")
if COLS["unit_mult"] and COLS["unit_mult"] in df.columns:
    pow10 = pd.to_numeric(df[COLS["unit_mult"]], errors="coerce").fillna(0)
    val = val * (10.0 ** pow10)

df["wage_usd_ppp_const"] = val

# ---------- Keep & clean core fields ----------
df["year"] = pd.to_numeric(df[COLS["time"]], errors="coerce").astype("Int64")
if COLS["country_code"] is None:
    df["country_code"] = df[COLS["country"]]  # fall back to label for grouping
else:
    df["country_code"] = df[COLS["country_code"]]

df["country"] = df[COLS["country"]]
df = df[["country", "country_code", "year", "wage_usd_ppp_const"]]
df = df[df["wage_usd_ppp_const"].notna() & (df["wage_usd_ppp_const"] > 0)]
print("After NA/zero filter:", len(df))

# ---------- Build START_YEAR base and compute growth ----------
base = (
    df[df["year"] == START_YEAR][["country_code", "wage_usd_ppp_const"]]
      .rename(columns={"wage_usd_ppp_const": f"base_{START_YEAR}"})
)
print("Countries with {START_YEAR} base:", base["country_code"].nunique())

panel = df.merge(base, on="country_code", how="inner")
panel = panel[panel[f"base_{START_YEAR}"] > 0]
panel[f"index_{START_YEAR}=100"] = 100.0 * panel["wage_usd_ppp_const"] / panel[f"base_{START_YEAR}"]

latest = (panel.sort_values(["country_code", "year"])
               .groupby("country_code", as_index=False)
               .tail(1))
latest[f"growth_since_{START_YEAR}_pct"] = 100.0 * (latest["wage_usd_ppp_const"]/latest[f"base_{START_YEAR}"] - 1.0)

# ---------- Save ----------
panel_out = panel[["country", "country_code", "year", "wage_usd_ppp_const", f"index_{START_YEAR}=100"]] \
    .sort_values(["country", "year"])
panel_out.to_csv(f"oecd_real_wages_panel_{START_YEAR}_latest.csv", index=False)

growth_out = latest[["country", "country_code", "year", f"growth_since_{START_YEAR}_pct", f"index_{START_YEAR}=100"]] \
    .sort_values(f"growth_since_{START_YEAR}_pct", ascending=False)
growth_out.to_csv(f"oecd_real_wages_growth_since_{START_YEAR}.csv", index=False)

print("Wrote CSVs:")
print(f" - oecd_real_wages_panel_{START_YEAR}_latest.csv  (rows:", len(panel_out), ")")
print(f" - oecd_real_wages_growth_since_{START_YEAR}.csv  (rows:", len(growth_out), ")")

# ---------- Plot ----------
plt.figure(figsize=(10, 12), dpi=160)
plt.barh(growth_out["country"], growth_out[f"growth_since_{START_YEAR}_pct"])
plt.xlabel(f"Real wage growth since {START_YEAR} (%)")
plt.ylabel("Country")
plt.title(f"Real Wages Growth Since {START_YEAR} (OECD, constant PPP USD)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"oecd_wages_growth_{START_YEAR}.png", bbox_inches="tight")
plt.close()
print(f"Wrote: oecd_wages_growth_{START_YEAR}.png")
