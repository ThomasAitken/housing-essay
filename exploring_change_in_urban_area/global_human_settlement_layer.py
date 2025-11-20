import pandas as pd, numpy as np
from pathlib import Path

MTUC_CSV   = Path("ghs_mtuc.csv")      # area by year
FIXED_XLSX  = Path("ghs_fixed_boundary.xlsx")           # population by year

CITIES = [
    ("Stockholm", "SWE"),
    ("Helsinki",  "FIN"),
    ("Sydney",    "AUS"),
    ("Melbourne", "AUS"),
    ("Auckland",  "NZL"),
    ("Wellington","NZL"),
    ("Toronto",   "CAN"),
    ("Vancouver", "CAN"),
    ("Montreal",  "CAN"),
    ("Paris",     "FRA"),
    ("Barcelona", "ESP"),
]

# ---------- helpers ----------
def read_any_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig","utf-8","cp1252","latin1"):
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, engine="python", sep=";", encoding="latin1")

def coerce_area(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace("\u00a0","", regex=False)
    both = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    s = s.mask(both, s.str.replace(".","", regex=False).str.replace(",",".", regex=False))
    only_comma = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s = s.mask(only_comma, s.str.replace(",",".", regex=False))
    s = s.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def coerce_pop(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pick_row(df, name_col, city):
    s = df[name_col].astype(str)
    q = s.str.strip().str.casefold().eq(city.casefold())
    hit = df.loc[q]
    if hit.empty:
        hit = df.loc[s.str.contains(city, case=False, na=False)]
    if hit.empty:
        raise RuntimeError(f"Not found: {city} in {name_col}")
    return hit.iloc[0]

COUNTRY_NAME_COL = "GC_UCN_MAI_2025"

# ---------- load MTUC (AREA) ----------
mt = read_any_csv(MTUC_CSV)
MT_A2000_COL = "MT_UCA_KM2_2000"
MT_A2025_COL = "MT_UCA_KM2_2025"
mt[MT_A2000_COL] = coerce_area(mt[MT_A2000_COL])
mt[MT_A2025_COL] = coerce_area(mt[MT_A2025_COL])

# ---------- load FIXED (POP) from XLSX ----------
xf = pd.ExcelFile(FIXED_XLSX)
sheet_name = "GHSL"
fx_df = xf.parse(sheet_name, dtype=str)

POP_2000_COL = "GH_POP_TOT_2000"
POP_2025_COL = "GH_POP_TOT_2025"

# coerce POP
fx_df[POP_2000_COL] = coerce_pop(fx_df[POP_2000_COL])
fx_df[POP_2025_COL] = coerce_pop(fx_df[POP_2025_COL])

# ---------- compute ----------
rows = []
for city, iso in CITIES:
    m = pick_row(mt, COUNTRY_NAME_COL, city)
    f = pick_row(fx_df, COUNTRY_NAME_COL, city)

    a0, a1 = float(m[MT_A2000_COL]), float(m[MT_A2025_COL])       # km²
    p0, p1 = (round(float(f[POP_2000_COL])), round(float(f[POP_2025_COL])))                   # persons (ints/NaN)

    dA = a1 - a0
    dP = (p1 - p0) if pd.notna(p1) and pd.notna(p0) else np.nan
    m2_per_person = (dA / dP * 1_000_000) if pd.notna(dP) and dP != 0 else np.nan

    density_2000 = (p0 / a0) if pd.notna(p0) and a0 > 0 else np.nan
    density_2025 = (p1 / a1) if pd.notna(p1) and a1 > 0 else np.nan
    density_chg_pct = None
    if pd.notna(density_2000) and pd.notna(density_2025) and density_2000 > 0:
        density_chg_pct = (density_2025 - density_2000) / density_2000 * 100
        # Compute "expansion-adjusted change in density (%)" according to the following assumptions:
        # - Assume that population density in expanded area matches the initial density in 2000
        # - Then find the density change in the original area
        if pd.notna(dA) and dA > 0:
            p_in_expanded_area = dA * density_2000
            p_in_original_area_2025 = p1 - p_in_expanded_area
            if p_in_original_area_2025 > 0 and a0 > 0:
                adj_density_2025 = p_in_original_area_2025 / a0
                adj_density_chg_pct = (adj_density_2025 - density_2000) / density_2000 * 100

    rows.append({
        "City": city, "ISO": iso,
        "Area_2000_km²": round(a0,2), "Area_2025_km²": round(a1,2), "ΔArea_km²": round(dA,2),
        "Pop_2000": None if pd.isna(p0) else int(p0),
        "Pop_2025": None if pd.isna(p1) else int(p1),
        "ΔPop": None if pd.isna(dP) else int(dP),
        "Expansion per added person (m²/person)": None if pd.isna(m2_per_person) else round(m2_per_person,1),
        "Change in Density (%)": None if pd.isna(density_chg_pct) else round(density_chg_pct,1),
        "Expansion-adjusted Change in Density (%)": None if pd.isna(adj_density_chg_pct) else round(adj_density_chg_pct,1),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
