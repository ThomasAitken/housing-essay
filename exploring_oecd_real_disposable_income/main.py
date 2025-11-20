
import argparse
import sys
import pandas as pd
from pathlib import Path

def compute_growth(input_csv: str, start_year: int, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = ['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV.")

    df = df[required].copy()
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['OBS_VALUE', 'TIME_PERIOD', 'REF_AREA'])

    def to_period_q(s: str):
        try:
            return pd.Period(s, freq='Q')
        except Exception:
            return pd.NaT
    df['PERIOD'] = df['TIME_PERIOD'].astype(str).map(to_period_q)
    df = df.dropna(subset=['PERIOD'])

    start_period = pd.Period(f"{start_year}Q1", freq='Q')
    cap_period = pd.Period("2025Q1", freq='Q')  # use this if present; otherwise per-country latest

    # Build start values (exact Q1 of start_year)
    df_keyed = df.set_index(['REF_AREA', 'PERIOD']).sort_index()

    # START: require exact Q1 of start_year
    try:
        start_vals = (df_keyed['OBS_VALUE']
                      .xs(start_period, level='PERIOD', drop_level=False)
                      .reset_index()
                      .rename(columns={'OBS_VALUE': 'START_VALUE'})[['REF_AREA','PERIOD','START_VALUE']])
        start_vals = start_vals.rename(columns={'PERIOD':'START_PERIOD'})
    except KeyError:
        # No country has that start period at all; create empty frame to still continue
        start_vals = pd.DataFrame(columns=['REF_AREA','START_PERIOD','START_VALUE'])

    # END: per country, choose 2025Q1 if available, else latest available period for that country
    # Compute each country's latest period <= cap_period; if none, then absolute latest
    def choose_end_period(g: pd.DataFrame) -> pd.Period:
        periods = g.index.get_level_values('PERIOD')
        # Prefer <= cap_period
        le_cap = [p for p in periods if p <= cap_period]
        if le_cap:
            return max(le_cap)
        # otherwise, just the latest available
        return max(periods)

    end_rows = []
    for (country, sub) in df_keyed.groupby(level='REF_AREA'):
        end_p = choose_end_period(sub)
        end_val = sub.loc[(country, end_p), 'OBS_VALUE']
        end_rows.append({'REF_AREA': country, 'END_PERIOD': end_p, 'END_VALUE': end_val})
    end_vals = pd.DataFrame(end_rows)

    # Merge and compute growth
    out = pd.merge(start_vals, end_vals, on='REF_AREA', how='outer')

    out['GROWTH_PCT'] = ((out['END_VALUE'] - out['START_VALUE']) / out['START_VALUE']) * 100.0

    # Status column
    def status_row(r):
        s_missing = pd.isna(r['START_VALUE'])
        e_missing = pd.isna(r['END_VALUE'])
        if not s_missing and not e_missing:
            # Flag whether END is the capped 2025Q1 or a fallback
            return 'ok (end=2025Q1)' if r.get('END_PERIOD', pd.NaT) == cap_period else 'ok (end=fallback)'
        if s_missing and not e_missing:
            return 'missing start'
        if not s_missing and e_missing:
            return 'missing end'
        return 'missing both'

    out['STATUS'] = out.apply(status_row, axis=1)

    # Order and save
    out = out[['REF_AREA','START_PERIOD','START_VALUE','END_PERIOD','END_VALUE','GROWTH_PCT','STATUS']]\
            .sort_values('REF_AREA').reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


INPUT_FILE = "real_disposable_income_per_capita.csv"
def main():
    parser = argparse.ArgumentParser(description="Compute growth % from start-year Q1 to 2025 Q1 (or latest available) per country.")
    parser.add_argument("--start-year", required=True, type=int, help="Start year (e.g., 2010). Uses Q1 of that year.")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()
    try:
        df = compute_growth(INPUT_FILE, args.start_year, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
