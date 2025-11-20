"""
Portugal: new-dwelling completions vs implied demand from population growth (2018–2024)

- Blue bars: dwellings completed (new family housing)
- Black line: implied dwellings from population change, using persons_per_dwelling (default 2.4)

You can edit the DATA section below to adjust values or extend to new years.
Optionally, you may pass --ppd to change persons-per-dwelling.

Outputs:
  - portugal_same_scale_dwellings_vs_implied_2018_2024.png
  - portugal_pop_vs_completions_2018_2024.csv
"""

import argparse
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# DATA (edit as needed)
# -----------------------
# INE resident population at 31 Dec (people)
POP_RESIDENT = {
    2018: 10_276_617,
    2019: 10_295_909,
    2020: 10_298_252,
    2021: 10_421_117,
    2022: 10_516_621,
    2023: 10_639_726,
    2024: 10_749_635,
}

# INE "fogos concluídos em construções novas para habitação familiar" (units)
# Note: 2024 is preliminary/placeholder; replace with final when published.
DWELLINGS_COMPLETED = {
    2018: round(18_181 / 1.232),  # ≈ 14,760 (2019 was +23.2% vs 2018)
    2019: 18_181,
    2020: 19_900,                 # rounded press-note figure for this cut
    2021: 23_522,
    2022: 23_489,
    2023: 23_652,
    2024: 24_650,                 # preliminary placeholder (“mais de 24 mil”)
}

# (Optional) Persons per dwelling by year if you want to vary it. If set, it overrides --ppd for matching years.
# Leave empty to use a single ppd for all years.
PPD_BY_YEAR = {
    # e.g., 2023: 2.40, 2024: 2.39
}

# -----------------------
# Script
# -----------------------

def build_dataframe(ppd_default: float) -> pd.DataFrame:
    years = sorted(set(POP_RESIDENT) & set(DWELLINGS_COMPLETED))
    df = pd.DataFrame({
        "year": years,
        "population_resident": [POP_RESIDENT[y] for y in years],
        "dwellings_completed": [DWELLINGS_COMPLETED[y] for y in years],
    })
    # Annual population change vs prior year (NaN for first year)
    df["pop_change"] = df["population_resident"].diff()

    # Persons per dwelling to use each year
    df["ppd"] = [PPD_BY_YEAR.get(y, ppd_default) for y in df["year"]]

    # Convert population change (people) -> implied dwellings (units) on the same y-axis
    # First year will be NaN because pop_change is NaN
    df["implied_dwellings_from_pop_change"] = (df["pop_change"] / df["ppd"]).round()

    # Helpful derived metric (not plotted, but useful to inspect)
    df["completions_per_1000_pop"] = 1000 * df["dwellings_completed"] / df["population_resident"]

    return df

def plot_same_scale(df: pd.DataFrame, out_png: Path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    mask = df["year"] >= 2019

    # Blue bars: actual completions
    ax.bar(df.loc[mask, "year"], df.loc[mask, "dwellings_completed"], label="Dwellings completed (new family housing)", alpha=0.85)

    # Black line: implied dwellings from population change / persons per dwelling
    ax.plot(df.loc[mask, "year"], df.loc[mask, "implied_dwellings_from_pop_change"], marker="o", linewidth=2, color="black",
            label="Implied dwellings from population change")

    ax.set_xlabel("Year")
    ax.set_ylabel("Dwellings (units)")
    title = "Portugal: New-dwelling completions vs implied demand from population growth (2019–2024)"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppd", type=float, default=2.4,
                        help="Persons per dwelling (default: 2.4). Ignored for years present in PPD_BY_YEAR.")
    parser.add_argument("--out-png", default="portugal_same_scale_dwellings_vs_implied_2018_2024.png",
                        help="Output PNG path")
    parser.add_argument("--out-csv", default="portugal_pop_vs_completions_2018_2024.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    df = build_dataframe(ppd_default=args.ppd)

    # Save CSV
    out_csv = Path(args.out_csv)
    df.to_csv(out_csv, index=False)

    # Plot
    out_png = Path(args.out_png)
    title_suffix = f"ppd={args.ppd}" if not PPD_BY_YEAR else "year-specific ppd"
    plot_same_scale(df, out_png, title_suffix=title_suffix)

    print(f"Wrote: {out_png.resolve()}")
    print(f"Wrote: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
