import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


# -----------------------------
# Helpers
# -----------------------------

def to_number(s):
    """Coerce Google Sheets-exported 'Number'-formatted cells to float.
       Handles thousands separators, spaces, percents, non-breaking spaces.
    """
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    s = str(s).strip()
    if s == "":
        return np.nan
    # Remove commas and spaces used as thousands separators
    s = s.replace(",", "").replace(" ", "")
    # Percent values like '12.3%'
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        import re
        s2 = re.sub(r"[^\d\.\-eE]", "", s)
        try:
            return float(s2)
        except ValueError:
            return np.nan

def compute_bivariate_stats(x: pd.Series, y: pd.Series):
    mask = (~x.isna()) & (~y.isna())
    x1, y1 = x[mask], y[mask]
    n = len(x1)
    if n < 2:
        return n, None
    pearson_r, pearson_p = stats.pearsonr(x1, y1)
    spearman_rho, spearman_p = stats.spearmanr(x1, y1)
    lr = stats.linregress(x1, y1)
    ols = {
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r_value": lr.rvalue,
        "p_value": lr.pvalue,
        "stderr": lr.stderr,
        "r2": lr.rvalue**2
    }
    return n, {
        "pearson": (pearson_r, pearson_p),
        "spearman": (spearman_rho, spearman_p),
        "ols": ols
    }

def format_bivar_box(n, statdict, note=None):
    pr, pp = statdict["pearson"]
    sr, sp = statdict["spearman"]
    ols = statdict["ols"]
    lines = [
        f"n = {n}",
        f"Pearson r = {pr:.2f} (p={pp:.3f})",
        f"Spearman ρ = {sr:.2f} (p={sp:.3f})",
        f"OLS slope = {ols['slope']:.3f}",
        f"R² = {ols['r2']:.3f}, p(slope)={ols['p_value']:.3f}",
    ]
    if note:
        lines.append(note)
    return "\n".join(lines)

def scatter_with_stats(df_plot, x_col, y_col, outpath, title, xlabel=None, ylabel=None, stats_text=None, draw_fit=True, reflect_stats_box=False):
    x = df_plot[x_col]
    y = df_plot[y_col]
    mask = (~x.isna()) & (~y.isna())
    dfp = df_plot.loc[mask].copy()

    plt.figure(figsize=(8,6), dpi=160)
    plt.scatter(dfp[x_col], dfp[y_col])
    # point labels
    for _, row in dfp.iterrows():
        plt.annotate(str(row['Country']), (row[x_col], row[y_col]), xytext=(5, 3), textcoords='offset points', fontsize=8)

    # fit line
    if draw_fit and len(dfp) >= 2:
        lr = stats.linregress(dfp[x_col], dfp[y_col])
        xs = np.linspace(dfp[x_col].min(), dfp[x_col].max(), 100)
        ys = lr.intercept + lr.slope * xs
        plt.plot(xs, ys, linewidth=1.5)

    plt.title(title)
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)

    # stats box
    if stats_text is None:
        n, statdict = compute_bivariate_stats(dfp[x_col], dfp[y_col])
        stats_text = format_bivar_box(n, statdict)

    stats_box_x_coord = 0.98
    stats_box_ha = "right"
    if reflect_stats_box:
        stats_box_x_coord = 0.02
        stats_box_ha = "left"
    plt.gca().text(
        stats_box_x_coord, 0.98, stats_text,
        transform=plt.gca().transAxes,
        ha=stats_box_ha, va="top",
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def get_bivar(df_subset, xcol, ycol, label):
    n_b, statdict = compute_bivariate_stats(df_subset[xcol], df_subset[ycol])
    if statdict is None:
        return (label, {"n": n_b})
    pr, pp = statdict["pearson"]
    sr, sp = statdict["spearman"]
    ols = statdict["ols"]
    return (label, {
        "n": n_b,
        "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp,
        "ols_slope": ols["slope"], "ols_p": ols["p_value"], "r2": ols["r2"]
    })


def compute_multivar_stats(results, y, yhat):
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    statdict = {
        "n": int(results.nobs),
        "k": int(results.df_model),  # number of predictors (excl. constant)
        "r2": float(results.rsquared),
        "adj_r2": float(results.rsquared_adj),
        "rmse": rmse,
        "f_p": float(results.f_pvalue) if results.f_pvalue is not None else np.nan,
        "aic": float(results.aic),
        "bic": float(results.bic),
    }
    return statdict

def format_multivar_box(statdict, results, x_cols_short, max_terms=5):
    # Header stats
    lines = [
        f"n = {statdict['n']:,}, k = {statdict['k']}",
        f"R² = {statdict['r2']:.3f}, adj. R² = {statdict['adj_r2']:.3f}",
        f"RMSE = {statdict['rmse']:.3f}",
    ]
    if np.isfinite(statdict["f_p"]):
        lines.append(f"F p-value = {statdict['f_p']:.3g}")
    # Strongest terms by |t| (exclude constant)
    summ = results.summary2().tables[1]  # coef table as DataFrame
    if "const" in summ.index:
        summ = summ.drop(index="const", errors="ignore")
    if len(summ) > 0:
        summ = summ.assign(abs_t=np.abs(summ["t"]))
        top = summ.sort_values("abs_t", ascending=False).head(max_terms)
        lines.append("Predictors:")
        i = 0
        for name, row in top.iterrows():
            if x_cols_short is not None:
                name = x_cols_short[i]
                i += 1
            beta = row["Coef."]
            pval = row["P>|t|"]
            lines.append(f"  {name}: β={beta:.3g} (p={pval:.3g})")
    return "\n".join(lines)

def scatter_multireg_with_stats(
    df,
    x_cols,
    y_col,
    outpath,
    title,
    xlabel=None,               # X-axis label (defaults to "Actual {y_col}")
    ylabel=None,               # Y-axis label (defaults to "Predicted {y_col}")
    add_constant=True,         # add intercept
    robust=None,               # None, or one of "HC0","HC1","HC2","HC3" for robust SEs
    reflect_stats_box=False,   # mirror the box to the left/top like your function
    custom_stats_text=None,    # override the stats box text if provided
    point_labels_col="Country", # optional column to use for point labels,
    x_cols_short=None       # optional short names for x_cols, in same order as x_cols
):
    # Drop rows with any NA across required cols
    cols_needed = [y_col] + list(x_cols)
    dfp = df.loc[df[cols_needed].notna().all(axis=1)].copy()
    if dfp.empty or len(dfp) < 2:
        raise ValueError("Not enough non-missing rows to fit the model.")

    y = dfp[y_col].astype(float)
    X = dfp[x_cols].astype(float)
    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    # Fit model
    if robust:
        results = sm.OLS(y, X).fit(cov_type=robust)
    else:
        results = sm.OLS(y, X).fit()

    # Predictions
    yhat = results.fittedvalues

    # Plot: Actual vs Predicted with 45° line
    plt.figure(figsize=(8, 6), dpi=160)
    plt.scatter(y, yhat)

    # Point labels (if available)
    if point_labels_col in dfp.columns:
        for _, row in dfp.iterrows():
            plt.annotate(
                str(row[point_labels_col]),
                (row[y_col], results.fittedvalues.loc[row.name]),
                xytext=(5, 3), textcoords="offset points", fontsize=8
            )

    # 45° line
    xy_min = float(min(y.min(), yhat.min()))
    xy_max = float(max(y.max(), yhat.max()))
    xs = np.linspace(xy_min, xy_max, 200)
    plt.plot(xs, xs, linewidth=1.5)

    plt.title(title)
    plt.xlabel(xlabel or f"Actual {y_col}")
    plt.ylabel(ylabel or f"Predicted {y_col}")

    # Stats box
    if custom_stats_text is None:
        statdict = compute_multivar_stats(results, y, yhat)
        stats_text = format_multivar_box(statdict, results, x_cols_short)
    else:
        stats_text = custom_stats_text

    stats_box_x = 0.98
    stats_box_ha = "right"
    if reflect_stats_box:
        stats_box_x = 0.02
        stats_box_ha = "left"

    plt.gca().text(
        stats_box_x, 0.98, stats_text,
        transform=plt.gca().transAxes,
        ha=stats_box_ha, va="top",
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

    return results  # so you can inspect .params, .pvalues, etc.


# Column names (constants)
COL_DENS_INIT = 'Average Major City Initial Urban Density'
COL_WEIGHTED_DENS_INIT = 'Average Major City Initial Urban Density (Weighted)'
COL_PPGMR    = 'Property to Population Growth Multiple Ratio'
COL_AVG_20C_PPB = 'Average Early 20th Century Persons per Building'
COL_DW_CHG   = '% Change in Dwellings per 1000'
COL_INIT_DW  = 'Initial Dwellings per 1,000 People'
COL_WAGE_GROW = 'Real Wage Growth %'
COL_EMPLOYMENT_GROW = 'Employment Growth %'
# 2010 only
COL_INIT_DW_POP_RATIO = 'Initial Dwellings to People Ratio'


def get_pop_growth_col_name(start_year):
    return f'{start_year}-2024 National Population Growth %'

def get_property_growth_col_name(start_year):
    return f'{start_year}-2025 National Real Residential Property Price Growth %'

def get_flats_percent_col_name(start_year):
    return f'Flats as % of Residential Stock c. {start_year}'

def compute_stats_from_2010(df, outdir):
    COL_POP_GROW = get_pop_growth_col_name('2010')
    COL_PROP_GROW = get_property_growth_col_name('2010')

    # --- Population growth compared to Property Price Growth, all countries ---
    scatter_with_stats(
        df, COL_POP_GROW, COL_PROP_GROW,
        outdir / "pop_growth_factor.png",
        title="Population Growth versus Real Property Growth (post-2010)",
    )

      # --- Population + Real wage growth compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df, [COL_POP_GROW, COL_WAGE_GROW], COL_PROP_GROW,
        outdir / "pop_wage_factor_multireg.png",
        title="Population + Real Wage Growth versus Real Property Growth",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010"]
    )

    # --- Population + Real wage growth compared to Property Price Growth, excl. Portugal ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Portugal"], [COL_POP_GROW, COL_WAGE_GROW], COL_PROP_GROW,
        outdir / "pop_wage_factor_multireg_excl_pt.png",
        title="Population + Real Wage Growth versus Real Property Growth (post-2010, excl. PT)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010"]
    )

    # --- Change in dwellings % compared to Property Price Growth, excl. Portugal ---
    scatter_with_stats(
        df, COL_DW_CHG, COL_PROP_GROW,
        outdir / "chg_dwellings.png",
        title="Change in Dwellings % versus Real Property Growth (post-2010)",
        # reflect_stats_box=True,
    )

    # --- Starting dwellings compared to Property Price Growth ---
    scatter_with_stats(
        df, COL_INIT_DW, COL_PROP_GROW,
        outdir / "start_dwellings_factor.png",
        title="Initial Dwellings versus Real Property Growth (post-2010)",
        reflect_stats_box=True,
    )

    # --- Dwellings to Pop ratio compared to Property Price Growth (excl. Portugal) ---
    scatter_with_stats(
        df[df['Country'] != "Portugal"], COL_INIT_DW_POP_RATIO, COL_PROP_GROW,
        outdir / "init_dwellings_to_pop_factor_excl_pt.png",
        title="Initial Dwellings to Population Ratio versus Real Property Growth (post-2010)",
    )

    # --- Dwellings to Pop ratio compared to Property Price Growth (excl. Portugal & Finland) ---
    scatter_with_stats(
        df[~df['Country'].isin(['Portugal', 'Finland'])], COL_INIT_DW_POP_RATIO, COL_PROP_GROW,
        outdir / "init_dwellings_to_pop_factor_excl_pt_ft.png",
        title="Initial Dwellings to Population Ratio versus Real Property Growth (post-2010, excl. PT & FT)",
    )


    # --- Real wage growth compared to Property Price Growth ---
    scatter_with_stats(
        df, COL_WAGE_GROW, COL_PROP_GROW,
        outdir / "wage_growth_factor.png",
        title="Real Wage Growth versus Real Property Growth (post-2010)",
        # reflect_stats_box=True,
    )

    # --- Change in dwellings % compared to Property Price Growth, excl. Portugal ---
    scatter_with_stats(
        df[df["Country"] != "Portugal"], COL_DW_CHG, COL_PROP_GROW,
        outdir / "chg_dwellings_excl_pt.png",
        title="Change in Dwellings % versus Real Property Growth (post-2010, excl. PT)",
        # reflect_stats_box=True,
    )

    # --- Starting dwellings compared to Property Price Growth, excl. Portugal ---
    scatter_with_stats(
        df[df["Country"] != "Portugal"], COL_INIT_DW, COL_PROP_GROW,
        outdir / "start_dwellings_factor_excl_pt.png",
        title="Initial Dwellings versus Real Property Growth (post-2010, excl. PT)",
        # reflect_stats_box=True,
    )

    # --- Real wage growth compared to Property Price Growth, excl. Portugal ---
    scatter_with_stats(
        df[df["Country"] != "Portugal"], COL_WAGE_GROW, COL_PROP_GROW,
        outdir / "wage_growth_factor_excl_pt.png",
        title="Real Wage Growth versus Real Property Growth (post-2010, excl. PT)",
        # reflect_stats_box=True,
    )

    # --- Population + Real wage growth + Employment growth compared to Property Price Growth ---
    scatter_multireg_with_stats(
        df, [COL_POP_GROW, COL_WAGE_GROW, COL_EMPLOYMENT_GROW], COL_PROP_GROW,
        outdir / "pop_wage_employment_multireg.png",
        title="Population + Real Wage Growth + Employment Growth versus Real Property Growth (post-2010)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010", "Employment Growth % since 2010"]
    )

    # --- Population + Real wage growth + Employment growth compared to Property Price Growth, excl. Portugal ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Portugal"], [COL_POP_GROW, COL_WAGE_GROW, COL_EMPLOYMENT_GROW], COL_PROP_GROW,
        outdir / "pop_wage_employment_multireg_excl_pt.png",
        title="Population + Real Wage Growth + Employment Growth versus Real Property Growth (post-2010, excl. PT)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010", "Employment Growth % since 2010"]
    )

    # --- Population + Real wage growth + Chg in Dwellings compared to Property Price Growth, excl. Portugal ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Portugal"], [COL_POP_GROW, COL_WAGE_GROW, COL_DW_CHG], COL_PROP_GROW,
        outdir / "pop_wage_chg_dwellings_multireg_excl_pt.png",
        title="Population + Real Wage Growth + Change in Dwellings versus Real Property Growth (post-2010, excl. PT)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010", "Change in Dwellings % since 2010"]
    )

    # --- Population + Real wage growth + Employment growth + Chg in Dwellings compared to Property Price Growth, excl. Portugal ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Portugal"], [COL_POP_GROW, COL_WAGE_GROW, COL_EMPLOYMENT_GROW, COL_DW_CHG], COL_PROP_GROW,
        outdir / "pop_wage_employment_chg_dwellings_multireg_excl_pt.png",
        title="Population + Real Wages + Employment + Dwellings versus Real Property Growth (post-2010, excl. PT)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010", "Employment Growth % since 2010", "Change in Dwellings % since 2010"]
    )


    # --- Population + Real wage growth + Chg in Dwellings + Init Dwellings compared to Property Price Growth, excl. Portugal ---
    scatter_multireg_with_stats( 
        df[df["Country"] != "Portugal"], [COL_POP_GROW, COL_WAGE_GROW, COL_DW_CHG], COL_PROP_GROW,
        outdir / "pop_wage_chg_dwellings_multireg_excl_pt.png",
        title="Population + Real Wage Growth + Change in Dwellings versus Real Property Growth (post-2010, excl. PT)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2010", "Real Wage Growth % since 2010", "Change in Dwellings % since 2010"]
    )

    # --- Save quick summary JSON + full OLS summaries ---
    summary = {}
    # TODO: fix this
    summary.update([
        get_bivar(df, COL_INIT_DW, COL_PPGMR, "Pred 1a (excl KR)"),
    ])
    # TODO: fix this
    # with open(outdir / "model_summaries.txt", "w") as fh:
        # TODO: fix this
        # fh.write("=== Prediction 2b OLS ===\n")
        # fh.write(str(model2b.summary()) + "\n\n")
        # fh.write("=== Prediction 3a OLS ===\n")
        # fh.write(str(model3a.summary()) + "\n\n")
        # fh.write("=== Prediction 3b OLS ===\n")
        # fh.write(str(model3b.summary()) + "\n\n")
    
    return summary

def compute_stats_from_2000(df, outdir):
    COL_POP_GROW = get_pop_growth_col_name('2000')
    COL_PROP_GROW = get_property_growth_col_name('2000')
    COL_FLATS_PC = get_flats_percent_col_name("2000")

    # --- Population growth compared to Property Price Growth, all countries ---
    scatter_with_stats(
        df, COL_POP_GROW, COL_PROP_GROW,
        outdir / "pop_growth_factor.png",
        title="Population Growth versus Real Property Growth",
        reflect_stats_box=True
    )

    # --- Real wage growth compared to Property Price Growth, all countries ---
    scatter_with_stats(
        df, COL_WAGE_GROW, COL_PROP_GROW,
        outdir / "real_wage_factor.png",
        title="Real Wage Growth versus Real Property Growth",
        reflect_stats_box=True
    )


    # --- Employment growth compared to Property Price Growth, all countries ---
    scatter_with_stats(
        df, COL_EMPLOYMENT_GROW, COL_PROP_GROW,
        outdir / "employment_factor.png",
        title="Employment Growth versus Real Property Growth",
        reflect_stats_box=True
    )

    # --- Population + Real wage growth compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df, [COL_POP_GROW, COL_WAGE_GROW], COL_PROP_GROW,
        outdir / "pop_wage_factor_multireg.png",
        title="Population + Real Wage Growth versus Real Property Growth",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2000", "Real Wage Growth % since 2000"]
    )

    # --- Population + Real wage growth compared to Property Price Growth, excl. Ireland ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Ireland"], [COL_POP_GROW, COL_WAGE_GROW], COL_PROP_GROW,
        outdir / "pop_wage_factor_multireg_excl_ir.png",
        title="Population + Real Wage Growth versus Real Property Growth (excl. IR)",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2000", "Real Wage Growth % since 2000"]
    )

    # --- Population + Employment + Real wage growth compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df, [COL_POP_GROW, COL_EMPLOYMENT_GROW, COL_WAGE_GROW], COL_PROP_GROW,
        outdir / "pop_employment_wage_factor_multireg.png",
        title="Population + Employment Growth + Real Wage Growth versus Real Property Growth",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2000", "Employment Growth % since 2000", "Real Wage Growth % since 2000"]
    )

        # --- Population + Employment + Real wage growth compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Ireland"], [COL_POP_GROW, COL_EMPLOYMENT_GROW, COL_WAGE_GROW], COL_PROP_GROW,
        outdir / "pop_employment_wage_factor_multireg_excl_ir.png",
        title="Population + Employment Growth + Real Wage Growth versus Real Property Growth",
        reflect_stats_box=True,
        x_cols_short=["Pop Growth % since 2000", "Employment Growth % since 2000", "Real Wage Growth % since 2000"]
    )

    # --- Population + Real wage growth + % Change Dwellings compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df, [COL_POP_GROW, COL_WAGE_GROW, COL_DW_CHG], COL_PROP_GROW,
        outdir / "pop_wage_dwellings_factor_multireg.png",
        title="Population + Real Wage Growth + Change Dwellings versus Real Property Growth",
        reflect_stats_box=True
    )

        # --- Population + Real wage growth + Initial Dwellings + % Change Dwellings compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df, [COL_POP_GROW, COL_WAGE_GROW, COL_INIT_DW, COL_DW_CHG], COL_PROP_GROW,
        outdir / "pop_wage_init_dwellings_chg_dwellings_factor_multireg.png",
        title="Population + Real Wage Growth + Initial Dwellings + Change Dwellings versus Real Property Growth",
        reflect_stats_box=True
    )

            # --- Population + Real wage growth + Initial Dwellings + % Change Dwellings compared to Property Price Growth, all countries ---
    scatter_multireg_with_stats(
        df[df["Country"] != "Ireland"], [COL_POP_GROW, COL_WAGE_GROW, COL_INIT_DW, COL_DW_CHG], COL_PROP_GROW,
        outdir / "pop_wage_init_dwellings_chg_dwellings_factor_multireg_excl_ir.png",
        title="Population + Real Wage Growth + Initial Dwellings + Change Dwellings versus Real Property Growth",
        reflect_stats_box=True
    )


    # --- Real wage growth compared to Property Price Growth, excl. Korea ---
    df_excl_kr = df[df['Country'] != 'Korea']
    scatter_with_stats(
        df_excl_kr, COL_WAGE_GROW, COL_PROP_GROW,
        outdir / "real_wage_factor_excl_kr.png",
        title="Real Wage Growth versus Real Property Growth [Excl. KR]",
        reflect_stats_box=True
    )

    # --- Prediction 1a (all): Initial urban density compared to PPGMR, all countries ---
    scatter_with_stats(
        df, COL_DENS_INIT, COL_PPGMR,
        outdir / "pred1a_all.png",
        title="Prediction 1 [Demographia Density, all countries]"
    )

    # --- Prediction 1c (excl. NZ, IR): % Flats compared to PPGMR ---
    # Exclude NZ for reason of lack of data and Ireland as outlier
    df_excl_nz_ir = df[~df['Country'].isin(['New Zealand', 'Ireland'])]
    scatter_with_stats(
        df_excl_nz_ir, COL_FLATS_PC, COL_PPGMR,
        outdir / "pred1c_excl_ir.png",
        title="Prediction 1 [Flats data, excl. NZ, IR]"
    )

    # --- Save quick summary JSON + full OLS summaries ---
    summary = {}
    # TODO: restore this part after I regain Copilot access
    summary.update([
    ])
    return summary

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(csv_path)

    # Coerce numeric columns
    num_cols = [
        COL_DENS_INIT, 'Average Major City Urban Density',
        '2000-2025 National Real Residential Property Price Growth %',
        '2010-2025 National Real Residential Property Price Growth %',
        '2000-2024 National Population Growth %',
        '2010-2024 National Population Growth %',
        'Flats as % of Residential Stock c. 2000',
        'Flats as % of Residential Stock c. 2010',
        COL_WAGE_GROW,
        COL_PPGMR,
        COL_INIT_DW, 'Later Dwellings per 1,000 People', COL_DW_CHG,
        'City 1 Initial Urban Area Density', 'City 2 Initial Urban Area Density',
        'City 1 Current Urban Density', 'City 2 Current Urban Density', COL_AVG_20C_PPB
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_number)

    if args.csv == "data_2010-2025.csv":
        summary = compute_stats_from_2010(df, outdir)
    else:
        summary = compute_stats_from_2000(df, outdir)

    (outdir / "summary_stats.json").write_text(json.dumps(summary, indent=2))

    print("Done. Figures and summaries saved to:", str(outdir.resolve()))


if __name__ == "__main__":
    main()
