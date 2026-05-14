import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from sklearn.linear_model import LinearRegression

# Load the GT data
GT_PATH = "gt/CoresGT_MNHT.csv"
GT_VAL_PATH = "gt/CoresGT_COL_STL.csv"

df_gt_base = pd.read_csv(GT_PATH)
df_gt_val_base = pd.read_csv(GT_VAL_PATH)


def compute_metrics(gt, pred):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - gt

    pearson_r = np.corrcoef(gt, pred)[0, 1]
    # Spearman correlation = Pearson correlation of ranks (no SciPy dependency).
    gt_rank = pd.Series(gt).rank(method="average").to_numpy()
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    spearman_r = np.corrcoef(gt_rank, pred_rank)[0, 1]
    mse = np.mean((gt - pred) ** 2)
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    mean_error = np.mean(err)
    std_error = np.std(err, ddof=1) if err.size > 1 else np.nan

    return pearson_r, spearman_r, r2, mse, mean_error, std_error


def sanitize_filename_part(value):
    # Replace Windows-forbidden filename characters and normalize whitespace.
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(value))
    sanitized = re.sub(r'\s+', '_', sanitized).strip('._')
    return sanitized or 'feature'


def plot_scatter_calibrated(gt, pred_calib, title, save_path):
    x = np.asarray(gt, dtype=float)
    y = np.asarray(pred_calib, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, alpha=0.7, s=100, label="Data points")
    ax.grid(True)
    ax.set_xlabel("Ground Truth", fontsize=24)
    ax.set_ylabel("Predicted (calibrated)", fontsize=24)
    ax.set_title(title, fontsize=24)

    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, "r--", linewidth=3, label=f"Fit (slope={reg.coef_[0]:.2f})")

    ax.legend(fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot: {save_path}")


def plot_bland_altman(ax, gt, pred, label, color):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    means = (gt + pred) / 2
    diffs = pred - gt
    bias = np.mean(diffs)
    loa = 1.96 * np.std(diffs, ddof=1)
    ax.scatter(means, diffs, alpha=0.6, label=label, color=color)
    return bias, loa


# Load the prediction data
# model_list = ["dynunet", "segresnet", "unet", "unet100k", "unetr"]
model_list = [{"model": "dynunet", "feat_pred": "Root Length Diameter Range 3 (px)"},
              {"model": "segresnet", "feat_pred": "Root Length Diameter Range 2 (px)"},
              {"model": "unet", "feat_pred": "Root Length Diameter Range 2 (px)"},
              {"model": "unetr", "feat_pred": "Root Length Diameter Range 2 (px)"}]

metrics_test_rows = []
metrics_val_rows = []
for model_entry in model_list:
    model_name = model_entry["model"]
    feat_pred = model_entry["feat_pred"]
    df_gt = df_gt_base.copy()
    df_gt_val = df_gt_val_base.copy()
    df_pred = pd.read_csv(f"outputs/prediction_{model_name}_mnht.csv")
    df_pred_val = pd.read_csv(f"outputs/prediction_{model_name}_col_stl.csv")

    print(f"\n--- {model_name} merge diagnostics ---")
    print(f"  df_gt_base columns : {list(df_gt_base.columns)}")
    print(f"  df_gt_val  columns : {list(df_gt_val_base.columns)}")
    print(f"  df_pred    columns : {list(df_pred.columns)}")
    print(f"  df_pred_val columns: {list(df_pred_val.columns)}")
    print(f"  df_gt_base ID sample  : {df_gt_base.iloc[:3, 0].tolist()}")
    print(f"  df_gt_val  ID sample  : {df_gt_val_base.iloc[:3, 0].tolist()}")
    print(f"  df_pred    soil_core sample  : {df_pred['soil_core'].iloc[:3].tolist() if 'soil_core' in df_pred.columns else 'COLUMN MISSING'}")
    print(f"  df_pred_val soil_core sample : {df_pred_val['soil_core'].iloc[:3].tolist() if 'soil_core' in df_pred_val.columns else 'COLUMN MISSING'}")

    # Align GT/pred rows by core ID before metric computations.
    pred_rename = {c: f"pred_{c}" for c in df_pred.columns if c != "soil_core"}
    pred_val_rename = {c: f"pred_{c}" for c in df_pred_val.columns if c != "soil_core"}
    df_pred = df_pred.rename(columns=pred_rename)
    df_pred_val = df_pred_val.rename(columns=pred_val_rename)

    df_m_full = (
        df_gt_base.merge(df_pred, left_on="label ID", right_on="soil_core", how="inner")
        .drop(columns=["soil_core"], errors="ignore")
        .rename(columns={"label ID": "soil_core"})
    )

    # Subset to selected Manhattan cores from filtered GT file.
    df_m = (
        df_gt
        .merge(df_pred, left_on="label ID", right_on="soil_core", how="inner")
        .drop(columns=["soil_core"], errors="ignore")
        .rename(columns={"label ID": "soil_core"})
        .reset_index(drop=True)
    )

    df_s = (
        df_gt_val.merge(df_pred_val, left_on="corename", right_on="soil_core", how="inner")
        .drop(columns=["soil_core"], errors="ignore")
        .rename(columns={"corename": "soil_core"})
    )


    feat_gt = "Length(cm)" # "Length(cm)", "ProjArea(cm2)", "SurfArea(cm2)", "AvgDiam(mm)", "L00", "L10", "L20", "L30", "L40"
    # Sum of 0<.L.<=1.000000  Sum of 1.0000000<.L.<=2.0000000	Sum of 2.0000000<.L.<=3.0000000	Sum of 3.0000000<.L.<=4.0000000	Sum of .L.>4.0000000
    # Use the same normalization convention as SubsetSelection.py:
    # standardize on the full matched split, then evaluate on selected subset.
    # MNHT: normalize using a random subsample of 10 cores from the full split
    mnht_subsample_gt   = df_m_full[feat_gt].sample(n=min(10, len(df_m_full)), random_state=42)
    mnht_subsample_pred = df_m_full[f"pred_{feat_pred}"].sample(n=min(10, len(df_m_full)), random_state=42)
    xmean, xstd = mnht_subsample_gt.mean(),   mnht_subsample_gt.std()
    ymean, ystd = mnht_subsample_pred.mean(), mnht_subsample_pred.std()
    # COL_STL: normalize using all cores
    x_valmean, x_valstd = df_s[feat_gt].mean(),               df_s[feat_gt].std()
    y_valmean, y_valstd = df_s[f"pred_{feat_pred}"].mean(),   df_s[f"pred_{feat_pred}"].std()

    # Exclude calibration cores from evaluation to avoid data leakage
    calibration_cores = set(df_m_full.loc[mnht_subsample_gt.index, "soil_core"])
    df_m_eval = df_m[~df_m["soil_core"].isin(calibration_cores)].reset_index(drop=True)

    x     = (df_m_eval[feat_gt]                  - xmean)     / xstd
    y     = (df_m_eval[f"pred_{feat_pred}"]      - ymean)     / ystd
    x_val = (df_s[feat_gt]                       - x_valmean) / x_valstd
    y_val = (df_s[f"pred_{feat_pred}"]           - y_valmean) / y_valstd

    print(f"  df_m_full rows: {len(df_m_full)}  |  df_m rows: {len(df_m)}  |  df_s rows: {len(df_s)}")
    if df_m_full.empty or df_s.empty:
        print(f"  [SKIP] One or more merged DataFrames are empty — check column names and core IDs above.")
        continue

    print(f"\n--- {model_name} normalization diagnostics ---")
    print(f"  MNHT subsample (n={len(mnht_subsample_gt)}): gt mean={xmean:.4f}, std={xstd:.4f} | pred mean={ymean:.4f}, std={ystd:.4f}")
    print(f"  Calibration cores excluded: {sorted(calibration_cores)}")
    print(f"  MNHT df_m_eval GT (n={len(df_m_eval)}): mean={df_m_eval[feat_gt].mean():.4f}, std={df_m_eval[feat_gt].std():.4f}")
    print(f"  COL_STL GT        (n={len(df_s)}):      mean={x_valmean:.4f}, std={x_valstd:.4f}")
    print(f"  x   (test GT normalized)  : mean={float(x.mean()):.4f}, std={float(x.std()):.4f}")
    print(f"  y   (test pred normalized): mean={float(y.mean()):.4f}, std={float(y.std()):.4f}")
    print(f"  x_val (val GT normalized) : mean={float(x_val.mean()):.4f}, std={float(x_val.std()):.4f}")
    print(f"  y_val (val pred normalized): mean={float(y_val.mean()):.4f}, std={float(y_val.std()):.4f}")

    pearson_test, spearman_test, r2_test, mse_test, me_test, sde_test = compute_metrics(x, y)
    pearson_val, spearman_val, r2_val, mse_val, me_val, sde_val = compute_metrics(x_val, y_val)

    metrics_test_rows.append(
        {
            "model": model_name,
            "pearson": pearson_test,
            "spearman": spearman_test,
            "r2": r2_test,
            "mse": mse_test,
            "mean_error": me_test,
            "std_error": sde_test,
            "n": len(x),
        }
    )
    metrics_val_rows.append(
        {
            "model": model_name,
            "pearson": pearson_val,
            "spearman": spearman_val,
            "r2": r2_val,
            "mse": mse_val,
            "mean_error": me_val,
            "std_error": sde_val,
            "n": len(x_val),
        }
    )


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.5, label="Test")
    ax.scatter(x_val, y_val, alpha=0.5, label="Validation")
    ax.set_xlabel(f"{feat_gt} GT")
    ax.set_ylabel(f"{feat_pred} Pred")
    ax.set_title(f"{feat_gt} GT vs {feat_pred} Pred ({model_name})")
    stats_text = (
        f"Test:  r={pearson_test:.3f}, ρ={spearman_test:.3f}\n"
        f"Val:   r={pearson_val:.3f}, ρ={spearman_val:.3f}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax.legend()

    # Bland-Altman plot
    fig_ba, ax_ba = plt.subplots(figsize=(8, 6))
    bias_test, loa_test = plot_bland_altman(ax_ba, x, y, label="Test", color="tab:blue")
    bias_val, loa_val = plot_bland_altman(ax_ba, x_val, y_val, label="Validation", color="tab:orange")
    # Draw lines for the test split (primary reference)
    ax_ba.axhline(bias_test, color="tab:blue", linestyle="--", linewidth=1)
    ax_ba.axhline(bias_test + loa_test, color="tab:blue", linestyle=":", linewidth=1)
    ax_ba.axhline(bias_test - loa_test, color="tab:blue", linestyle=":", linewidth=1)
    ax_ba.axhline(0, color="black", linewidth=0.8)
    ba_text = (
        f"Test:  bias={bias_test:.3f}, LoA=[{bias_test - loa_test:.3f}, {bias_test + loa_test:.3f}]\n"
        f"Val:   bias={bias_val:.3f}, LoA=[{bias_val - loa_val:.3f}, {bias_val + loa_val:.3f}]"
    )
    ax_ba.text(
        0.05, 0.95, ba_text,
        transform=ax_ba.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax_ba.set_xlabel("Mean of GT and Pred")
    ax_ba.set_ylabel("Pred − GT")
    ax_ba.set_title(f"Bland-Altman: {model_name}")
    ax_ba.legend()
    os.makedirs("outputs", exist_ok=True)
    ba_path = f"outputs/bland_altman_{model_name}_{sanitize_filename_part(feat_gt)}.png"
    fig_ba.savefig(ba_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ba)
    print(f"Saved Bland-Altman plot: {ba_path}")

    # Denormalize predictions back to GT scale (calibrated)
    # y/y_val are normalized by prediction stats; map them into GT units via GT stats
    y_calib     = y.to_numpy()     * xstd     + xmean
    y_val_calib = y_val.to_numpy() * x_valstd + x_valmean
    x_orig      = df_m_eval[feat_gt].to_numpy()
    x_val_orig  = df_s[feat_gt].to_numpy()

    os.makedirs("outputs", exist_ok=True)
    plot_scatter_calibrated(
        x_orig, y_calib,
        title=f"{model_name} — MNHT",
        save_path=f"outputs/scatter_{model_name}_mnht_{sanitize_filename_part(feat_gt)}.png",
    )
    plot_scatter_calibrated(
        x_val_orig, y_val_calib,
        title=f"{model_name} — COL_STL",
        save_path=f"outputs/scatter_{model_name}_col_stl_{sanitize_filename_part(feat_gt)}.png",
    )

    # Per-core deviation for COL/STL
    df_dev = df_s[["soil_core", feat_gt]].copy()
    df_dev["pred_calibrated"] = y_val_calib
    df_dev["abs_error"] = np.abs(df_dev["pred_calibrated"] - df_dev[feat_gt])
    df_dev = df_dev.sort_values("abs_error", ascending=False).reset_index(drop=True)
    worst = df_dev.iloc[0]
    print(f"\n  [{model_name}] COL/STL worst deviation:")
    print(f"    core={worst['soil_core']}  GT={worst[feat_gt]:.3f}  pred={worst['pred_calibrated']:.3f}  |error|={worst['abs_error']:.3f}")
    print(df_dev.to_string(index=False))

plt.show()

os.makedirs("outputs", exist_ok=True)
df_metrics_test = pd.DataFrame(metrics_test_rows)
df_metrics_val = pd.DataFrame(metrics_val_rows)
feat_gt_file = sanitize_filename_part(feat_gt)
feat_pred_file = sanitize_filename_part(feat_pred)
path_test = f"outputs/model_metrics_mnht_feat_gt_{feat_gt_file}.csv"
path_val = f"outputs/model_metrics_col_stl_feat_gt_{feat_gt_file}.csv"
df_metrics_test.to_csv(path_test, index=False)
df_metrics_val.to_csv(path_val, index=False)
print(f"Saved test metrics: {path_test}")
print(f"Saved validation metrics: {path_val}")