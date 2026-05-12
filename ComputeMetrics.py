import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

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
    df_pred = pd.read_csv(f"outputs/prediction_{model_name}_col_stl.csv")
    df_pred_val = pd.read_csv(f"outputs/prediction_{model_name}_mnht.csv")

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
    xmean, ymean = df_m_full[feat_gt].mean(), df_m_full[f"pred_{feat_pred}"].mean()
    xstd, ystd = df_m_full[feat_gt].std(), df_m_full[f"pred_{feat_pred}"].std()
    x_valmean, y_valmean = df_s[feat_gt].mean(), df_s[f"pred_{feat_pred}"].mean()
    x_valstd, y_valstd = df_s[feat_gt].std(), df_s[f"pred_{feat_pred}"].std()

    x = (df_m[feat_gt] - xmean) / xstd
    y = (df_m[f"pred_{feat_pred}"] - ymean) / ystd
    x_val = (df_s[feat_gt] - x_valmean) / x_valstd
    y_val = (df_s[f"pred_{feat_pred}"] - y_valmean) / y_valstd
    # x_val = x_val * xstd + xmean
    #y_val = y_val * ystd + ymean
    # x = x*xstd + xmean
    #y = y*ystd + ymean
    # x_val = x_val*x_valstd + x_valmean
    #y_val = y_val*y_valstd + y_valmean

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
    ax_ba.set_title(f"Bland-Altman: {feat_gt} vs {feat_pred} ({model_name})")
    ax_ba.legend()
    os.makedirs("outputs", exist_ok=True)
    ba_path = f"outputs/bland_altman_{model_name}_{sanitize_filename_part(feat_gt)}.png"
    fig_ba.savefig(ba_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ba)
    print(f"Saved Bland-Altman plot: {ba_path}")

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