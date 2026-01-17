import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

def plot_diagonal(R, layer_name):
    
    R_cpu = R.detach().float().cpu()
    m = R_cpu.shape[1]
    dims = torch.arange(m)

    # --- per-column stats (use absolute values) ---
    diag_vals = torch.diag(R_cpu).abs()                # |diag|
    off_sum = torch.full((m,), float("nan"))         # |mean(off-diag)|
    off_stds  = torch.full((m,), float("nan"))         # std(|off-diag|)
    for j in range(2, m):
        col_upper = R_cpu[:j, j]                # strictly above diagonal, absolute
        off_sum[j] = col_upper.abs().sum().item()
        off_stds[j]  = col_upper.std(unbiased=False).item()

    # --- safe ratio using absolute values ---
    x_full   = dims.numpy()
    y_diag   = diag_vals.numpy()
    y_sum   = off_sum.numpy()
    y_std    = off_stds.numpy()

    y_ratio = np.abs(y_sum) / np.abs(y_diag)
    y_ratio = np.nan_to_num(y_ratio, nan=0.0)

    # --- sample every 50 columns (and include the last column) ---
    step = 50
    idx = np.arange(1, m, step)
    if idx[-1] != m - 1:
        idx = np.r_[idx, m - 1]  # ensure last point included

    xs      = x_full[idx]
    y_diag_s  = y_diag[idx]
    y_sum_s  = y_sum[idx]
    y_std_s   = y_std[idx]
    y_ratio_s = y_ratio[idx]

    # --- plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(xs, y_diag_s, linewidth=2, marker='o', markersize=3, label='Diagonal')
    plt.plot(xs, y_sum_s, linewidth=2, linestyle='--', marker='s', markersize=3, label='Off-diagonal sum')

    # Ratio (|off-mean| / |diag|)
    plt.plot(xs, np.abs(y_ratio_s), linewidth=2, marker='x', markersize=3, label='Ratio = |off-sum| / |diag|')

    # Labels / styling
    if layer_name:
        layer_info = f"({layer_name})"

    plt.yscale('symlog', linthresh=1e-8, linscale=1.0, subs=None, base=10)

    plt.xlabel('Dimension (column index)')
    plt.ylabel('Value')
    # If you want log y-axis now that values are nonnegative, uncomment:
    # plt.yscale('log')
    plt.title(layer_info)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_key = f"{layer_info.replace(' ', '_').replace('(', '').replace(')', '')}_R_diag"
    wandb.log({plot_key: wandb.Image(plt)})

    plt.close()
