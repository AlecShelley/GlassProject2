import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# Legacy full Sherlock sweep:
# N_ATOMS = 2500000
# PHI_SWEEP = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.90]
# GAMMA_TARGETS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25]

N_ATOMS = 10_000_000
PHI_SWEEP = [0.82, 0.84, 0.86]
GAMMA_TARGETS = [0.03, 0.05, 0.07]
BOX_SIZE = 1.0

FONT = {
    "suptitle": 46,
    "panel_title": 30,
    "axis_label": 32,
    "tick": 30,
    "y_tick": 30,
    "legend": 28,
    "annotation": 30,
    "colorbar_label": 30,
}

def format_scientific_n(n):
    return f"{n:.1e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")

def safe_label(label):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)

def load_run_metadata(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as fh:
            return json.load(fh)
    return {}

def load_run_label(data_dir, run_label=None, metadata=None):
    if run_label:
        return run_label
    metadata = metadata or load_run_metadata(data_dir)
    if "run_id" in metadata:
        return metadata["run_id"]
    return os.path.basename(os.path.normpath(data_dir))

def find_sweep_file(data_dir, phi, grid, gamma_target=None):
    candidates = []
    if gamma_target is not None:
        candidates.append(os.path.join(data_dir, f"sweep_phi{phi:.2f}_gamma{gamma_target:.2f}_grid{grid}.npz"))
    candidates.append(os.path.join(data_dir, f"sweep_phi{phi:.2f}_grid{grid}.npz"))
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return candidates[0]

def render_local_dashboard(data_dir="parameter_sweep_3x3_short", plot_dir=None, run_label=None, n_atoms=None):
    metadata = load_run_metadata(data_dir)
    n_atoms = int(n_atoms or metadata.get("n_atoms", N_ATOMS))
    phi_sweep = metadata.get("phi_sweep", PHI_SWEEP)
    grid_divs = metadata.get("grid_divs")
    gamma_targets = metadata.get("gamma_targets", GAMMA_TARGETS)
    sweep_mode = metadata.get("sweep_mode", "grid" if grid_divs else "gamma")
    column_values = grid_divs if sweep_mode == "grid" else gamma_targets
    expected_arrays = len(phi_sweep) * len(column_values)
    print(f"Scanning '{data_dir}/' for {expected_arrays} arrays...")
    run_label = load_run_label(data_dir, run_label, metadata)
    plot_dir = plot_dir or os.path.join("plots", safe_label(run_label))
    os.makedirs(plot_dir, exist_ok=False)
    
    speedup_matrix = np.zeros((len(phi_sweep), len(column_values)))
    gamma_labels = np.zeros((len(phi_sweep), len(column_values)))
    
    fig, axes = plt.subplots(len(phi_sweep), len(column_values), figsize=(48, 40), sharex='col', sharey='row')
    n_label = format_scientific_n(n_atoms)
    sweep_title = "Fixed-K Sweep" if sweep_mode == "grid" else "3x3 Short Sweep"
    fig.suptitle(f"Async FIRE vs Global FIRE: {sweep_title} (N={n_label})", fontsize=FONT["suptitle"], y=0.94)
    
    for i, phi in enumerate(phi_sweep):
        current_radius = np.sqrt((phi * BOX_SIZE**2) / (n_atoms * np.pi))
        if sweep_mode == "grid":
            k_values = [int(k) for k in column_values]
            gamma_values = [None] * len(k_values)
        else:
            gamma_values = column_values
            k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in gamma_values]
            
        for j, (gamma_target, grid) in enumerate(zip(gamma_values, k_values)):
            ax = axes[i, j]
            filename = find_sweep_file(data_dir, phi, grid, gamma_target)
            
            if os.path.exists(filename):
                data = np.load(filename)
                t_glob, e_glob = data['t_hist_glob'], data['e_hist_glob']
                t_async, e_async = data['t_hist_async'], data['e_hist_async']
                time_glob, time_async = data['time_glob'], data['time_async']
                
                gamma_val = data['gamma'] if 'gamma' in data else (8 * current_radius * grid) / BOX_SIZE
                gamma_labels[i, j] = gamma_val
                
                # Absolute Speedup Multiplier
                speedup_matrix[i, j] = float(time_glob) / float(time_async)
                
                ax.plot(t_glob, e_glob, 'k--', linewidth=4, label='Global')
                ax.plot(t_async, e_async, 'r-', linewidth=4, label='Async')
                ax.set_yscale('log')
                ax.set_title(
                    rf"$\phi$={phi:.2f} | $\gamma$={gamma_val:.2f} | K={grid}x{grid}",
                    fontsize=FONT["panel_title"],
                    pad=20,
                )
                ax.tick_params(axis='x', labelsize=FONT["tick"], width=2, length=8)
                ax.tick_params(axis='y', which='both', labelsize=FONT["y_tick"], width=2, length=8)
                ax.yaxis.get_offset_text().set_fontsize(FONT["y_tick"])
                if i == len(phi_sweep) - 1:
                    ax.set_xlabel("Wall Time (s)", fontsize=FONT["axis_label"], labelpad=18)
                if j == 0:
                    ax.set_ylabel("Energy", fontsize=FONT["axis_label"], labelpad=18)
                ax.grid(alpha=0.3)
                if i == 0 and j == 0: ax.legend(fontsize=FONT["legend"])
            else:
                speedup_matrix[i, j] = np.nan
                ax.text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=FONT["annotation"])
                ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    run_slug = safe_label(run_label)
    dashboard_path = os.path.join(plot_dir, f"FigS1_Short_3x3_Dashboard_{run_slug}.pdf")
    plt.savefig(dashboard_path, bbox_inches='tight')
    plt.close()
    print(f"Short 3x3 dashboard saved to {dashboard_path}.")

    # 2. GENERATE THE HEATMAP PHASE DIAGRAM
    masked_data = np.ma.masked_invalid(np.flipud(speedup_matrix))
    y_labels = [f"{p:.2f}" for p in reversed(phi_sweep)]
    if sweep_mode == "grid":
        x_labels = [str(int(k)) for k in column_values]
    else:
        x_labels = [f"{g:.2f}" for g in column_values]
    
    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked_data, cmap=cmap, vmin=0.5, vmax=3.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wall-Clock Speedup Multiplier", fontsize=FONT["colorbar_label"], labelpad=22)
    cbar.ax.tick_params(labelsize=FONT["tick"], width=2, length=8)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=FONT["tick"])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=FONT["y_tick"])
    ax.tick_params(axis='y', which='both', labelsize=FONT["y_tick"], width=2, length=8)

    mask = np.ma.getmaskarray(masked_data)
    for row in range(masked_data.shape[0]):
        for col in range(masked_data.shape[1]):
            if not mask[row, col]:
                ax.text(col, row, f"{masked_data[row, col]:.1f}", ha="center", va="center", color="black", fontsize=FONT["annotation"])
    
    ax.set_title(f"{sweep_title} Speedup Check (N={n_label})", fontsize=FONT["suptitle"], pad=28)
    xlabel = "Grid Divisions (K)" if sweep_mode == "grid" else r"Boundary Friction Fraction ($\gamma$)"
    ax.set_xlabel(xlabel, fontsize=FONT["axis_label"], labelpad=24)
    ax.set_ylabel(r"Packing Fraction ($\phi$)", fontsize=FONT["axis_label"], labelpad=24)
    ax.tick_params(axis='both', width=2, length=8)
    
    heatmap_path = os.path.join(plot_dir, f"Fig3_Short_3x3_Speedup_Heatmap_{run_slug}.pdf")
    fig.savefig(heatmap_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Optimization Heatmap saved to {heatmap_path}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Render the 3x3 Async FIRE dashboard.")
    parser.add_argument("--data-dir", default="parameter_sweep_3x3_short")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--n-atoms", type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    render_local_dashboard(args.data_dir, args.plot_dir, args.run_label, args.n_atoms)
