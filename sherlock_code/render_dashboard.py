import os
import numpy as np
import matplotlib.pyplot as plt

N_ATOMS = 2500000 
PHI_SWEEP = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.90]
GAMMA_TARGETS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25]
BOX_SIZE = 1.0

def render_local_dashboard(data_dir="parameter_sweep_10x10"):
    print(f"Scanning '{data_dir}/' for 100 arrays...")
    
    speedup_matrix = np.zeros((len(PHI_SWEEP), len(GAMMA_TARGETS)))
    gamma_labels = np.zeros((len(PHI_SWEEP), len(GAMMA_TARGETS)))
    
    fig, axes = plt.subplots(len(PHI_SWEEP), len(GAMMA_TARGETS), figsize=(48, 40), sharex='col', sharey='row')
    fig.suptitle(f"Async FIRE vs Global FIRE: 100-State Parameter Sweep (N={N_ATOMS})", fontsize=36, y=0.92)
    
    for i, phi in enumerate(PHI_SWEEP):
        current_radius = np.sqrt((phi * BOX_SIZE**2) / (N_ATOMS * np.pi))
        k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in GAMMA_TARGETS]
        k_values = sorted(list(set(k_values)))
        while len(k_values) < len(GAMMA_TARGETS):
            k_values.append(k_values[-1] + 2)
            
        for j, grid in enumerate(k_values[:len(GAMMA_TARGETS)]):
            ax = axes[i, j]
            filename = os.path.join(data_dir, f"sweep_phi{phi:.2f}_grid{grid}.npz")
            
            if os.path.exists(filename):
                data = np.load(filename)
                t_glob, e_glob = data['t_hist_glob'], data['e_hist_glob']
                t_async, e_async = data['t_hist_async'], data['e_hist_async']
                time_glob, time_async = data['time_glob'], data['time_async']
                
                gamma_val = data['gamma'] if 'gamma' in data else (8 * current_radius * grid) / BOX_SIZE
                gamma_labels[i, j] = gamma_val
                
                # Absolute Speedup Multiplier
                speedup_matrix[i, j] = float(time_glob) / float(time_async)
                
                ax.plot(t_glob, e_glob, 'k--', linewidth=2, label='Global')
                ax.plot(t_async, e_async, 'r-', linewidth=2, label='Async')
                ax.set_yscale('log')
                ax.set_title(rf"$\phi$={phi:.2f} | $\gamma$={gamma_val:.2f}", fontsize=14)
                ax.grid(alpha=0.3)
                if i == 0 and j == 0: ax.legend()
            else:
                speedup_matrix[i, j] = np.nan
                ax.text(0.5, 0.5, "Missing", ha='center', va='center')
                ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig("FigS1_Massive_10x10_Dashboard.pdf", bbox_inches='tight')
    plt.close()
    print("Massive 10x10 Dashboard saved.")

    # 2. GENERATE THE HEATMAP PHASE DIAGRAM
    plt.figure(figsize=(12, 8))
    masked_data = np.ma.masked_invalid(np.flipud(speedup_matrix))
    y_labels = [f"{p:.2f}" for p in reversed(PHI_SWEEP)]
    x_labels = [f"{g:.2f}" for g in GAMMA_TARGETS]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked_data, cmap=cmap, vmin=0.5, vmax=3.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wall-Clock Speedup Multiplier")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    mask = np.ma.getmaskarray(masked_data)
    for row in range(masked_data.shape[0]):
        for col in range(masked_data.shape[1]):
            if not mask[row, col]:
                ax.text(col, row, f"{masked_data[row, col]:.1f}", ha="center", va="center", color="black")
    
    plt.title(f"Integrability Phase Diagram: Absolute Speedup (N={N_ATOMS})", fontsize=16)
    plt.xlabel(r"Boundary Friction Fraction ($\gamma$)", fontsize=14)
    plt.ylabel(r"Packing Fraction ($\phi$)", fontsize=14)
    
    plt.savefig("Fig3_Speedup_Heatmap.pdf", bbox_inches='tight')
    print("Optimization Heatmap saved.")

if __name__ == '__main__':
    render_local_dashboard()
