import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os

# ==========================================
# 1. ACCELERATED G(R) CALCULATOR
# ==========================================
@njit
def calculate_gr_numba(pos, box_size, radius, bins=100, r_max=4.0):
    r_max_scaled = r_max * (2*radius)
    hist = np.zeros(bins)
    edges = np.linspace(0, r_max_scaled, bins+1)
    dr = edges[1] - edges[0]
    n_pts = pos.shape[0]

    for i in range(n_pts):
        for j in range(i+1, n_pts):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dx = dx - box_size * np.round(dx / box_size)
            dy = dy - box_size * np.round(dy / box_size)
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < r_max_scaled and dist > 1e-8:
                idx = int(dist / dr)
                if idx < bins:
                    hist[idx] += 2.0  # +2 accounts for i->j and j->i
                    
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    density = n_pts / (box_size**2)
    ideal_counts = 2 * np.pi * r_centers * dr * density * n_pts
    gr = hist / ideal_counts
    return r_centers / (2*radius), gr

# ==========================================
# 2. PRL RENDER ROUTINES
# ==========================================
def render_missing_prl_figures(data_file="structural_data.npz"):
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run capture script first.")
        return

    data = np.load(data_file)
    pos_async = data['pos_async']
    pos_glob = data['pos_glob']
    forces = data['forces_async']
    d_dt = data['d_dt']
    grid_divs = int(data['grid_divs'])
    radius = float(data['radius'])
    BOX_SIZE = 1.0
    
    plt.rcParams.update({'font.size': 14, 'axes.linewidth': 2})
    
    # --- FIGURE 1: Mathematical Phase Separation ---
    print("Rendering Fig 1: Bimodal Histogram...")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.hist(d_dt, bins=25, color='purple', edgecolor='black', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Local Domain Time Step ($\Delta t_k$)")
    ax1.set_ylabel("Number of Domains")
    ax1.set_title("Mathematical Phase Separation")
    plt.tight_layout()
    plt.savefig("Fig1_Bimodal_Histogram.pdf")
    
    # --- FIGURE 2: Spatial Sensor Map ---
    print("Rendering Fig 2: Spatial Sensor Map...")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    dt_grid = d_dt.reshape((grid_divs, grid_divs)).T
    im = ax2.imshow(dt_grid, extent=[0, BOX_SIZE, 0, BOX_SIZE], origin='lower', 
                    cmap='coolwarm', alpha=0.6)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(r"Local Time Step ($\Delta t$)")
    
    force_mags = np.linalg.norm(forces, axis=1)
    active_mask = force_mags > (np.mean(force_mags) * 0.5) 
    ax2.scatter(pos_async[active_mask, 0], pos_async[active_mask, 1], 
                s=0.5, c='black', alpha=0.6)
    
    ax2.set_xlim(0, BOX_SIZE); ax2.set_ylim(0, BOX_SIZE)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Autonomous Sensor Validation")
    plt.tight_layout()
    plt.savefig("Fig2_Spatial_Sensor.pdf")
    
    # --- FIGURE 3: Topological Invariance g(r) ---
    print("Rendering Fig 3: Topological Invariance g(r)...")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    r_async, gr_async = calculate_gr_numba(pos_async, BOX_SIZE, radius)
    r_glob, gr_glob = calculate_gr_numba(pos_glob, BOX_SIZE, radius)
    
    ax3.plot(r_glob, gr_glob, 'k--', linewidth=4, label='Global FIRE')
    ax3.plot(r_async, gr_async, 'r-', linewidth=2, label='Async FIRE')
    ax3.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, label='Contact Radius')
    
    ax3.set_xlim(0.8, 2.5)
    ax3.set_xlabel(r"Distance $r/2R$")
    ax3.set_ylabel(r"Radial Distribution Function $g(r)$")
    ax3.set_title("Topological Invariance")
    ax3.legend()
    plt.tight_layout()
    plt.savefig("Fig3_Topological_Invariance.pdf")
    print("All figures successfully rendered.")

if __name__ == '__main__':
    render_missing_prl_figures()