#!/usr/bin/env python3
"""
Standalone script to generate the 4 publication figures for the
Async FIRE paper. No DOCX dependencies — only numpy + matplotlib.

Output: paper_figures/ directory with PNG + PDF for each figure.

  fig1_phase_diagram  — The Phenomenological Phase Diagram
  fig2_mechanism      — The Mechanism of Failure
  fig3_regularization — Topological Regularization (D,K coupling)
  fig4_reliability    — Algorithmic Reliability (multi-seed)

Usage:
  python generate_plots.py                  # default: ~/Downloads
  python generate_plots.py --data-dir /path/to/data
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ── CLI ──
parser = argparse.ArgumentParser(description='Generate Async FIRE paper figures')
parser.add_argument('--data-dir', default=os.path.expanduser('~/Downloads'),
                    help='Root directory containing experiment .npz data')
parser.add_argument('--output-dir', default=None,
                    help='Output directory for figures (default: <data-dir>/paper_figures)')
args = parser.parse_args()

BASE = args.data_dir
FIG_DIR = args.output_dir or os.path.join(BASE, 'paper_figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Plot style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
})

C_GLOB = '#d62728'
C_ASYNC = '#1f77b4'
C_ASYNC2 = '#2ca02c'
C_GOLD = '#DAA520'


def load(path):
    return np.load(path, allow_pickle=True)


def save_fig(fig, name):
    png = os.path.join(FIG_DIR, f'{name}.png')
    pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f'  Saved: {png}')
    print(f'  Saved: {pdf}')


# ════════════════════════════════════════════════════════════════
# FIGURE 1: The Phenomenological Phase Diagram
# ════════════════════════════════════════════════════════════════
def make_fig1():
    print('Building Figure 1: Phase Diagram...')
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.32)

    # --- Panel A: Final Energy vs phi ---
    ax_a = fig.add_subplot(gs[0])

    phi_data = {}

    for subdir, phis in [('ttt_low_phi', [0.76, 0.80, 0.84]),
                          ('ttt_low_phi_boundary', [0.77, 0.78, 0.79])]:
        d = os.path.join(BASE, subdir)
        for phi in phis:
            gf = os.path.join(d, f'ttt_phi{phi:.2f}_global.npz')
            af = os.path.join(d, f'ttt_phi{phi:.2f}_async_K7.npz')
            if os.path.exists(gf):
                g = load(gf)
                a = load(af)
                phi_data[phi] = {
                    'e_glob': float(g['energy_trace'][-1]),
                    'e_async': float(a['energy_trace'][-1]),
                }

    ttt = os.path.join(BASE, 'time_to_target')
    for phi in [0.85, 0.86, 0.87, 0.88, 0.89, 0.90]:
        gf = os.path.join(ttt, f'ttt_phi{phi:.2f}_global.npz')
        af = os.path.join(ttt, f'ttt_phi{phi:.2f}_async_K7.npz')
        if os.path.exists(gf):
            g = load(gf)
            a = load(af)
            phi_data[phi] = {
                'e_glob': float(g['energy_trace'][-1]),
                'e_async': float(a['energy_trace'][-1]),
            }

    phis = sorted(phi_data.keys())
    e_glob = [max(phi_data[p]['e_glob'], 1e-32) for p in phis]
    e_async = [max(phi_data[p]['e_async'], 1e-32) for p in phis]

    ax_a.axvspan(0.755, 0.775, color='#ffcccc', alpha=0.4, zorder=0)
    ax_a.axvspan(0.775, 0.875, color='#ccffcc', alpha=0.3, zorder=0)
    ax_a.axvspan(0.875, 0.905, color='#ffcccc', alpha=0.4, zorder=0)

    ax_a.text(0.765, 1e-20, 'Stall', fontsize=9, color='#cc0000', fontweight='bold',
              rotation=90, va='center', ha='center')
    ax_a.text(0.825, 1e-20, 'Global wins', fontsize=10, color='#006600',
              fontweight='bold', ha='center', va='center')
    ax_a.text(0.89, 1e-20, 'Diverge', fontsize=9, color='#cc0000', fontweight='bold',
              rotation=90, va='center', ha='center')

    ax_a.semilogy(phis, e_glob, 'o-', color=C_GLOB, label='Global FIRE',
                  markersize=8, zorder=5, linewidth=2)
    ax_a.semilogy(phis, e_async, 's-', color=C_ASYNC, label='Async FIRE (K=7, D=3)',
                  markersize=8, zorder=5, linewidth=2)

    ax_a.set_xlabel('Packing fraction $\\phi$')
    ax_a.set_ylabel('Final energy')
    ax_a.set_title('(a) Phase diagram', fontweight='bold')
    ax_a.legend(loc='lower left', fontsize=9)
    ax_a.set_xlim(0.755, 0.905)
    ax_a.set_ylim(1e-32, 1e4)

    ax_a.axvline(x=0.775, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_a.axvline(x=0.875, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # --- Panel B: phi=0.77 low-density stall ---
    ax_b = fig.add_subplot(gs[1])
    d = os.path.join(BASE, 'ttt_low_phi_boundary')
    g = load(os.path.join(d, 'ttt_phi0.77_global.npz'))
    a = load(os.path.join(d, 'ttt_phi0.77_async_K7.npz'))

    ax_b.semilogy(g['time_trace'], g['energy_trace'], color=C_GLOB,
                   label='Global FIRE', linewidth=2, alpha=0.9)
    ax_b.semilogy(a['time_trace'], a['energy_trace'], color=C_ASYNC,
                   label='Async FIRE', linewidth=2, alpha=0.9)

    ax_b.axhline(y=1e-4, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.text(280, 2e-4, '$E = 10^{-4}$', fontsize=8, color='gray', ha='right')

    eg = g['energy_trace'][-1]
    ax_b.annotate(f'Trapped at\n$E = {eg:.1e}$',
                  xy=(200, eg), fontsize=9, color=C_GLOB,
                  fontweight='bold', ha='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.8))

    ax_b.set_xlabel('Wall-clock time (s)')
    ax_b.set_ylabel('Total energy')
    ax_b.set_title('(b) Low-density stall, $\\phi = 0.77$', fontweight='bold')
    ax_b.legend(fontsize=9, loc='upper right')
    ax_b.set_ylim(1e-8, 2e3)

    # --- Panel C: phi=0.90 high-density divergence ---
    ax_c = fig.add_subplot(gs[2])
    ttt_dir = os.path.join(BASE, 'time_to_target')
    g = load(os.path.join(ttt_dir, 'ttt_phi0.90_global.npz'))
    a = load(os.path.join(ttt_dir, 'ttt_phi0.90_async_K7.npz'))

    ax_c.semilogy(g['time_trace'], g['energy_trace'], color=C_GLOB,
                   label='Global FIRE', linewidth=2, alpha=0.9)
    ax_c.semilogy(a['time_trace'], a['energy_trace'], color=C_ASYNC,
                   label='Async FIRE', linewidth=2, alpha=0.9)

    ax_c.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    eg = g['energy_trace'][-1]
    ea = a['energy_trace'][-1]
    ax_c.annotate(f'$E = {eg:.0f}$\n(diverged)',
                  xy=(250, eg), fontsize=9, color=C_GLOB,
                  fontweight='bold', ha='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.8))
    ax_c.annotate(f'$E = {ea:.3f}$',
                  xy=(250, ea), fontsize=9, color=C_ASYNC,
                  ha='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0e8ff', alpha=0.8))

    ax_c.set_xlabel('Wall-clock time (s)')
    ax_c.set_ylabel('Total energy')
    ax_c.set_title('(c) High-density divergence, $\\phi = 0.90$', fontweight='bold')
    ax_c.legend(fontsize=9, loc='center right')
    ax_c.set_ylim(1e-1, 2e3)

    fig.suptitle('Figure 1. Global FIRE fails at both extremes of the jamming transition',
                 fontweight='bold', fontsize=14, y=1.03)
    plt.tight_layout()
    save_fig(fig, 'fig1_phase_diagram')


# ════════════════════════════════════════════════════════════════
# FIGURE 2: The Mechanism of Failure
# ════════════════════════════════════════════════════════════════
def make_fig2():
    print('Building Figure 2: Mechanism of Failure...')
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    ra = os.path.join(BASE, 'reset_analysis')
    phis = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90]

    # --- Panel A: Reset rate bar chart ---
    g_rates = []
    a_rates = []
    for phi in phis:
        g = load(os.path.join(ra, f'reset_phi{phi:.2f}_global.npz'))
        a = load(os.path.join(ra, f'reset_phi{phi:.2f}_async_K7.npz'))
        g_rates.append(float(g['reset_rate']) * 100)
        a_rates.append(float(a['avg_frac_reset']) * 100)

    x = np.arange(len(phis))
    w = 0.35
    ax_a.bar(x - w/2, g_rates, w, color=C_GLOB, label='Global (% steps with reset)',
             alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_a.bar(x + w/2, a_rates, w, color=C_ASYNC, label='Async (% domains resetting)',
             alpha=0.85, edgecolor='black', linewidth=0.5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f'{p:.2f}' for p in phis])
    ax_a.set_xlabel('Packing fraction $\\phi$')
    ax_a.set_ylabel('Reset rate (%)')
    ax_a.set_title('(a) The frequency myth: Global resets\n'
                    'are RARE at high density', fontweight='bold')
    ax_a.legend(fontsize=8, loc='upper right')

    ax_a.annotate(f'{g_rates[-1]:.1f}% resets\nbut $E = 860$',
                  xy=(x[-1] - w/2, g_rates[-1]),
                  xytext=(x[-1] - 1.5, g_rates[-1] + 3),
                  fontsize=9, fontweight='bold', color=C_GLOB,
                  arrowprops=dict(arrowstyle='->', color=C_GLOB, lw=1.5),
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.9))

    # --- Panel B: Fraction at dt_max ---
    phi_styles = {0.85: ('-', C_ASYNC), 0.87: ('--', C_ASYNC2), 0.90: ('-.', C_GLOB)}
    for phi, (ls, col) in phi_styles.items():
        a = load(os.path.join(ra, f'reset_phi{phi:.2f}_async_K7.npz'))
        ax_b.plot(a['time_trace'], a['frac_at_dtmax_trace'],
                  label=f'$\\phi = {phi:.2f}$', linewidth=2, linestyle=ls, color=col)

    ax_b.set_xlabel('Wall-clock time (s)')
    ax_b.set_ylabel('Fraction of domains at $\\Delta t_{\\mathrm{max}}$')
    ax_b.set_title('(b) Spatial heterogeneity: 94% cruise,\n'
                    '6% actively throttle at $\\phi = 0.90$', fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.set_ylim(0, 1.05)

    ax_b.axhline(y=0.94, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_b.text(5, 0.955, '94%', fontsize=9, color='gray')

    ax_b.fill_between([0, 120], 0.88, 1.0, color=C_ASYNC, alpha=0.05)
    ax_b.annotate('6% active\nthrottling',
                  xy=(100, 0.94), xytext=(80, 0.75),
                  fontsize=9, fontweight='bold', color=C_GLOB,
                  arrowprops=dict(arrowstyle='->', color=C_GLOB, lw=1),
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    fig.suptitle('Figure 2. The problem is reset granularity, not frequency',
                 fontweight='bold', fontsize=14, y=1.03)
    plt.tight_layout()
    save_fig(fig, 'fig2_mechanism')


# ════════════════════════════════════════════════════════════════
# FIGURE 3: Topological Regularization
# ════════════════════════════════════════════════════════════════
def make_fig3():
    print('Building Figure 3: Topological Regularization...')

    dirs = {
        2:  os.path.join(BASE, 'sweep_delayed_2'),
        3:  os.path.join(BASE, 'sweep_delayed'),
        5:  os.path.join(BASE, 'sweep_delayed_5'),
        7:  os.path.join(BASE, 'sweep_delayed_7'),
        8:  os.path.join(BASE, 'sweep_delayed_8'),
        10: os.path.join(BASE, 'sweep_delayed_10'),
    }

    results = {}
    for D, d in dirs.items():
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith('.npz'):
                continue
            data = np.load(os.path.join(d, f))
            phi = float(data['phi'])
            grid = int(data['grid_divs'])
            e_async = float(data['e_hist_async'][-1])
            results[(phi, D, grid)] = e_async

    D_values = sorted(dirs.keys())

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2], wspace=0.30)

    # --- Panel A: Heatmap at phi=0.90 ---
    ax_a = fig.add_subplot(gs[0])
    phi = 0.90
    K_values = sorted(set(g for (p, dd, g) in results if abs(p - phi) < 0.001))

    matrix = np.full((len(D_values), len(K_values)), np.nan)
    for i, D in enumerate(D_values):
        for j, K in enumerate(K_values):
            key = (phi, D, K)
            if key in results:
                matrix[i, j] = results[key]

    display = matrix.copy()
    display[matrix > 100] = np.nan

    im = ax_a.imshow(display, aspect='auto', cmap='viridis_r',
                     vmin=0.7, vmax=1.0, interpolation='nearest')

    for i in range(len(D_values)):
        for j in range(len(K_values)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if val > 100:
                txt, color, wt = 'FAIL', 'red', 'bold'
            else:
                txt = f'{val:.3f}'
                color = 'white' if val < 0.78 else 'black'
                valid = display[~np.isnan(display)]
                wt = 'bold' if len(valid) > 0 and val == np.nanmin(valid) else 'normal'
            ax_a.text(j, i, txt, ha='center', va='center',
                      fontsize=8, color=color, fontweight=wt)

    ax_a.set_xticks(range(len(K_values)))
    ax_a.set_xticklabels(K_values)
    ax_a.set_yticks(range(len(D_values)))
    ax_a.set_yticklabels([f'$D={d}$' for d in D_values])
    ax_a.set_xlabel('Grid size $K$')
    ax_a.set_ylabel('Delayed reset patience $D$')
    ax_a.set_title('(a) $(D, K)$ energy landscape,\n$\\phi = 0.90$', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax_a, shrink=0.8, pad=0.02)
    cbar.set_label('Final energy')

    best_val = np.inf
    best_i, best_j = 0, 0
    for i in range(len(D_values)):
        for j in range(len(K_values)):
            v = matrix[i, j]
            if not np.isnan(v) and v < best_val:
                best_val = v
                best_i, best_j = i, j
    rect = Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                      linewidth=3, edgecolor='lime', facecolor='none')
    ax_a.add_patch(rect)

    # --- Panel B: Optimal K* vs D ---
    ax_b = fig.add_subplot(gs[1])

    opt_K = []
    opt_E = []
    valid_D = []
    for D in D_values:
        K_vals = sorted(set(g for (p, dd, g) in results if abs(p - phi) < 0.001))
        bk, be = None, np.inf
        for K in K_vals:
            key = (phi, D, K)
            if key in results and results[key] < be and results[key] < 100:
                be = results[key]
                bk = K
        if bk is not None:
            valid_D.append(D)
            opt_K.append(bk)
            opt_E.append(be)

    ax_b.plot(valid_D, opt_K, 'o-', color=C_ASYNC, markersize=10, linewidth=2.5)

    for d, k, e in zip(valid_D, opt_K, opt_E):
        ax_b.annotate(f'$E={e:.3f}$', (d, k), textcoords='offset points',
                      xytext=(12, 8), fontsize=8, color=C_ASYNC,
                      arrowprops=dict(arrowstyle='->', color=C_ASYNC, lw=0.5))

    ax_b.axhspan(opt_K[-2] - 2, opt_K[-1] + 5, color=C_GOLD, alpha=0.1)
    ax_b.annotate('Saturation\n$D \\geq 8$',
                  xy=(9, opt_K[-1]), fontsize=9, fontweight='bold',
                  color=C_GOLD, ha='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8dc', alpha=0.9))

    ax_b.set_xlabel('Delayed reset patience $D$')
    ax_b.set_ylabel('Optimal grid size $K^*$')
    ax_b.set_title('(b) Operating curve,\n$\\phi = 0.90$', fontweight='bold')
    ax_b.set_xticks(D_values)

    # --- Panel C: Grid-insensitivity at phi=0.84, D=10 ---
    ax_c = fig.add_subplot(gs[2])
    d10 = os.path.join(BASE, 'sweep_delayed_10')

    K_colors = {7: '#1f77b4', 15: '#ff7f0e', 22: '#2ca02c', 29: '#d62728', 36: '#9467bd'}

    for K in sorted(K_colors.keys()):
        fpath = os.path.join(d10, f'sweep_phi0.84_grid{K}.npz')
        if not os.path.exists(fpath):
            continue
        data = load(fpath)
        t = data['t_hist_async']
        e = data['e_hist_async']
        ax_c.semilogy(t, e, color=K_colors[K], label=f'$K = {K}$',
                      linewidth=2, alpha=0.85)

    fpath = os.path.join(d10, 'sweep_phi0.84_grid7.npz')
    if os.path.exists(fpath):
        data = load(fpath)
        ax_c.semilogy(data['t_hist_glob'], data['e_hist_glob'],
                      color='black', label='Global', linewidth=2.5,
                      linestyle='--', alpha=0.6)

    ax_c.set_xlabel('Wall-clock time (s)')
    ax_c.set_ylabel('Energy')
    ax_c.set_title('(c) Grid-insensitivity:\n'
                    '$D = 10$, $\\phi = 0.84$ — all $K$ collapse', fontweight='bold')
    ax_c.legend(fontsize=8, ncol=2, loc='upper right')
    ax_c.set_ylim(1e-7, 1e3)

    ax_c.axhspan(8e-7, 1.5e-6, color=C_ASYNC, alpha=0.1)
    ax_c.text(60, 1.1e-6, '$E \\approx 1.1 \\times 10^{-6}$ (all $K$)',
              fontsize=9, fontweight='bold', color=C_ASYNC, ha='right',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#e0e8ff', alpha=0.8))

    fig.suptitle('Figure 3. The $(D, K)$ coupling: temporal patience enables spatial refinement',
                 fontweight='bold', fontsize=14, y=1.03)
    plt.tight_layout()
    save_fig(fig, 'fig3_regularization')


# ════════════════════════════════════════════════════════════════
# FIGURE 4: Algorithmic Reliability
# ════════════════════════════════════════════════════════════════
def make_fig4():
    print('Building Figure 4: Algorithmic Reliability...')

    seeds = [42, 137, 256, 512, 1024]
    seed_colors = {42: '#1f77b4', 137: '#ff7f0e', 256: '#2ca02c', 512: '#d62728', 1024: '#9467bd'}

    def get_dir(seed, phi):
        if seed == 42:
            return os.path.join(BASE, 'ttt_low_phi_boundary') if phi < 0.85 else os.path.join(BASE, 'time_to_target')
        return os.path.join(BASE, f'ttt_seed{seed}')

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel A: Spaghetti plot at phi=0.88 ---
    g_finals_88 = []
    a_finals_88 = []

    for seed in seeds:
        d = get_dir(seed, 0.88)
        gf = os.path.join(d, f'ttt_phi0.88_global.npz')
        af = os.path.join(d, f'ttt_phi0.88_async_K7.npz')
        if not os.path.exists(gf):
            continue

        g = load(gf)
        a = load(af)

        ax_a.semilogy(g['time_trace'], g['energy_trace'], color=seed_colors[seed],
                      linewidth=1.5, alpha=0.7, linestyle='--')
        ax_a.semilogy(a['time_trace'], a['energy_trace'], color=seed_colors[seed],
                      linewidth=1.5, alpha=0.7, linestyle='-')

        g_finals_88.append(float(g['energy_trace'][-1]))
        a_finals_88.append(float(a['energy_trace'][-1]))

    custom_lines = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
    ]
    for s in seeds:
        custom_lines.append(Line2D([0], [0], color=seed_colors[s], linewidth=2.5))
    labels = ['Global', 'Async'] + [f'seed {s}' for s in seeds]
    ax_a.legend(custom_lines, labels, fontsize=7, ncol=2, loc='lower left')

    ax_a.set_xlabel('Wall-clock time (s)')
    ax_a.set_ylabel('Total energy')
    ax_a.set_title('(a) $\\phi = 0.88$: Global trajectories scatter\n'
                    'wildly; Async trajectories cluster tightly', fontweight='bold')
    ax_a.set_ylim(1e-6, 1e4)

    ax_a.annotate(f'Global: $E$ = {min(g_finals_88):.1e} to {max(g_finals_88):.0f}',
                  xy=(0.02, 0.98), xycoords='axes fraction',
                  fontsize=9, fontweight='bold', color=C_GLOB, va='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.9))
    ax_a.annotate(f'Async: $E$ = {min(a_finals_88):.3f} to {max(a_finals_88):.3f}',
                  xy=(0.02, 0.88), xycoords='axes fraction',
                  fontsize=9, fontweight='bold', color=C_ASYNC, va='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0e8ff', alpha=0.9))

    # --- Panel B: Variability ratio ---
    phis_test = [0.77, 0.78, 0.88, 0.89]
    g_ratios = []
    a_ratios = []
    phi_labels = []

    for phi in phis_test:
        g_finals = []
        a_finals = []
        for seed in seeds:
            d = get_dir(seed, phi)
            gf = os.path.join(d, f'ttt_phi{phi:.2f}_global.npz')
            af = os.path.join(d, f'ttt_phi{phi:.2f}_async_K7.npz')
            if not os.path.exists(gf):
                continue
            g = load(gf)
            a = load(af)
            gv = float(g['energy_trace'][-1])
            av = float(a['energy_trace'][-1])
            if gv > 0:
                g_finals.append(gv)
            a_finals.append(av)

        if g_finals and a_finals:
            g_ratio = max(g_finals) / max(min(g_finals), 1e-32)
            a_ratio = max(a_finals) / max(min(a_finals), 1e-32)
            g_ratios.append(np.log10(max(g_ratio, 1)))
            a_ratios.append(np.log10(max(a_ratio, 1)))
            phi_labels.append(f'$\\phi = {phi:.2f}$')

    x = np.arange(len(phi_labels))
    w = 0.35
    ax_b.bar(x - w/2, g_ratios, w, color=C_GLOB, alpha=0.85,
             edgecolor='black', linewidth=0.5, label='Global FIRE')
    ax_b.bar(x + w/2, a_ratios, w, color=C_ASYNC, alpha=0.85,
             edgecolor='black', linewidth=0.5, label='Async FIRE')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(phi_labels)
    ax_b.set_ylabel('$\\log_{10}$(max / min final energy across 5 seeds)')
    ax_b.set_title('(b) Outcome variability: Global is\n'
                    'unpredictable near boundaries', fontweight='bold')
    ax_b.legend(fontsize=9)

    for i, (gr, ar) in enumerate(zip(g_ratios, a_ratios)):
        g_val = 10**gr
        a_val = 10**ar
        g_str = f'{g_val:.0e}' if g_val > 100 else f'{g_val:.1f}'
        a_str = f'{a_val:.1f}'
        ax_b.text(i - w/2, gr + 0.15, f'{g_str}\u00d7', ha='center', fontsize=8,
                  fontweight='bold', color=C_GLOB)
        ax_b.text(i + w/2, ar + 0.15, f'{a_str}\u00d7', ha='center', fontsize=8,
                  fontweight='bold', color=C_ASYNC)

    ax_b.axhline(y=0, color='black', linewidth=0.5)

    fig.suptitle('Figure 4. Async FIRE delivers predictable results; Global FIRE is a lottery near boundaries',
                 fontweight='bold', fontsize=13, y=1.03)
    plt.tight_layout()
    save_fig(fig, 'fig4_reliability')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Data directory: {BASE}')
    print(f'Output directory: {FIG_DIR}\n')
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print(f'\nAll 4 figures saved to {FIG_DIR}/')
