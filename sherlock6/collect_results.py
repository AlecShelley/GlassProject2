#!/usr/bin/env python3
"""
Aggregate async FIRE + LAMMPS results and generate publication figures.

Usage:
    python collect_results.py --resultsdir results/ --outdir figures/

Produces:
    1. K-D sweep plots: E/N vs K for each (N, phi) with D-curves + LAMMPS baselines
    2. Convergence heatmaps: final energy as function of (K, D)
    3. Timing comparison: wall time vs K for each D
    4. Summary CSV for tables
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json, os, re, csv, glob

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
})

K_VALUES = [1, 2, 4, 8, 16, 32]
D_VALUES = [0, 1, 2, 3, 5, 8]

D_COLORS = {
    0: '#1f77b4', 1: '#aec7e8', 2: '#ff7f0e',
    3: '#ffbb78', 5: '#2ca02c', 8: '#d62728',
}
D_MARKERS = {0: 'o', 1: 'p', 2: 's', 3: 'P', 5: '^', 8: 'v'}


def load_fire_results(resultsdir):
    """Load all async FIRE JSON results into a dict keyed by (N, phi, K, D)."""
    data = {}
    for fpath in glob.glob(os.path.join(resultsdir, 'fire_N*.json')):
        with open(fpath) as f:
            r = json.load(f)
        key = (r['N'], r['phi'], r['K'], r['D'])
        data[key] = r
    return data


def load_lammps_results(resultsdir):
    """Load LAMMPS log files, return dict keyed by (N, phi, method)."""
    data = {}
    for fpath in glob.glob(os.path.join(resultsdir, 'lammps_*.log')):
        fname = os.path.basename(fpath)
        m = re.match(r'lammps_(bitzek|default)_N(\d+)_phi(\d+)\.log', fname)
        if not m:
            continue
        method = m.group(1)
        N = int(m.group(2))
        phi_str = m.group(3)
        phi = float(phi_str[0] + '.' + phi_str[1:])

        energy = None
        with open(fpath) as f:
            for line in f:
                em = re.search(r'FINAL_ENERGY:\s+([\d.eE+-]+)', line)
                if em:
                    energy = float(em.group(1))

        if energy is not None:
            data[(N, phi, method)] = energy
    return data


def get_n_phi_combos(fire_data):
    """Extract unique (N, phi) combinations from fire data."""
    combos = sorted(set((k[0], k[1]) for k in fire_data))
    return combos


def plot_kd_sweep(fire_data, lammps_data, N, phi, outdir):
    """Plot E/N vs K for different D values, with LAMMPS baselines."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    e_bitzek = lammps_data.get((N, phi, 'bitzek'))
    e_default = lammps_data.get((N, phi, 'default'))

    ref = max(v for v in (e_bitzek, e_default) if v is not None) if (e_bitzek or e_default) else 100
    e_cutoff = max(ref * 3, 1.0)

    for D in D_VALUES:
        k_vals, e_vals = [], []
        for K in K_VALUES:
            r = fire_data.get((N, phi, K, D))
            if r and r['energy_per_n_lj'] < e_cutoff and r['status'] != 'diverged':
                k_vals.append(K)
                e_vals.append(r['energy_per_n_lj'])

        if k_vals:
            ax.plot(k_vals, e_vals, f'{D_MARKERS[D]}-', color=D_COLORS[D],
                    markersize=8, linewidth=2, label=f'Async FIRE $D={D}$', zorder=3)

    if e_bitzek is not None:
        ax.axhline(y=e_bitzek, color='#2196F3', linestyle='--', linewidth=1.5,
                   label='LAMMPS FIRE 1.0 (Bitzek)', alpha=0.8, zorder=2)
    if e_default is not None:
        ax.axhline(y=e_default, color='#F44336', linestyle=':', linewidth=1.5,
                   label='LAMMPS FIRE 2.0 (Guénolé)', alpha=0.8, zorder=2)

    ax.set_xlabel('Grid divisions $K$', fontsize=13)
    ax.set_ylabel('Final energy per particle $E/N$', fontsize=13)
    n_exp = int(np.log10(N))
    ax.set_title(f'Bidisperse 50:50, $N = 10^{n_exp}$, $\\phi = {phi}$\n'
                 f'Async FIRE: effect of $K$ and $D$',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

    stem = f'kd_sweep_N{N}_phi{phi:.2f}'
    fig.savefig(os.path.join(outdir, f'{stem}.png'))
    fig.savefig(os.path.join(outdir, f'{stem}.pdf'))
    plt.close()
    print(f"  {stem}")


def plot_heatmap(fire_data, N, phi, outdir):
    """Heatmap of final energy as function of (K, D)."""
    mat = np.full((len(D_VALUES), len(K_VALUES)), np.nan)

    for di, D in enumerate(D_VALUES):
        for ki, K in enumerate(K_VALUES):
            r = fire_data.get((N, phi, K, D))
            if r:
                e = r['energy_per_n_lj']
                if r['status'] == 'diverged':
                    mat[di, ki] = np.nan
                else:
                    mat[di, ki] = np.log10(max(e, 1e-15))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', origin='lower')

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES)
    ax.set_yticks(range(len(D_VALUES)))
    ax.set_yticklabels(D_VALUES)
    ax.set_xlabel('Grid divisions $K$', fontsize=13)
    ax.set_ylabel('Delayed reset $D$', fontsize=13)

    cbar = fig.colorbar(im, ax=ax, label='$\\log_{10}(E/N)$')

    for di in range(len(D_VALUES)):
        for ki in range(len(K_VALUES)):
            r = fire_data.get((N, phi, K_VALUES[ki], D_VALUES[di]))
            if r:
                if r['status'] == 'diverged':
                    ax.text(ki, di, '×', ha='center', va='center',
                            fontsize=14, color='red', fontweight='bold')
                elif r['status'] == 'stagnated':
                    ax.text(ki, di, 'S', ha='center', va='center',
                            fontsize=8, color='gray')

    n_exp = int(np.log10(N))
    ax.set_title(f'Convergence quality: $N=10^{n_exp}$, $\\phi={phi}$\n'
                 f'× = diverged, S = stagnated',
                 fontsize=14, fontweight='bold')

    stem = f'heatmap_N{N}_phi{phi:.2f}'
    fig.savefig(os.path.join(outdir, f'{stem}.png'))
    fig.savefig(os.path.join(outdir, f'{stem}.pdf'))
    plt.close()
    print(f"  {stem}")


def plot_timing(fire_data, N, phi, outdir):
    """Wall time vs K for different D values."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for D in D_VALUES:
        k_vals, t_vals = [], []
        for K in K_VALUES:
            r = fire_data.get((N, phi, K, D))
            if r and r['status'] != 'diverged':
                k_vals.append(K)
                t_vals.append(r['time_s'] / 3600)

        if k_vals:
            ax.plot(k_vals, t_vals, f'{D_MARKERS[D]}-', color=D_COLORS[D],
                    markersize=8, linewidth=2, label=f'$D={D}$', zorder=3)

    ax.set_xlabel('Grid divisions $K$', fontsize=13)
    ax.set_ylabel('Wall time (hours)', fontsize=13)
    n_exp = int(np.log10(N))
    ax.set_title(f'Timing: $N=10^{n_exp}$, $\\phi={phi}$',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

    stem = f'timing_N{N}_phi{phi:.2f}'
    fig.savefig(os.path.join(outdir, f'{stem}.png'))
    fig.savefig(os.path.join(outdir, f'{stem}.pdf'))
    plt.close()
    print(f"  {stem}")


def write_csv(fire_data, lammps_data, outdir):
    """Write summary CSV for easy table generation."""
    csvpath = os.path.join(outdir, 'summary.csv')
    with open(csvpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'phi', 'K', 'D', 'E_per_N_LJ', 'f_rms', 'fmax',
                     'steps', 'time_s', 'status',
                     'lammps_bitzek', 'lammps_default'])
        for key in sorted(fire_data.keys()):
            r = fire_data[key]
            N, phi, K, D = key
            eb = lammps_data.get((N, phi, 'bitzek'), '')
            ed = lammps_data.get((N, phi, 'default'), '')
            w.writerow([
                N, phi, K, D,
                f"{r['energy_per_n_lj']:.10e}",
                f"{r['f_rms']:.6e}",
                f"{r['fmax']:.6e}",
                r['steps'], f"{r['time_s']:.1f}", r['status'],
                eb, ed,
            ])
    print(f"  summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resultsdir', default='results/')
    ap.add_argument('--outdir', default='figures/')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading results...")
    fire_data = load_fire_results(args.resultsdir)
    lammps_data = load_lammps_results(args.resultsdir)
    print(f"  {len(fire_data)} async FIRE results")
    print(f"  {len(lammps_data)} LAMMPS baselines")

    if not fire_data:
        print("No results found. Check --resultsdir path.")
        return

    combos = get_n_phi_combos(fire_data)
    print(f"  {len(combos)} (N, phi) combinations\n")

    print("Generating K-D sweep plots...")
    for N, phi in combos:
        plot_kd_sweep(fire_data, lammps_data, N, phi, args.outdir)

    print("\nGenerating heatmaps...")
    for N, phi in combos:
        plot_heatmap(fire_data, N, phi, args.outdir)

    print("\nGenerating timing plots...")
    for N, phi in combos:
        plot_timing(fire_data, N, phi, args.outdir)

    print("\nWriting CSV...")
    write_csv(fire_data, lammps_data, args.outdir)

    print(f"\nDone! All figures in {args.outdir}")


if __name__ == '__main__':
    main()
