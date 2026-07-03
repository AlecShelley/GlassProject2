#!/usr/bin/env python3
"""
Generate all parameter files, LAMMPS inputs, and submission scripts
for a comprehensive K-D sweep on Sherlock.

Usage:
    python gen_sweep.py            # uses defaults
    python gen_sweep.py --dry-run  # preview without writing

Produces:
    params_fire_N{N}.txt          — one line per async FIRE job
    params_lammps.txt             — one line per LAMMPS baseline job
    lammps_inputs/                — LAMMPS .in files + table
    submit_all.sh                 — master submission script
"""
import argparse
import numpy as np
import os

# ── Sweep parameters ────────────────────────────────────────────────
# Adjust these for your publication needs

N_VALUES   = [1_000_000, 10_000_000]
PHI_VALUES = [0.82, 0.84, 0.86, 0.88, 0.90]
K_VALUES   = [1, 2, 4, 8, 16, 32]
D_VALUES   = [0, 1, 2, 3, 5, 8]

# ── Physics ─────────────────────────────────────────────────────────
K_SPRING   = 5000.0
R_SMALL    = 0.5
R_LARGE    = 0.7
SIGMA_SS   = 2 * R_SMALL   # 1.0
SIGMA_SL   = R_SMALL + R_LARGE  # 1.2
SIGMA_LL   = 2 * R_LARGE   # 1.4

# ── SLURM resource profiles per N ──────────────────────────────────
RESOURCE_PROFILES = {
    1_000_000:  {'cpus': 16, 'mem': '32G',  'time': '24:00:00'},
    10_000_000: {'cpus': 20, 'mem': '64G',  'time': '48:00:00'},
}


def box_size(N, phi):
    return np.sqrt(N * np.pi * (R_SMALL**2 + R_LARGE**2) / (2 * phi))


def gen_table(outdir):
    """Generate tabulated harmonic repulsive potential for LAMMPS."""
    filepath = os.path.join(outdir, 'harmonic_repulsive_bidisperse.table')
    n_pts = 10000
    r_max = SIGMA_LL

    with open(filepath, 'w') as f:
        f.write(f"# Bidisperse harmonic repulsive: V = 0.5*k*(sigma - r)^2\n")
        f.write(f"# k = {K_SPRING}, 50:50 ratio 1.4:1\n")
        f.write(f"# Sections: SS (sigma=1.0), SL (sigma=1.2), LL (sigma=1.4)\n\n")

        for name, sigma in [('HARM_SS', SIGMA_SS), ('HARM_SL', SIGMA_SL),
                            ('HARM_LL', SIGMA_LL)]:
            r_min = r_max / n_pts
            f.write(f"{name}\n")
            f.write(f"N {n_pts} R {r_min:.10f} {r_max:.10f}\n\n")

            for i in range(1, n_pts + 1):
                r = r_min + (r_max - r_min) * (i - 1) / (n_pts - 1)
                if r < sigma:
                    E = 0.5 * K_SPRING * (sigma - r)**2
                    F = K_SPRING * (sigma - r)
                else:
                    E = 0.0
                    F = 0.0
                f.write(f"{i} {r:.10f} {E:.10f} {F:.10f}\n")
            f.write("\n")

    return filepath


def gen_lammps_input(N, phi, method, outdir):
    """Generate a single LAMMPS input file."""
    L = box_size(N, phi)
    tag = f"{phi:.2f}".replace('.', '')
    fname = f"lammps_{method}_N{N}_phi{tag}.in"

    label = 'FIRE 1.0 (Bitzek 2006)' if method == 'bitzek' else 'FIRE 2.0 (Guenole 2020)'

    with open(os.path.join(outdir, fname), 'w') as f:
        f.write(f"# {label} — Bidisperse N={N}, phi={phi:.3f}\n\n")
        f.write("dimension       2\n")
        f.write("units           lj\n")
        f.write("atom_style      atomic\n")
        f.write("boundary        p p p\n\n")
        f.write(f"variable        L equal {L:.6f}\n")
        f.write("region          box block 0 $L 0 $L -0.5 0.5\n")
        f.write("create_box      2 box\n")
        f.write(f"create_atoms    1 random {N} 42 box\n\n")
        f.write("set             group all type/fraction 2 0.5 12345\n\n")
        f.write("mass            1 1.0\n")
        f.write("mass            2 1.0\n\n")
        f.write("pair_style      table linear 10000\n")
        f.write("pair_coeff      1 1 harmonic_repulsive_bidisperse.table HARM_SS 1.4\n")
        f.write("pair_coeff      1 2 harmonic_repulsive_bidisperse.table HARM_SL 1.4\n")
        f.write("pair_coeff      2 2 harmonic_repulsive_bidisperse.table HARM_LL 1.4\n\n")
        f.write("fix             f2d all enforce2d\n")
        f.write("timestep        0.001\n\n")
        f.write("thermo          1000\n")
        f.write("thermo_style    custom step pe press fmax cpu\n\n")
        f.write("min_style       fire\n")
        if method == 'bitzek':
            f.write("min_modify      dmax 100.0 delaystep 5 alpha0 0.1 "
                    "integrator verlet halfstepback no tmin 1e-6\n")
        f.write("\n")
        f.write("minimize        1.0e-12 1.0e-8 200000 1000000\n\n")
        f.write('print "FINAL_ENERGY: $(pe)"\n')
        f.write('print "FINAL_FMAX: $(fmax)"\n')

    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--outdir', default='.', help='Base output directory')
    args = ap.parse_args()

    base = args.outdir
    lammps_dir = os.path.join(base, 'lammps_inputs')

    if not args.dry_run:
        os.makedirs(lammps_dir, exist_ok=True)
        os.makedirs(os.path.join(base, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(base, 'results'), exist_ok=True)

    # ── 1. Async FIRE parameter files (one per N) ───────────────────
    for N in N_VALUES:
        pfile = os.path.join(base, f'params_fire_N{N}.txt')
        count = 0
        if not args.dry_run:
            with open(pfile, 'w') as f:
                for phi in PHI_VALUES:
                    for K in K_VALUES:
                        for D in D_VALUES:
                            f.write(f"{N} {phi} {K} {D}\n")
                            count += 1
        else:
            count = len(PHI_VALUES) * len(K_VALUES) * len(D_VALUES)
        print(f"Async FIRE N={N:>10,d}: {count} jobs → {pfile}")

    # ── 2. LAMMPS parameter file ────────────────────────────────────
    lammps_pfile = os.path.join(base, 'params_lammps.txt')
    lcount = 0
    if not args.dry_run:
        with open(lammps_pfile, 'w') as f:
            for N in N_VALUES:
                for phi in PHI_VALUES:
                    for method in ['bitzek', 'default']:
                        tag = f"{phi:.2f}".replace('.', '')
                        infile = f"lammps_{method}_N{N}_phi{tag}.in"
                        logfile = f"lammps_{method}_N{N}_phi{tag}.log"
                        ntasks = 32 if N >= 10_000_000 else 16
                        f.write(f"{infile} {logfile} {ntasks}\n")
                        lcount += 1
    else:
        lcount = len(N_VALUES) * len(PHI_VALUES) * 2
    print(f"LAMMPS baselines:         {lcount} jobs → {lammps_pfile}")

    # ── 3. Generate table + LAMMPS input files ──────────────────────
    if not args.dry_run:
        gen_table(lammps_dir)
        print(f"Table file → {lammps_dir}/harmonic_repulsive_bidisperse.table")
        for N in N_VALUES:
            for phi in PHI_VALUES:
                for method in ['bitzek', 'default']:
                    fname = gen_lammps_input(N, phi, method, lammps_dir)
        print(f"LAMMPS inputs → {lammps_dir}/ ({lcount} files)")

    # ── 4. Master submission script ─────────────────────────────────
    submit_path = os.path.join(base, 'submit_all.sh')
    if not args.dry_run:
        with open(submit_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Master submission script — run from the sweep directory\n")
            f.write("# Submits all async FIRE and LAMMPS jobs as SLURM arrays\n\n")
            f.write("set -e\n\n")

            for N in N_VALUES:
                n_jobs = len(PHI_VALUES) * len(K_VALUES) * len(D_VALUES)
                res = RESOURCE_PROFILES.get(N, {'cpus': 16, 'mem': '32G', 'time': '48:00:00'})
                f.write(f"# ── Async FIRE N={N:,d} ({n_jobs} jobs) ──\n")
                f.write(f"FIRE_N{N}=$(sbatch \\\n")
                f.write(f"    --array=1-{n_jobs} \\\n")
                f.write(f"    --cpus-per-task={res['cpus']} \\\n")
                f.write(f"    --mem={res['mem']} \\\n")
                f.write(f"    --time={res['time']} \\\n")
                f.write(f"    --export=PARAM_FILE=params_fire_N{N}.txt \\\n")
                f.write(f"    submit_fire.sbatch | awk '{{print $NF}}')\n")
                f.write(f'echo "Submitted async FIRE N={N:,d}: $FIRE_N{N}"\n\n')

            f.write(f"# ── LAMMPS baselines ({lcount} jobs) ──\n")
            f.write(f"LAMMPS=$(sbatch \\\n")
            f.write(f"    --array=1-{lcount} \\\n")
            f.write(f"    submit_lammps.sbatch | awk '{{print $NF}}')\n")
            f.write(f'echo "Submitted LAMMPS baselines: $LAMMPS"\n\n')

            f.write('echo "All jobs submitted. Monitor with: squeue -u $USER"\n')

        os.chmod(submit_path, 0o755)
        print(f"Submission script → {submit_path}")

    # ── Summary ─────────────────────────────────────────────────────
    total_fire = sum(len(PHI_VALUES) * len(K_VALUES) * len(D_VALUES) for _ in N_VALUES)
    total = total_fire + lcount
    print(f"\nTotal: {total_fire} async FIRE + {lcount} LAMMPS = {total} jobs")
    print(f"System sizes: {[f'{n:,}' for n in N_VALUES]}")
    print(f"Phi values:   {PHI_VALUES}")
    print(f"K values:     {K_VALUES}")
    print(f"D values:     {D_VALUES}")

    if args.dry_run:
        print("\n(dry run — no files written)")


if __name__ == '__main__':
    main()
