#!/usr/bin/env python3
"""
Comparison Run — FIRE Parameter Sweep Compute Script for Sherlock HPC
Converted from comparison_run.ipynb

Concurrency is configurable:
  python comparison_run_compute.py --cpus 16
  python comparison_run_compute.py --cpus 8 --n-atoms 1000000 --wall-time 120

NUMBA_NUM_THREADS is set from --cpus before Numba is imported.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="FIRE Parameter Sweep — Comparison Run")
parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="Number of Numba parallel threads (default: SLURM_CPUS_PER_TASK or 1)")
parser.add_argument("--n-atoms", type=int, default=1000000, help="Number of atoms (default: 1000000)")
parser.add_argument("--wall-time", type=float, default=600.0, help="Max wall time per run in seconds (default: 600)")
parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "parameter_sweep"),
                    help="Output directory for .npz files")
parser.add_argument("--delayed-reset", type=int, default=0, metavar="D",
                    help="Require D consecutive negative-power steps before domain reset (0=immediate, default: 0)")
parser.add_argument("--jitter", action="store_true",
                    help="Randomly jitter domain grid each step to break persistent boundary effects")
parser.add_argument("--tol", type=float, default=1e-4,
                    help="Force-norm convergence tolerance (default: 1e-4, use 1e-8 for equal-time experiments)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for initial configuration (default: 42)")
args = parser.parse_args()

os.environ["NUMBA_NUM_THREADS"] = str(args.cpus)

import numpy as np
import time
from numba import njit, prange

# ==========================================
# 1. SYSTEM PARAMETERS & SWEEP TARGETS
# ==========================================
N_ATOMS = args.n_atoms
BOX_SIZE = 1.0
K_SPRING = 5000.0
MASS = 1.0

DT_INIT = 0.001
DT_MAX = 0.01
N_MIN = 5
F_INC = 1.1
F_DEC = 0.5
ALPHA_START = 0.1
F_ALPHA = 0.99
TOL = args.tol

MAX_STEPS = 100000
CAPTURE_STEP = 150
MAX_WALL_TIME = args.wall_time

PHI_SWEEP = [0.84, 0.90]  # focused equal-time experiment (restore to [0.80, 0.82, 0.84, 0.86, 0.90] for full sweep)
GAMMA_TARGETS = [0.03, 0.06, 0.09, 0.12, 0.15] 


# ==========================================
# 2. NUMBA JIT-COMPILED CELL LIST PHYSICS
# ==========================================
@njit
def build_cell_list(pos, box_size, cutoff):
    n_atoms = pos.shape[0]
    n_cells = int(np.floor(box_size / cutoff))

    if n_cells < 3:
        n_cells = 3
    cell_size = box_size / n_cells

    head = np.full((n_cells, n_cells), -1, dtype=np.int32)
    next_p = np.full(n_atoms, -1, dtype=np.int32)

    for i in range(n_atoms):
        cx = int(pos[i, 0] / cell_size)
        cy = int(pos[i, 1] / cell_size)
        cx = max(0, min(cx, n_cells - 1))
        cy = max(0, min(cy, n_cells - 1))

        next_p[i] = head[cx, cy]
        head[cx, cy] = i

    return head, next_p, n_cells


@njit(parallel=True)
def get_forces_global_numba(pos, radius, k_spring, box_size):
    n_atoms = pos.shape[0]
    forces = np.zeros((n_atoms, 2))

    cutoff = 2.0 * radius
    head, next_p, n_cells = build_cell_list(pos, box_size, cutoff)
    cell_size = box_size / n_cells

    for i in prange(n_atoms):
        cx = int(pos[i, 0] / cell_size)
        cy = int(pos[i, 1] / cell_size)
        cx = max(0, min(cx, n_cells - 1))
        cy = max(0, min(cy, n_cells - 1))

        fx_i = 0.0
        fy_i = 0.0

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx_cell = (cx + dx) % n_cells
                ny_cell = (cy + dy) % n_cells

                j = head[nx_cell, ny_cell]
                while j != -1:
                    if i != j:
                        dx_vec = pos[i, 0] - pos[j, 0]
                        dy_vec = pos[i, 1] - pos[j, 1]

                        dx_vec = dx_vec - box_size * np.round(dx_vec / box_size)
                        dy_vec = dy_vec - box_size * np.round(dy_vec / box_size)

                        dist_sq = dx_vec**2 + dy_vec**2

                        if dist_sq < cutoff**2 and dist_sq > 1e-12:
                            dist = np.sqrt(dist_sq)
                            f_mag = k_spring * (cutoff - dist)
                            fx_i += (dx_vec / dist) * f_mag
                            fy_i += (dy_vec / dist) * f_mag

                    j = next_p[j]

        forces[i, 0] = fx_i
        forces[i, 1] = fy_i

    return forces


@njit(parallel=True)
def get_total_energy_cell_list_numba(pos, radius, k_spring, box_size):
    n_atoms = pos.shape[0]
    cutoff = 2.0 * radius
    head, next_p, n_cells = build_cell_list(pos, box_size, cutoff)
    cell_size = box_size / n_cells

    energy = 0.0
    for i in prange(n_atoms):
        cx = int(pos[i, 0] / cell_size)
        cy = int(pos[i, 1] / cell_size)
        cx = max(0, min(cx, n_cells - 1))
        cy = max(0, min(cy, n_cells - 1))

        local_energy = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx_cell = (cx + dx) % n_cells
                ny_cell = (cy + dy) % n_cells

                j = head[nx_cell, ny_cell]
                while j != -1:
                    if j > i:
                        dx_vec = pos[i, 0] - pos[j, 0]
                        dy_vec = pos[i, 1] - pos[j, 1]

                        dx_vec = dx_vec - box_size * np.round(dx_vec / box_size)
                        dy_vec = dy_vec - box_size * np.round(dy_vec / box_size)

                        dist_sq = dx_vec**2 + dy_vec**2
                        if dist_sq < cutoff**2:
                            dist = np.sqrt(dist_sq)
                            overlap = cutoff - dist
                            local_energy += 0.5 * k_spring * (overlap ** 2)
                    j = next_p[j]

        energy += local_energy

    return energy


def map_atoms_to_grid(pos, divs):
    ix = np.clip(np.floor(pos[:, 0] * divs).astype(int), 0, divs - 1)
    iy = np.clip(np.floor(pos[:, 1] * divs).astype(int), 0, divs - 1)
    return ix * divs + iy


@njit
def sync_domain_dt(d_dt, grid_divs):
    synced = np.empty_like(d_dt)
    for gx in range(grid_divs):
        for gy in range(grid_divs):
            min_dt = d_dt[gx * grid_divs + gy]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx = (gx + dx) % grid_divs
                    ny = (gy + dy) % grid_divs
                    min_dt = min(min_dt, d_dt[nx * grid_divs + ny])
            synced[gx * grid_divs + gy] = min_dt
    return synced


# ==========================================
# 3. TIMED FIRE ENGINES
# ==========================================
def run_global_fire(pos_init, max_steps, radius):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    dt = DT_INIT
    alpha = ALPHA_START
    npos = 0

    energy_history = []
    dt_history = []
    time_history = []

    forces = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)

    t0 = time.time()
    for step in range(max_steps):
        if step % 10 == 0:
            elapsed = time.time() - t0
            e_curr = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_history.append(e_curr)
            dt_history.append(dt)
            time_history.append(elapsed)

            if elapsed > MAX_WALL_TIME:
                break

        vel += 0.5 * forces * dt / MASS
        pos += vel * dt
        pos = np.mod(pos, BOX_SIZE)

        forces_new = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)
        vel += 0.5 * forces_new * dt / MASS
        forces = forces_new

        P_global = np.sum(forces * vel)
        v_norm_global = np.sqrt(np.sum(vel ** 2))
        f_norm_global = np.sqrt(np.sum(forces ** 2))

        if f_norm_global > 1e-12:
            vel = (1.0 - alpha) * vel + alpha * (forces / f_norm_global) * v_norm_global

        if P_global > 0:
            npos += 1
            if npos > N_MIN:
                dt = min(dt * F_INC, DT_MAX)
                alpha *= F_ALPHA
        else:
            npos = 0
            dt *= F_DEC
            vel[:, :] = 0.0
            alpha = ALPHA_START

        if np.sqrt(np.sum(forces ** 2) / (2 * N_ATOMS)) < TOL:
            break

    return pos, np.array(energy_history), np.array(dt_history), np.array(time_history), time.time() - t0


def run_async_fire(pos_init, max_steps, capture_step, radius, grid_divs,
                   delayed_reset=0, use_jitter=False):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    n_domains = grid_divs * grid_divs

    d_dt = np.full(n_domains, DT_INIT)
    d_alpha = np.full(n_domains, ALPHA_START)
    d_npos = np.zeros(n_domains, dtype=int)
    d_neg = np.zeros(n_domains, dtype=int)

    energy_history = []
    dt_mean_history = []
    time_history = []
    snapshot_dt = None
    accumulated_dt = np.zeros(n_domains)

    cell_size = BOX_SIZE / grid_divs

    forces = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)

    t0 = time.time()
    for step in range(max_steps):
        accumulated_dt += d_dt

        if step % 10 == 0:
            elapsed = time.time() - t0
            e_current = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_history.append(e_current)
            dt_mean_history.append(np.mean(d_dt))
            time_history.append(elapsed)

            if elapsed > MAX_WALL_TIME:
                if snapshot_dt is None:
                    snapshot_dt = np.copy(d_dt)
                break

        if step == capture_step:
            snapshot_dt = np.copy(d_dt)

        if use_jitter:
            jitter = np.random.rand(2) * cell_size
            atom_indices = map_atoms_to_grid(np.mod(pos + jitter, BOX_SIZE), grid_divs)
        else:
            atom_indices = map_atoms_to_grid(pos, grid_divs)

        synced_dt = sync_domain_dt(d_dt, grid_divs)
        atom_dt = synced_dt[atom_indices][:, np.newaxis]

        vel += 0.5 * forces * atom_dt / MASS
        pos += vel * atom_dt
        pos = np.mod(pos, BOX_SIZE)

        forces_new = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)
        vel += 0.5 * forces_new * atom_dt / MASS
        forces = forces_new

        # --- Per-atom FIRE steering (Bitzek 2006) ---
        P_atom = np.sum(forces * vel, axis=1)

        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        f_mag = np.linalg.norm(forces, axis=1, keepdims=True)
        atom_alpha = d_alpha[atom_indices][:, np.newaxis]

        mask_f = (f_mag > 1e-12).flatten()
        if np.any(mask_f):
            vel[mask_f] = (1 - atom_alpha[mask_f]) * vel[mask_f] + atom_alpha[mask_f] * (forces[mask_f] / f_mag[mask_f]) * v_mag[mask_f]

        # --- Domain-level dt adaptation (async clocks) ---
        p_domain = np.bincount(atom_indices, weights=P_atom, minlength=n_domains)

        mask_up = p_domain > 0
        d_npos[mask_up] += 1
        d_neg[mask_up] = 0
        mask_grow = mask_up & (d_npos > N_MIN)
        d_dt[mask_grow] = np.minimum(d_dt[mask_grow] * F_INC, DT_MAX)
        d_alpha[mask_grow] *= F_ALPHA

        mask_down = p_domain <= 0
        d_neg[mask_down] += 1
        d_npos[mask_down] = 0

        if delayed_reset > 0:
            mask_reset = d_neg >= delayed_reset
        else:
            mask_reset = mask_down

        d_dt[mask_reset] *= F_DEC
        d_alpha[mask_reset] = ALPHA_START
        vel[mask_reset[atom_indices]] = 0.0
        d_neg[mask_reset] = 0

        if np.sqrt(np.sum(forces ** 2) / (2 * N_ATOMS)) < TOL:
            if snapshot_dt is None:
                snapshot_dt = np.copy(d_dt)
            break

    if snapshot_dt is None:
        snapshot_dt = np.copy(d_dt)

    return pos, snapshot_dt, np.array(energy_history), np.array(dt_mean_history), np.array(time_history), time.time() - t0, accumulated_dt


# ==========================================
# 4. AUTOMATED PARAMETER SPACE RUNNER
# ==========================================
if __name__ == "__main__":
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(args.seed)
    base_positions = np.random.rand(N_ATOMS, 2)

    print(f"Comparison Run — FIRE Parameter Sweep")
    print(f"  N_ATOMS:        {N_ATOMS:,}")
    print(f"  Numba threads:  {args.cpus}")
    print(f"  Wall time/run:  {MAX_WALL_TIME}s")
    print(f"  Delayed reset:  {args.delayed_reset if args.delayed_reset > 0 else 'off (immediate)'}")
    print(f"  Grid jitter:    {'on' if args.jitter else 'off'}")
    print(f"  Tolerance:      {TOL:.0e}")
    print(f"  Random seed:    {args.seed}")
    print(f"  Output dir:     {output_dir}")
    print(f"  SLURM Job ID:   {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"  Node:           {os.environ.get('SLURM_NODELIST', 'local')}")
    print()

    print("Warming up Numba JIT compiler...")
    dummy_r = np.sqrt((0.82 * BOX_SIZE ** 2) / (10 * np.pi))
    dummy_pos = base_positions[:10]
    _ = get_forces_global_numba(dummy_pos, dummy_r, K_SPRING, BOX_SIZE)
    _ = get_total_energy_cell_list_numba(dummy_pos, dummy_r, K_SPRING, BOX_SIZE)
    print("JIT warmup complete.\n")

    print(f"Starting 25-run parameter sweep...")
    print(f"Densities (phi): {PHI_SWEEP}")
    print(f"Gamma targets:   {GAMMA_TARGETS}")
    print("=" * 70)

    sweep_t0 = time.time()

    for phi_idx, phi in enumerate(PHI_SWEEP):
        current_radius = np.sqrt((phi * BOX_SIZE ** 2) / (N_ATOMS * np.pi))

        k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in GAMMA_TARGETS]
        k_values = sorted(list(set(k_values)))

        while len(k_values) < len(GAMMA_TARGETS):
            k_values.append(k_values[-1] + 2)
        k_values = k_values[:len(GAMMA_TARGETS)]

        print(f"\n[DENSITY PHASE {phi_idx + 1}/5] phi = {phi:.2f} (Radius: {current_radius:.6f})")
        print(f"  Grid sizes (K): {k_values}")
        print("  Running Global FIRE baseline...")

        pos_glob, e_hist_glob, dt_hist_glob, t_hist_glob, time_glob = run_global_fire(
            base_positions, MAX_STEPS, current_radius
        )
        print(f"  Global FIRE done in {time_glob:.1f}s, final energy: {e_hist_glob[-1]:.4e}")

        for grid_idx, grid in enumerate(k_values):
            actual_gamma = (8 * current_radius * grid) / BOX_SIZE
            print(f"  -> Async FIRE {grid}x{grid} grid (gamma={actual_gamma:.3f})...", end=" ", flush=True)

            pos_async, dt_grid_snap, e_hist_async, dt_hist_async, t_hist_async, time_async, accumulated_dt = run_async_fire(
                base_positions, MAX_STEPS, CAPTURE_STEP, current_radius, grid,
                delayed_reset=args.delayed_reset, use_jitter=args.jitter
            )

            out_filename = os.path.join(output_dir, f"sweep_phi{phi:.2f}_grid{grid}.npz")
            np.savez_compressed(
                out_filename,
                pos_async=pos_async, pos_glob=pos_glob, dt_grid_snap=dt_grid_snap,
                e_hist_async=e_hist_async, e_hist_glob=e_hist_glob,
                dt_hist_async=dt_hist_async, dt_hist_glob=dt_hist_glob,
                t_hist_async=t_hist_async, t_hist_glob=t_hist_glob,
                time_async=time_async, time_glob=time_glob,
                accumulated_dt=accumulated_dt, radius=current_radius,
                k_spring=K_SPRING, grid_divs=grid, n_atoms=N_ATOMS,
                box_size=BOX_SIZE, phi=phi, gamma=actual_gamma, seed=args.seed,
            )

            print(f"done in {time_async:.1f}s | E_glob={e_hist_glob[-1]:.4e} E_async={e_hist_async[-1]:.4e}")

    total_time = time.time() - sweep_t0
    print("\n" + "=" * 70)
    print(f"Parameter sweep complete! Total wall time: {total_time:.1f}s ({total_time / 3600:.2f} hours)")
    print(f"Results saved to: {output_dir}/")
