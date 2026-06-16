#!/usr/bin/env python3
"""
Reset Frequency Analysis — Why Async FIRE Outperforms Global FIRE Above Jamming

Tracks per-step reset statistics for both methods:
  - Global FIRE: how often the single global P < 0 triggers a full system reset
  - Async FIRE:  what fraction of domains reset each step

This quantifies the "slowest boat in the convoy" problem: at high density,
Global resets on nearly every step, destroying productive motion in relaxed
regions. Async limits damage to the troubled domains.

Usage:
  python reset_analysis.py --cpus 6 --delayed-reset 3
  python reset_analysis.py --cpus 6 --phi 0.85 0.86 0.87 0.88 0.89 0.90
  python reset_analysis.py --cpus 6 --phi 0.88 --grids 7 --wall-time 120

NUMBA_NUM_THREADS is set from --cpus before Numba is imported.
"""

import argparse
import os
import json

parser = argparse.ArgumentParser(description="Reset Frequency Analysis — Global vs Async FIRE")
parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="Number of Numba parallel threads (default: SLURM_CPUS_PER_TASK or 1)")
parser.add_argument("--n-atoms", type=int, default=1000000, help="Number of atoms (default: 1000000)")
parser.add_argument("--wall-time", type=float, default=300.0, help="Max wall time per run in seconds (default: 300)")
parser.add_argument("--phi", type=float, nargs="+", default=[0.85, 0.86, 0.87, 0.88, 0.89, 0.90],
                    help="Packing fractions to test")
parser.add_argument("--grids", type=int, nargs="+", default=None,
                    help="Grid sizes K for async FIRE (default: [7])")
parser.add_argument("--delayed-reset", type=int, default=3, metavar="D",
                    help="Consecutive negative-power steps before domain reset (default: 3)")
parser.add_argument("--output-dir", type=str, default="reset_analysis",
                    help="Output directory (default: reset_analysis)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="Log diagnostics every N steps (default: 10)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for initial configuration (default: 42)")
args = parser.parse_args()

os.environ["NUMBA_NUM_THREADS"] = str(args.cpus)

import numpy as np
import time
from numba import njit, prange

# ==========================================
# SYSTEM PARAMETERS
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

MAX_STEPS = 200000
MAX_WALL_TIME = args.wall_time
LOG_INTERVAL = args.log_interval


# ==========================================
# NUMBA KERNELS
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
# GLOBAL FIRE WITH RESET TRACKING
# ==========================================
def run_global_fire_reset_tracking(pos_init, radius):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    dt = DT_INIT
    alpha = ALPHA_START
    npos = 0

    time_trace = []
    energy_trace = []
    dt_trace = []
    reset_trace = []
    cumulative_resets = 0
    total_steps = 0

    forces = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)

    t0 = time.time()
    for step in range(MAX_STEPS):
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            e_curr = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_trace.append(e_curr)
            time_trace.append(elapsed)
            dt_trace.append(dt)
            reset_trace.append(cumulative_resets)

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
            cumulative_resets += 1

        total_steps = step + 1

    total_time = time.time() - t0
    reset_rate = cumulative_resets / total_steps if total_steps > 0 else 0.0

    return {
        "energy_trace": np.array(energy_trace),
        "time_trace": np.array(time_trace),
        "dt_trace": np.array(dt_trace),
        "reset_trace": np.array(reset_trace),
        "total_resets": cumulative_resets,
        "total_steps": total_steps,
        "reset_rate": reset_rate,
        "final_energy": energy_trace[-1] if energy_trace else np.inf,
        "total_time": total_time,
    }


# ==========================================
# ASYNC FIRE WITH RESET TRACKING
# ==========================================
def run_async_fire_reset_tracking(pos_init, radius, grid_divs, delayed_reset=3):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    n_domains = grid_divs * grid_divs

    d_dt = np.full(n_domains, DT_INIT)
    d_alpha = np.full(n_domains, ALPHA_START)
    d_npos = np.zeros(n_domains, dtype=int)
    d_neg = np.zeros(n_domains, dtype=int)

    time_trace = []
    energy_trace = []
    dt_mean_trace = []
    dt_min_trace = []
    dt_max_trace = []
    frac_reset_trace = []
    frac_growing_trace = []
    frac_at_dtmax_trace = []
    cumulative_domain_resets = 0
    total_steps = 0

    forces = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)

    t0 = time.time()
    for step in range(MAX_STEPS):
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            e_current = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_trace.append(e_current)
            time_trace.append(elapsed)
            dt_mean_trace.append(np.mean(d_dt))
            dt_min_trace.append(np.min(d_dt))
            dt_max_trace.append(np.max(d_dt))
            frac_at_dtmax_trace.append(np.mean(d_dt >= DT_MAX * 0.99))

            if elapsed > MAX_WALL_TIME:
                break

        atom_indices = map_atoms_to_grid(pos, grid_divs)
        synced_dt = sync_domain_dt(d_dt, grid_divs)
        atom_dt = synced_dt[atom_indices][:, np.newaxis]

        vel += 0.5 * forces * atom_dt / MASS
        pos += vel * atom_dt
        pos = np.mod(pos, BOX_SIZE)

        forces_new = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)
        vel += 0.5 * forces_new * atom_dt / MASS
        forces = forces_new

        P_atom = np.sum(forces * vel, axis=1)
        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        f_mag = np.linalg.norm(forces, axis=1, keepdims=True)
        atom_alpha = d_alpha[atom_indices][:, np.newaxis]

        mask_f = (f_mag > 1e-12).flatten()
        if np.any(mask_f):
            vel[mask_f] = (1 - atom_alpha[mask_f]) * vel[mask_f] + atom_alpha[mask_f] * (forces[mask_f] / f_mag[mask_f]) * v_mag[mask_f]

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

        n_reset_this_step = int(np.sum(mask_reset))
        cumulative_domain_resets += n_reset_this_step

        if step % LOG_INTERVAL == 0:
            frac_reset_trace.append(n_reset_this_step / n_domains)
            frac_growing_trace.append(int(np.sum(mask_grow)) / n_domains)

        d_dt[mask_reset] *= F_DEC
        d_alpha[mask_reset] = ALPHA_START
        vel[mask_reset[atom_indices]] = 0.0
        d_neg[mask_reset] = 0

        total_steps = step + 1

    total_time = time.time() - t0
    avg_frac_reset = cumulative_domain_resets / (total_steps * n_domains) if total_steps > 0 else 0.0

    return {
        "energy_trace": np.array(energy_trace),
        "time_trace": np.array(time_trace),
        "dt_mean_trace": np.array(dt_mean_trace),
        "dt_min_trace": np.array(dt_min_trace),
        "dt_max_trace": np.array(dt_max_trace),
        "frac_reset_trace": np.array(frac_reset_trace),
        "frac_growing_trace": np.array(frac_growing_trace),
        "frac_at_dtmax_trace": np.array(frac_at_dtmax_trace),
        "cumulative_domain_resets": cumulative_domain_resets,
        "total_steps": total_steps,
        "n_domains": n_domains,
        "avg_frac_reset": avg_frac_reset,
        "final_energy": energy_trace[-1] if energy_trace else np.inf,
        "total_time": total_time,
    }


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    grids = args.grids if args.grids else [7]

    np.random.seed(args.seed)
    base_positions = np.random.rand(N_ATOMS, 2)

    print("Reset Frequency Analysis — Global vs Async FIRE")
    print(f"  N_ATOMS:        {N_ATOMS:,}")
    print(f"  Numba threads:  {args.cpus}")
    print(f"  Wall time/run:  {MAX_WALL_TIME}s")
    print(f"  Delayed reset:  {args.delayed_reset}")
    print(f"  Log interval:   every {LOG_INTERVAL} step(s)")
    print(f"  Random seed:    {args.seed}")
    print(f"  Densities:      {args.phi}")
    print(f"  Grids:          {grids}")
    print(f"  Output dir:     {output_dir}")
    print()

    print("Warming up Numba JIT...")
    dummy_r = np.sqrt((0.84 * BOX_SIZE ** 2) / (10 * np.pi))
    dummy_pos = base_positions[:10]
    _ = get_forces_global_numba(dummy_pos, dummy_r, K_SPRING, BOX_SIZE)
    _ = get_total_energy_cell_list_numba(dummy_pos, dummy_r, K_SPRING, BOX_SIZE)
    print("JIT warmup complete.\n")

    all_results = {}

    for phi in args.phi:
        current_radius = np.sqrt((phi * BOX_SIZE ** 2) / (N_ATOMS * np.pi))

        print("=" * 70)
        print(f"phi = {phi:.2f} | radius = {current_radius:.6f}")
        print("=" * 70)

        # --- Global FIRE ---
        print(f"  Global FIRE ...", end=" ", flush=True)
        glob = run_global_fire_reset_tracking(base_positions, current_radius)
        print(f"done in {glob['total_time']:.1f}s | "
              f"E={glob['final_energy']:.4e} | "
              f"resets={glob['total_resets']}/{glob['total_steps']} "
              f"({glob['reset_rate']:.1%} of steps)")

        np.savez_compressed(
            os.path.join(output_dir, f"reset_phi{phi:.2f}_global.npz"),
            energy_trace=glob["energy_trace"],
            time_trace=glob["time_trace"],
            dt_trace=glob["dt_trace"],
            reset_trace=glob["reset_trace"],
            phi=phi, radius=current_radius, n_atoms=N_ATOMS,
            total_resets=glob["total_resets"],
            total_steps=glob["total_steps"],
            reset_rate=glob["reset_rate"], seed=args.seed,
        )

        phi_results = {
            "global": {
                "total_resets": glob["total_resets"],
                "total_steps": glob["total_steps"],
                "reset_rate": glob["reset_rate"],
                "final_energy": float(glob["final_energy"]),
                "total_time": glob["total_time"],
            }
        }

        # --- Async FIRE ---
        for grid in grids:
            actual_gamma = (8 * current_radius * grid) / BOX_SIZE
            print(f"  Async {grid}x{grid} (gamma={actual_gamma:.3f}) ...", end=" ", flush=True)

            async_res = run_async_fire_reset_tracking(
                base_positions, current_radius, grid,
                delayed_reset=args.delayed_reset,
            )

            print(f"done in {async_res['total_time']:.1f}s | "
                  f"E={async_res['final_energy']:.4e} | "
                  f"avg domain reset fraction={async_res['avg_frac_reset']:.1%} | "
                  f"domains at dt_max={np.mean(async_res['frac_at_dtmax_trace']):.1%}")

            np.savez_compressed(
                os.path.join(output_dir, f"reset_phi{phi:.2f}_async_K{grid}.npz"),
                energy_trace=async_res["energy_trace"],
                time_trace=async_res["time_trace"],
                dt_mean_trace=async_res["dt_mean_trace"],
                dt_min_trace=async_res["dt_min_trace"],
                dt_max_trace=async_res["dt_max_trace"],
                frac_reset_trace=async_res["frac_reset_trace"],
                frac_growing_trace=async_res["frac_growing_trace"],
                frac_at_dtmax_trace=async_res["frac_at_dtmax_trace"],
                phi=phi, radius=current_radius, n_atoms=N_ATOMS,
                grid_divs=grid, gamma=actual_gamma, seed=args.seed,
                cumulative_domain_resets=async_res["cumulative_domain_resets"],
                total_steps=async_res["total_steps"],
                n_domains=async_res["n_domains"],
                avg_frac_reset=async_res["avg_frac_reset"],
            )

            phi_results[f"async_K{grid}"] = {
                "cumulative_domain_resets": async_res["cumulative_domain_resets"],
                "total_steps": async_res["total_steps"],
                "n_domains": async_res["n_domains"],
                "avg_frac_reset": async_res["avg_frac_reset"],
                "avg_frac_at_dtmax": float(np.mean(async_res["frac_at_dtmax_trace"])),
                "final_energy": float(async_res["final_energy"]),
                "total_time": async_res["total_time"],
                "gamma": actual_gamma,
            }

        all_results[f"phi_{phi:.2f}"] = phi_results

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("SUMMARY — Reset Frequency Analysis")
    print("=" * 90)
    print(f"{'Method':<25} {'phi':>6} {'Reset Rate':>12} {'Atoms Reset':>14} {'Final E':>14} {'dt Status':>16}")
    print("-" * 90)

    for phi in args.phi:
        key = f"phi_{phi:.2f}"
        if key not in all_results:
            continue
        res = all_results[key]
        g = res["global"]

        print(f"{'Global FIRE':<25} {phi:>6.2f} {g['reset_rate']:>11.1%} {'100% (all)':>14} {g['final_energy']:>14.4e} {'single dt':>16}")

        for label, data in res.items():
            if label == "global":
                continue
            frac_str = f"{data['avg_frac_reset']:.1%}"
            dtmax_str = f"{data['avg_frac_at_dtmax']:.0%} at dt_max"
            print(f"  {label:<23} {phi:>6.2f} {frac_str:>12} {frac_str:>14} {data['final_energy']:>14.4e} {dtmax_str:>16}")

    # --- Comparison insight ---
    print("\n" + "-" * 90)
    print("KEY INSIGHT:")
    for phi in args.phi:
        key = f"phi_{phi:.2f}"
        if key not in all_results:
            continue
        res = all_results[key]
        g_rate = res["global"]["reset_rate"]
        for label, data in res.items():
            if label == "global":
                continue
            a_rate = data["avg_frac_reset"]
            wasted = g_rate - a_rate
            if wasted > 0:
                atoms_saved = int(wasted * N_ATOMS)
                print(f"  phi={phi:.2f}: Global resets {g_rate:.1%} of steps vs Async {a_rate:.1%} of domains.")
                print(f"           → {wasted:.1%} of resets are unnecessary = {atoms_saved:,} atoms/step having productive velocity destroyed.")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary JSON:     {summary_path}")
