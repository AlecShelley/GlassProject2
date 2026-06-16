#!/usr/bin/env python3
"""
Time-to-Target Experiment — Async FIRE vs Global FIRE Speed Comparison

Measures wall-clock time for each method to reach a target energy threshold.
Logs energy densely (every step) for high-resolution energy-vs-time curves.

Usage:
  python time_to_target.py --cpus 6 --delayed-reset 3
  python time_to_target.py --cpus 6 --energy-target 0.001 --phi 0.84 0.90
  python time_to_target.py --cpus 6 --grids 5 7 10 --phi 0.84

NUMBA_NUM_THREADS is set from --cpus before Numba is imported.
"""

import argparse
import os
import json

parser = argparse.ArgumentParser(description="Time-to-Target — Async vs Global FIRE")
parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="Number of Numba parallel threads (default: SLURM_CPUS_PER_TASK or 1)")
parser.add_argument("--n-atoms", type=int, default=1000000, help="Number of atoms (default: 1000000)")
parser.add_argument("--wall-time", type=float, default=120.0, help="Max wall time per run in seconds (default: 120)")
parser.add_argument("--energy-target", type=float, default=0.01,
                    help="Energy threshold to race toward (default: 0.01)")
parser.add_argument("--phi", type=float, nargs="+", default=[0.84, 0.90],
                    help="Packing fractions to test (default: 0.84 0.90)")
parser.add_argument("--grids", type=int, nargs="+", default=None,
                    help="Grid sizes K for async FIRE (default: auto from gamma targets)")
parser.add_argument("--delayed-reset", type=int, default=3, metavar="D",
                    help="Consecutive negative-power steps before domain reset (default: 3)")
parser.add_argument("--output-dir", type=str, default="time_to_target",
                    help="Output directory (default: time_to_target)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="Log energy every N steps (default: 1 for max resolution)")
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
ENERGY_TARGET = args.energy_target
LOG_INTERVAL = args.log_interval

GAMMA_TARGETS = [0.03, 0.06, 0.09, 0.12, 0.15]


# ==========================================
# NUMBA KERNELS (identical to comparison_run_compute.py)
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
# FIRE ENGINES WITH TIME-TO-TARGET TRACKING
# ==========================================
def run_global_fire_ttt(pos_init, radius, energy_target):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    dt = DT_INIT
    alpha = ALPHA_START
    npos = 0

    energy_trace = []
    time_trace = []
    time_to_target = None

    forces = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)

    t0 = time.time()
    for step in range(MAX_STEPS):
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            e_curr = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_trace.append(e_curr)
            time_trace.append(elapsed)

            if time_to_target is None and e_curr <= energy_target:
                time_to_target = elapsed

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

    total_time = time.time() - t0
    return {
        "energy_trace": np.array(energy_trace),
        "time_trace": np.array(time_trace),
        "time_to_target": time_to_target,
        "final_energy": energy_trace[-1] if energy_trace else np.inf,
        "total_time": total_time,
        "steps": step + 1,
    }


def run_async_fire_ttt(pos_init, radius, grid_divs, energy_target, delayed_reset=3):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    n_domains = grid_divs * grid_divs

    d_dt = np.full(n_domains, DT_INIT)
    d_alpha = np.full(n_domains, ALPHA_START)
    d_npos = np.zeros(n_domains, dtype=int)
    d_neg = np.zeros(n_domains, dtype=int)

    energy_trace = []
    time_trace = []
    time_to_target = None

    forces = get_forces_global_numba(pos, radius, K_SPRING, BOX_SIZE)

    t0 = time.time()
    for step in range(MAX_STEPS):
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            e_current = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_trace.append(e_current)
            time_trace.append(elapsed)

            if time_to_target is None and e_current <= energy_target:
                time_to_target = elapsed

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

        d_dt[mask_reset] *= F_DEC
        d_alpha[mask_reset] = ALPHA_START
        vel[mask_reset[atom_indices]] = 0.0
        d_neg[mask_reset] = 0

    total_time = time.time() - t0
    return {
        "energy_trace": np.array(energy_trace),
        "time_trace": np.array(time_trace),
        "time_to_target": time_to_target,
        "final_energy": energy_trace[-1] if energy_trace else np.inf,
        "total_time": total_time,
        "steps": step + 1,
    }


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(args.seed)
    base_positions = np.random.rand(N_ATOMS, 2)

    print("Time-to-Target Experiment — Async vs Global FIRE")
    print(f"  N_ATOMS:        {N_ATOMS:,}")
    print(f"  Numba threads:  {args.cpus}")
    print(f"  Wall time/run:  {MAX_WALL_TIME}s")
    print(f"  Energy target:  {ENERGY_TARGET:.0e}")
    print(f"  Delayed reset:  {args.delayed_reset}")
    print(f"  Random seed:    {args.seed}")
    print(f"  Log interval:   every {LOG_INTERVAL} step(s)")
    print(f"  Densities:      {args.phi}")
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

        if args.grids:
            k_values = args.grids
        else:
            k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in GAMMA_TARGETS]
            k_values = sorted(list(set(k_values)))
            while len(k_values) < len(GAMMA_TARGETS):
                k_values.append(k_values[-1] + 2)
            k_values = k_values[:len(GAMMA_TARGETS)]

        print("=" * 70)
        print(f"phi = {phi:.2f} | radius = {current_radius:.6f} | grids = {k_values}")
        print("=" * 70)

        # --- Global FIRE ---
        print(f"  Global FIRE ...", end=" ", flush=True)
        glob_result = run_global_fire_ttt(base_positions, current_radius, ENERGY_TARGET)
        ttt_str = f"{glob_result['time_to_target']:.2f}s" if glob_result['time_to_target'] is not None else "NEVER"
        print(f"done in {glob_result['total_time']:.1f}s | "
              f"final E={glob_result['final_energy']:.4e} | "
              f"time-to-target={ttt_str} | "
              f"steps={glob_result['steps']}")

        phi_results = {"global": {
            "time_to_target": glob_result["time_to_target"],
            "final_energy": float(glob_result["final_energy"]),
            "total_time": glob_result["total_time"],
            "steps": glob_result["steps"],
        }}

        np.savez_compressed(
            os.path.join(output_dir, f"ttt_phi{phi:.2f}_global.npz"),
            energy_trace=glob_result["energy_trace"],
            time_trace=glob_result["time_trace"],
            phi=phi, radius=current_radius, n_atoms=N_ATOMS,
            energy_target=ENERGY_TARGET, seed=args.seed,
        )

        # --- Async FIRE for each grid ---
        for grid in k_values:
            actual_gamma = (8 * current_radius * grid) / BOX_SIZE
            print(f"  Async {grid}x{grid} (gamma={actual_gamma:.3f}) ...", end=" ", flush=True)

            async_result = run_async_fire_ttt(
                base_positions, current_radius, grid, ENERGY_TARGET,
                delayed_reset=args.delayed_reset,
            )
            ttt_str = f"{async_result['time_to_target']:.2f}s" if async_result['time_to_target'] is not None else "NEVER"
            print(f"done in {async_result['total_time']:.1f}s | "
                  f"final E={async_result['final_energy']:.4e} | "
                  f"time-to-target={ttt_str} | "
                  f"steps={async_result['steps']}")

            phi_results[f"async_K{grid}"] = {
                "time_to_target": async_result["time_to_target"],
                "final_energy": float(async_result["final_energy"]),
                "total_time": async_result["total_time"],
                "steps": async_result["steps"],
                "gamma": actual_gamma,
            }

            np.savez_compressed(
                os.path.join(output_dir, f"ttt_phi{phi:.2f}_async_K{grid}.npz"),
                energy_trace=async_result["energy_trace"],
                time_trace=async_result["time_trace"],
                phi=phi, radius=current_radius, n_atoms=N_ATOMS,
                grid_divs=grid, gamma=actual_gamma,
                energy_target=ENERGY_TARGET, seed=args.seed,
            )

        all_results[f"phi_{phi:.2f}"] = phi_results

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY — Time to reach E < {:.0e}".format(ENERGY_TARGET))
    print("=" * 70)
    print(f"{'Method':<25} {'phi':>6} {'Time-to-Target':>16} {'Final Energy':>14} {'Speedup':>10}")
    print("-" * 70)

    for phi in args.phi:
        key = f"phi_{phi:.2f}"
        if key not in all_results:
            continue
        res = all_results[key]
        glob_ttt = res["global"]["time_to_target"]

        glob_ttt_str = f"{glob_ttt:.2f}s" if glob_ttt is not None else "NEVER"
        print(f"{'Global FIRE':<25} {phi:>6.2f} {glob_ttt_str:>16} {res['global']['final_energy']:>14.4e} {'baseline':>10}")

        for label, data in res.items():
            if label == "global":
                continue
            async_ttt = data["time_to_target"]
            async_ttt_str = f"{async_ttt:.2f}s" if async_ttt is not None else "NEVER"

            if glob_ttt is not None and async_ttt is not None and async_ttt > 0:
                speedup = f"{glob_ttt / async_ttt:.2f}x"
            elif glob_ttt is None and async_ttt is not None:
                speedup = "INF"
            elif async_ttt is None:
                speedup = "N/A"
            else:
                speedup = "N/A"

            print(f"  {label:<23} {phi:>6.2f} {async_ttt_str:>16} {data['final_energy']:>14.4e} {speedup:>10}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary JSON:     {summary_path}")
