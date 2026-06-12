#!/usr/bin/env python3
"""
Comparison Run — FIRE Parameter Sweep Compute Script for Sherlock HPC
Converted from comparison_run.ipynb

Concurrency is configurable:
  python comparison_run_compute.py --cpus 16
  python comparison_run_compute.py --cpus 8 --n-atoms 1000000 --wall-time 720

The old Sherlock implementation uses one Numba-compiled serial kernel per
worker process.  --cpus controls the number of worker processes.
"""

import argparse
import multiprocessing as mp
import os
import sys

parser = argparse.ArgumentParser(description="FIRE Parameter Sweep — Comparison Run")
parser.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="Number of worker processes (default: SLURM_CPUS_PER_TASK or 1)")
parser.add_argument("--n-atoms", type=int, default=1000000, help="Number of atoms (default: 1000000)")
parser.add_argument("--wall-time", type=float, default=720.0, help="Max wall time per run in seconds (default: 720)")
parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "parameter_sweep"),
                    help="Output directory for .npz files")
args = parser.parse_args()

os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import time
from numba import njit

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
TOL = 1e-4

MAX_STEPS = 100000
CAPTURE_STEP = 150
MAX_WALL_TIME = args.wall_time

PHI_SWEEP = [0.80, 0.82, 0.84, 0.86, 0.90]
GAMMA_TARGETS = [0.01, 0.05, 0.08, 0.1, 0.13]
BASE_POSITIONS = None


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


@njit
def get_total_energy_cell_list_numba(pos, radius, k_spring, box_size):
    energy = 0.0
    cutoff = 2.0 * radius
    cutoff_sq = cutoff * cutoff
    head, next_p, n_cells = build_cell_list(pos, box_size, cutoff)

    for cx in range(n_cells):
        for cy in range(n_cells):
            i = head[cx, cy]
            while i != -1:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx = (cx + dx) % n_cells
                        ny = (cy + dy) % n_cells
                        j = head[nx, ny]
                        while j != -1:
                            if i < j:
                                dx_vec = pos[i, 0] - pos[j, 0]
                                dy_vec = pos[i, 1] - pos[j, 1]
                                dx_vec = dx_vec - box_size * np.round(dx_vec / box_size)
                                dy_vec = dy_vec - box_size * np.round(dy_vec / box_size)
                                dist_sq = dx_vec * dx_vec + dy_vec * dy_vec
                                if dist_sq < cutoff_sq:
                                    dist = np.sqrt(dist_sq)
                                    overlap = cutoff - dist
                                    energy += 0.5 * k_spring * overlap * overlap
                            j = next_p[j]
                i = next_p[i]
    return energy


@njit
def get_forces_energy_scalar_dt_numba(pos, radius, k_spring, box_size):
    forces = np.zeros_like(pos)
    energy = 0.0
    cutoff = 2.0 * radius
    cutoff_sq = cutoff * cutoff
    head, next_p, n_cells = build_cell_list(pos, box_size, cutoff)

    for cx in range(n_cells):
        for cy in range(n_cells):
            i = head[cx, cy]
            while i != -1:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx = (cx + dx) % n_cells
                        ny = (cy + dy) % n_cells
                        j = head[nx, ny]
                        while j != -1:
                            if i < j:
                                dx_vec = pos[i, 0] - pos[j, 0]
                                dy_vec = pos[i, 1] - pos[j, 1]
                                dx_vec = dx_vec - box_size * np.round(dx_vec / box_size)
                                dy_vec = dy_vec - box_size * np.round(dy_vec / box_size)
                                dist_sq = dx_vec * dx_vec + dy_vec * dy_vec

                                if dist_sq < cutoff_sq:
                                    dist = np.sqrt(dist_sq)
                                    overlap = cutoff - dist
                                    energy += 0.5 * k_spring * overlap * overlap

                                    if dist_sq > 1e-12:
                                        f_mag = k_spring * overlap
                                        fx = (dx_vec / dist) * f_mag
                                        fy = (dy_vec / dist) * f_mag

                                        forces[i, 0] += fx
                                        forces[i, 1] += fy
                                        forces[j, 0] -= fx
                                        forces[j, 1] -= fy
                            j = next_p[j]
                i = next_p[i]
    return forces, energy


@njit
def get_forces_dt_energy_domains_numba(pos, domain_dt, atom_indices, radius, k_spring, box_size):
    forces = np.zeros_like(pos)
    forces_dt = np.zeros_like(pos)
    energy = 0.0
    cutoff = 2.0 * radius
    cutoff_sq = cutoff * cutoff
    head, next_p, n_cells = build_cell_list(pos, box_size, cutoff)

    for cx in range(n_cells):
        for cy in range(n_cells):
            i = head[cx, cy]
            while i != -1:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx = (cx + dx) % n_cells
                        ny = (cy + dy) % n_cells
                        j = head[nx, ny]
                        while j != -1:
                            if i < j:
                                dx_vec = pos[i, 0] - pos[j, 0]
                                dy_vec = pos[i, 1] - pos[j, 1]
                                dx_vec = dx_vec - box_size * np.round(dx_vec / box_size)
                                dy_vec = dy_vec - box_size * np.round(dy_vec / box_size)
                                dist_sq = dx_vec * dx_vec + dy_vec * dy_vec

                                if dist_sq < cutoff_sq:
                                    dist = np.sqrt(dist_sq)
                                    overlap = cutoff - dist
                                    energy += 0.5 * k_spring * overlap * overlap

                                    if dist_sq > 1e-12:
                                        f_mag = k_spring * overlap
                                        fx = (dx_vec / dist) * f_mag
                                        fy = (dy_vec / dist) * f_mag

                                        forces[i, 0] += fx
                                        forces[i, 1] += fy
                                        forces[j, 0] -= fx
                                        forces[j, 1] -= fy

                                        dt_ij = min(domain_dt[atom_indices[i]], domain_dt[atom_indices[j]])
                                        forces_dt[i, 0] += fx * dt_ij
                                        forces_dt[i, 1] += fy * dt_ij
                                        forces_dt[j, 0] -= fx * dt_ij
                                        forces_dt[j, 1] -= fy * dt_ij
                            j = next_p[j]
                i = next_p[i]
    return forces, forces_dt, energy


@njit
def get_forces_dt_energy_only_domains_numba(pos, domain_dt, atom_indices, radius, k_spring, box_size):
    forces_dt = np.zeros_like(pos)
    energy = 0.0
    cutoff = 2.0 * radius
    cutoff_sq = cutoff * cutoff
    head, next_p, n_cells = build_cell_list(pos, box_size, cutoff)

    for cx in range(n_cells):
        for cy in range(n_cells):
            i = head[cx, cy]
            while i != -1:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx = (cx + dx) % n_cells
                        ny = (cy + dy) % n_cells
                        j = head[nx, ny]
                        while j != -1:
                            if i < j:
                                dx_vec = pos[i, 0] - pos[j, 0]
                                dy_vec = pos[i, 1] - pos[j, 1]
                                dx_vec = dx_vec - box_size * np.round(dx_vec / box_size)
                                dy_vec = dy_vec - box_size * np.round(dy_vec / box_size)
                                dist_sq = dx_vec * dx_vec + dy_vec * dy_vec

                                if dist_sq < cutoff_sq:
                                    dist = np.sqrt(dist_sq)
                                    overlap = cutoff - dist
                                    energy += 0.5 * k_spring * overlap * overlap

                                    if dist_sq > 1e-12:
                                        f_mag = k_spring * overlap
                                        fx = (dx_vec / dist) * f_mag
                                        fy = (dy_vec / dist) * f_mag
                                        dt_ij = min(domain_dt[atom_indices[i]], domain_dt[atom_indices[j]])

                                        forces_dt[i, 0] += fx * dt_ij
                                        forces_dt[i, 1] += fy * dt_ij
                                        forces_dt[j, 0] -= fx * dt_ij
                                        forces_dt[j, 1] -= fy * dt_ij
                            j = next_p[j]
                i = next_p[i]
    return forces_dt, energy


@njit
def map_atoms_to_grid_inplace(pos, divs, out):
    n_atoms = pos.shape[0]
    for i in range(n_atoms):
        ix = int(np.floor(pos[i, 0] * divs))
        iy = int(np.floor(pos[i, 1] * divs))
        if ix < 0:
            ix = 0
        elif ix >= divs:
            ix = divs - 1
        if iy < 0:
            iy = 0
        elif iy >= divs:
            iy = divs - 1
        out[i] = ix * divs + iy


@njit
def wrap_positions_inplace(pos, box_size):
    n_atoms = pos.shape[0]
    for i in range(n_atoms):
        pos[i, 0] = pos[i, 0] - box_size * np.floor(pos[i, 0] / box_size)
        pos[i, 1] = pos[i, 1] - box_size * np.floor(pos[i, 1] / box_size)


@njit
def global_kick_drift_inplace(pos, vel, forces, dt, mass, box_size):
    n_atoms = pos.shape[0]
    half_dt_over_mass = 0.5 * dt / mass
    for i in range(n_atoms):
        vel[i, 0] += half_dt_over_mass * forces[i, 0]
        vel[i, 1] += half_dt_over_mass * forces[i, 1]
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
    wrap_positions_inplace(pos, box_size)


@njit
def global_second_kick_inplace(vel, forces, dt, mass):
    n_atoms = vel.shape[0]
    half_dt_over_mass = 0.5 * dt / mass
    for i in range(n_atoms):
        vel[i, 0] += half_dt_over_mass * forces[i, 0]
        vel[i, 1] += half_dt_over_mass * forces[i, 1]


@njit
def async_kick_drift_inplace(pos, vel, forces_dt, atom_indices, domain_dt, mass, box_size):
    n_atoms = pos.shape[0]
    half_inv_mass = 0.5 / mass
    for i in range(n_atoms):
        vel[i, 0] += half_inv_mass * forces_dt[i, 0]
        vel[i, 1] += half_inv_mass * forces_dt[i, 1]
        dt_i = domain_dt[atom_indices[i]]
        pos[i, 0] += vel[i, 0] * dt_i
        pos[i, 1] += vel[i, 1] * dt_i
    wrap_positions_inplace(pos, box_size)


@njit
def async_second_kick_inplace(vel, forces_dt, mass):
    n_atoms = vel.shape[0]
    half_inv_mass = 0.5 / mass
    for i in range(n_atoms):
        vel[i, 0] += half_inv_mass * forces_dt[i, 0]
        vel[i, 1] += half_inv_mass * forces_dt[i, 1]


@njit
def apply_global_fire_mixing_inplace(vel, forces, alpha):
    n_atoms = vel.shape[0]
    p_global = 0.0
    force_sq = 0.0

    for i in range(n_atoms):
        fx = forces[i, 0]
        fy = forces[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        p_global += fx * vx + fy * vy
        force_sq += fx * fx + fy * fy

    one_minus_alpha = 1.0 - alpha
    for i in range(n_atoms):
        fx = forces[i, 0]
        fy = forces[i, 1]
        f_sq = fx * fx + fy * fy
        if f_sq > 1e-24:
            vx = vel[i, 0]
            vy = vel[i, 1]
            v_mag = np.sqrt(vx * vx + vy * vy)
            f_mag = np.sqrt(f_sq)
            force_scale = alpha * v_mag / f_mag
            vel[i, 0] = one_minus_alpha * vx + force_scale * fx
            vel[i, 1] = one_minus_alpha * vy + force_scale * fy

    return p_global, force_sq


@njit
def apply_async_fire_mixing_inplace(vel, forces, atom_indices, domain_alpha, n_domains):
    n_atoms = vel.shape[0]
    p_domain = np.zeros(n_domains)
    force_sq = 0.0

    for i in range(n_atoms):
        fx = forces[i, 0]
        fy = forces[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        p_domain[atom_indices[i]] += fx * vx + fy * vy
        force_sq += fx * fx + fy * fy

    for i in range(n_atoms):
        fx = forces[i, 0]
        fy = forces[i, 1]
        f_sq = fx * fx + fy * fy
        if f_sq > 1e-24:
            alpha = domain_alpha[atom_indices[i]]
            vx = vel[i, 0]
            vy = vel[i, 1]
            v_mag = np.sqrt(vx * vx + vy * vy)
            f_mag = np.sqrt(f_sq)
            force_scale = alpha * v_mag / f_mag
            vel[i, 0] = (1.0 - alpha) * vx + force_scale * fx
            vel[i, 1] = (1.0 - alpha) * vy + force_scale * fy

    return p_domain, force_sq


@njit
def zero_velocity_for_down_domains_inplace(vel, atom_indices, domain_is_down):
    n_atoms = vel.shape[0]
    for i in range(n_atoms):
        if domain_is_down[atom_indices[i]]:
            vel[i, 0] = 0.0
            vel[i, 1] = 0.0


# ==========================================
# 3. TIMED FIRE ENGINES
# ==========================================
def run_global_fire(pos_init, max_steps, radius):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    dt = DT_INIT
    alpha = ALPHA_START
    npos = 0

    energy_history, dt_history, time_history = [], [], []
    t0 = time.time()
    forces, current_energy = get_forces_energy_scalar_dt_numba(pos, radius, K_SPRING, BOX_SIZE)
    for step in range(max_steps):
        if step % 10 == 0:
            elapsed = time.time() - t0
            energy_history.append(current_energy)
            dt_history.append(dt)
            time_history.append(elapsed)

            if elapsed > MAX_WALL_TIME:
                break

        global_kick_drift_inplace(pos, vel, forces, dt, MASS, BOX_SIZE)
        forces, current_energy = get_forces_energy_scalar_dt_numba(pos, radius, K_SPRING, BOX_SIZE)
        global_second_kick_inplace(vel, forces, dt, MASS)

        P_global, force_sq = apply_global_fire_mixing_inplace(vel, forces, alpha)
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

        if np.sqrt(force_sq / (2 * N_ATOMS)) < TOL:
            break

    return pos, np.array(energy_history), np.array(dt_history), np.array(time_history), time.time() - t0


def run_async_fire(pos_init, max_steps, capture_step, radius, grid_divs):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    n_domains = grid_divs * grid_divs

    d_dt = np.full(n_domains, DT_INIT)
    d_alpha = np.full(n_domains, ALPHA_START)
    d_npos = np.zeros(n_domains, dtype=int)

    energy_history = []
    dt_mean_history = []
    time_history = []
    snapshot_dt = None
    accumulated_dt = np.zeros(n_domains)

    t0 = time.time()
    atom_indices = np.empty(pos.shape[0], dtype=np.int64)
    map_atoms_to_grid_inplace(pos, grid_divs, atom_indices)
    forces_dt, current_energy = get_forces_dt_energy_only_domains_numba(
        pos, d_dt, atom_indices, radius, K_SPRING, BOX_SIZE
    )
    for step in range(max_steps):
        accumulated_dt += d_dt

        if step % 10 == 0:
            elapsed = time.time() - t0
            energy_history.append(current_energy)
            dt_mean_history.append(np.mean(d_dt))
            time_history.append(elapsed)

            if elapsed > MAX_WALL_TIME:
                if snapshot_dt is None:
                    snapshot_dt = np.copy(d_dt)
                break

        if step == capture_step:
            snapshot_dt = np.copy(d_dt)

        async_kick_drift_inplace(pos, vel, forces_dt, atom_indices, d_dt, MASS, BOX_SIZE)
        map_atoms_to_grid_inplace(pos, grid_divs, atom_indices)
        forces, forces_dt, current_energy = get_forces_dt_energy_domains_numba(
            pos, d_dt, atom_indices, radius, K_SPRING, BOX_SIZE
        )
        async_second_kick_inplace(vel, forces_dt, MASS)

        p_domain, force_sq = apply_async_fire_mixing_inplace(vel, forces, atom_indices, d_alpha, n_domains)
        mask_up = p_domain > 0
        d_npos[mask_up] += 1
        mask_grow = mask_up & (d_npos > N_MIN)
        d_dt[mask_grow] = np.minimum(d_dt[mask_grow] * F_INC, DT_MAX)
        d_alpha[mask_grow] *= F_ALPHA

        mask_down = p_domain <= 0
        dt_changed = np.any(mask_grow) or np.any(mask_down)
        d_npos[mask_down] = 0
        d_dt[mask_down] *= F_DEC
        d_alpha[mask_down] = ALPHA_START
        zero_velocity_for_down_domains_inplace(vel, atom_indices, mask_down)

        if np.sqrt(force_sq / (2 * N_ATOMS)) < TOL:
            if snapshot_dt is None:
                snapshot_dt = np.copy(d_dt)
            break

        if dt_changed:
            forces_dt, current_energy = get_forces_dt_energy_only_domains_numba(
                pos, d_dt, atom_indices, radius, K_SPRING, BOX_SIZE
            )

    if snapshot_dt is None:
        snapshot_dt = np.copy(d_dt)

    return pos, snapshot_dt, np.array(energy_history), np.array(dt_mean_history), np.array(time_history), time.time() - t0, accumulated_dt


def run_sweep_point(task):
    if BASE_POSITIONS is None:
        raise RuntimeError("BASE_POSITIONS was not initialized before starting worker pool")

    output_dir, phi_idx, phi, grid_idx, grid, current_radius = task
    actual_gamma = (8 * current_radius * grid) / BOX_SIZE
    point_t0 = time.time()

    print(
        f"  -> point phi={phi:.2f}, K={grid}x{grid}, gamma={actual_gamma:.3f}: "
        "global + async",
        flush=True,
    )

    pos_glob, e_hist_glob, dt_hist_glob, t_hist_glob, time_glob = run_global_fire(
        BASE_POSITIONS, MAX_STEPS, current_radius
    )
    pos_async, dt_grid_snap, e_hist_async, dt_hist_async, t_hist_async, time_async, accumulated_dt = run_async_fire(
        BASE_POSITIONS, MAX_STEPS, CAPTURE_STEP, current_radius, grid
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
        box_size=BOX_SIZE, phi=phi, gamma=actual_gamma,
        point_time=time.time() - point_t0,
        phi_idx=phi_idx, grid_idx=grid_idx,
    )

    return (
        f"     saved {out_filename} in {time.time() - point_t0:.1f}s "
        f"(global {time_glob:.1f}s, async {time_async:.1f}s, "
        f"E_glob={e_hist_glob[-1]:.4e}, E_async={e_hist_async[-1]:.4e})"
    )


# ==========================================
# 4. AUTOMATED PARAMETER SPACE RUNNER
# ==========================================
if __name__ == "__main__":
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    BASE_POSITIONS = np.random.rand(N_ATOMS, 2)

    print(f"Comparison Run — FIRE Parameter Sweep")
    print(f"  N_ATOMS:        {N_ATOMS:,}")
    print(f"  Worker procs:   {args.cpus}")
    print(f"  Numba threads:  1 per worker")
    print(f"  Wall time/run:  {MAX_WALL_TIME}s")
    print(f"  Output dir:     {output_dir}")
    print(f"  SLURM Job ID:   {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"  Node:           {os.environ.get('SLURM_NODELIST', 'local')}")
    print()

    print("Warming up Numba JIT compiler...")
    dummy_r = np.sqrt((0.82 * BOX_SIZE ** 2) / (10 * np.pi))
    dummy_pos = BASE_POSITIONS[:10].copy()
    dummy_vel = np.zeros_like(dummy_pos)
    dummy_domains = np.zeros(10, dtype=np.int64)
    dummy_dt = np.ones(4) * DT_INIT
    dummy_alpha = np.ones(4) * ALPHA_START
    dummy_down = np.zeros(4, dtype=np.bool_)
    dummy_forces, _ = get_forces_energy_scalar_dt_numba(dummy_pos, dummy_r, K_SPRING, BOX_SIZE)
    dummy_forces_async, dummy_forces_dt, _ = get_forces_dt_energy_domains_numba(
        dummy_pos, dummy_dt, dummy_domains, dummy_r, K_SPRING, BOX_SIZE
    )
    dummy_forces_dt_only, _ = get_forces_dt_energy_only_domains_numba(
        dummy_pos, dummy_dt, dummy_domains, dummy_r, K_SPRING, BOX_SIZE
    )
    _ = get_total_energy_cell_list_numba(dummy_pos, dummy_r, K_SPRING, BOX_SIZE)
    map_atoms_to_grid_inplace(dummy_pos, 2, dummy_domains)
    global_kick_drift_inplace(dummy_pos, dummy_vel, dummy_forces, DT_INIT, MASS, BOX_SIZE)
    global_second_kick_inplace(dummy_vel, dummy_forces, DT_INIT, MASS)
    async_kick_drift_inplace(dummy_pos, dummy_vel, dummy_forces_dt, dummy_domains, dummy_dt, MASS, BOX_SIZE)
    async_second_kick_inplace(dummy_vel, dummy_forces_dt_only, MASS)
    _ = apply_global_fire_mixing_inplace(dummy_vel, dummy_forces, ALPHA_START)
    _ = apply_async_fire_mixing_inplace(dummy_vel, dummy_forces_async, dummy_domains, dummy_alpha, 4)
    zero_velocity_for_down_domains_inplace(dummy_vel, dummy_domains, dummy_down)
    print("JIT warmup complete.\n")

    print(f"Starting 25-run parameter sweep...")
    print(f"Densities (phi): {PHI_SWEEP}")
    print(f"Gamma targets:   {GAMMA_TARGETS}")
    print("=" * 70)

    sweep_t0 = time.time()
    tasks = []

    for phi_idx, phi in enumerate(PHI_SWEEP):
        current_radius = np.sqrt((phi * BOX_SIZE ** 2) / (N_ATOMS * np.pi))

        k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in GAMMA_TARGETS]
        k_values = sorted(list(set(k_values)))

        while len(k_values) < len(GAMMA_TARGETS):
            k_values.append(k_values[-1] + 2)
        k_values = k_values[:len(GAMMA_TARGETS)]

        print(f"\n[DENSITY PHASE {phi_idx + 1}/5] phi = {phi:.2f} (Radius: {current_radius:.6f})")
        print(f"  Grid sizes (K): {k_values}")
        for grid_idx, grid in enumerate(k_values):
            tasks.append((output_dir, phi_idx, phi, grid_idx, grid, current_radius))

    print(f"\nLaunching {len(tasks)} comparison points across {args.cpus} workers.")
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.cpus) as pool:
        for message in pool.imap_unordered(run_sweep_point, tasks):
            print(message, flush=True)

    total_time = time.time() - sweep_t0
    print("\n" + "=" * 70)
    print(f"Parameter sweep complete! Total wall time: {total_time:.1f}s ({total_time / 3600:.2f} hours)")
    print(f"Results saved to: {output_dir}/")
