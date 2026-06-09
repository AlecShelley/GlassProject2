import numpy as np
import argparse
import json
import time
import os
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
from numba import njit

# ==========================================
# 1. SYSTEM PARAMETERS & SWEEP TARGETS
# ==========================================
# Legacy full Sherlock sweep:
# N_ATOMS = 2500000
# MAX_WALL_TIME = 120.0
# PHI_SWEEP = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.90]
# GAMMA_TARGETS = [0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25]

N_ATOMS = 10_000_000
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
TOL = -1.0

MAX_STEPS = 100000        
CAPTURE_STEP = 150  
POINT_WALL_TIME = 120.0
FIRE_WALL_TIME = POINT_WALL_TIME / 2.0
LOG_INTERVAL = 1

# Short 3x3 Sherlock smoke sweep. Middle values are phi=0.84 and gamma=0.05.
PHI_SWEEP = [0.82, 0.84, 0.86]
GAMMA_TARGETS = [0.03, 0.05, 0.07]
OUTPUT_ROOT = "parameter_sweep_3x3_short_runs"
OUTPUT_DIR = None
N_WORKERS = len(PHI_SWEEP) * len(GAMMA_TARGETS)
BASE_POSITIONS = None

# ==========================================
# 2. NUMBA JIT-COMPILED CELL LIST PHYSICS 
# ==========================================
@njit
def build_cell_list(pos, box_size, cutoff):
    n_atoms = pos.shape[0]
    n_cells = int(np.floor(box_size / cutoff))
    if n_cells < 3: n_cells = 3
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
def get_forces_cell_list_numba(pos, atom_dt, radius, k_spring, box_size):
    n_atoms = pos.shape[0]
    forces = np.zeros_like(pos)
    forces_dt = np.zeros_like(pos)
    cutoff = 2.0 * radius
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
                                dist_sq = dx_vec**2 + dy_vec**2
                                
                                if dist_sq < cutoff**2 and dist_sq > 1e-12:
                                    dist = np.sqrt(dist_sq)
                                    f_mag = k_spring * (cutoff - dist)
                                    fx = (dx_vec / dist) * f_mag
                                    fy = (dy_vec / dist) * f_mag
                                    
                                    forces[i, 0] += fx
                                    forces[i, 1] += fy
                                    forces[j, 0] -= fx
                                    forces[j, 1] -= fy
                                    
                                    dt_ij = min(atom_dt[i], atom_dt[j])
                                    forces_dt[i, 0] += fx * dt_ij
                                    forces_dt[i, 1] += fy * dt_ij
                                    forces_dt[j, 0] -= fx * dt_ij
                                    forces_dt[j, 1] -= fy * dt_ij
                            j = next_p[j]
                i = next_p[i]
    return forces, forces_dt

@njit
def get_total_energy_cell_list_numba(pos, radius, k_spring, box_size):
    energy = 0.0
    cutoff = 2.0 * radius
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
                                dist_sq = dx_vec**2 + dy_vec**2
                                if dist_sq < cutoff**2:
                                    dist = np.sqrt(dist_sq)
                                    overlap = cutoff - dist
                                    energy += 0.5 * k_spring * (overlap**2)
                            j = next_p[j]
                i = next_p[i]
    return energy

def map_atoms_to_grid(pos, divs):
    ix = np.clip(np.floor(pos[:, 0] * divs).astype(int), 0, divs-1)
    iy = np.clip(np.floor(pos[:, 1] * divs).astype(int), 0, divs-1)
    return ix * divs + iy

# ==========================================
# 3. TIMED FIRE ENGINES
# ==========================================
def run_global_fire(pos_init, max_steps, radius, wall_time_limit):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    dt = DT_INIT
    alpha = ALPHA_START
    npos = 0
    energy_history, dt_history, time_history = [], [], []
    
    t0 = time.time()
    for step in range(max_steps):
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            e_curr = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_history.append(e_curr)
            dt_history.append(dt)
            time_history.append(elapsed)
            if elapsed >= wall_time_limit: break
            
        atom_dt = np.full(N_ATOMS, dt)
        forces, forces_dt = get_forces_cell_list_numba(pos, atom_dt, radius, K_SPRING, BOX_SIZE)
        
        vel += 0.5 * forces_dt / MASS
        pos += vel * dt
        pos = np.mod(pos, BOX_SIZE)
        
        forces_new, forces_dt_new = get_forces_cell_list_numba(pos, atom_dt, radius, K_SPRING, BOX_SIZE)
        vel += 0.5 * forces_dt_new / MASS
        forces = forces_new
        
        P_global = np.sum(forces * vel)
        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        f_mag = np.linalg.norm(forces, axis=1, keepdims=True)
        mask = (f_mag > 1e-12).flatten()
        
        if np.any(mask):
            vel[mask] = (1 - alpha) * vel[mask] + alpha * (forces[mask]/f_mag[mask]) * v_mag[mask]
            
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
            
        if np.sqrt(np.sum(forces**2) / (2 * N_ATOMS)) < TOL: break
            
    return np.array(energy_history), np.array(dt_history), np.array(time_history), time.time() - t0

def run_async_fire(pos_init, max_steps, radius, grid_divs, wall_time_limit):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    n_domains = grid_divs * grid_divs
    
    d_dt = np.full(n_domains, DT_INIT)
    d_alpha = np.full(n_domains, ALPHA_START)
    d_npos = np.zeros(n_domains, dtype=int)
    
    energy_history, dt_mean_history, time_history = [], [], []
    
    atom_indices = map_atoms_to_grid(pos, grid_divs)
    forces, forces_dt = get_forces_cell_list_numba(pos, d_dt[atom_indices], radius, K_SPRING, BOX_SIZE)
    
    t0 = time.time()
    for step in range(max_steps):
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            e_current = get_total_energy_cell_list_numba(pos, radius, K_SPRING, BOX_SIZE)
            energy_history.append(e_current)
            dt_mean_history.append(np.mean(d_dt))
            time_history.append(elapsed)
            if elapsed >= wall_time_limit: break
                
        vel += 0.5 * forces_dt / MASS
        pos += vel * d_dt[atom_indices][:, np.newaxis]
        pos = np.mod(pos, BOX_SIZE)
        
        atom_indices = map_atoms_to_grid(pos, grid_divs)
        forces_new, forces_dt_new = get_forces_cell_list_numba(pos, d_dt[atom_indices], radius, K_SPRING, BOX_SIZE)
        vel += 0.5 * forces_dt_new / MASS
        forces = forces_new; forces_dt = forces_dt_new
        
        p_domain = np.bincount(atom_indices, weights=np.sum(forces * vel, axis=1), minlength=n_domains)
        atom_alpha = d_alpha[atom_indices][:, np.newaxis]
        v_mag = np.linalg.norm(vel, axis=1, keepdims=True)
        f_mag = np.linalg.norm(forces, axis=1, keepdims=True)
        mask = (f_mag > 1e-12).flatten()
        if np.any(mask): 
            vel[mask] = (1 - atom_alpha[mask]) * vel[mask] + atom_alpha[mask] * (forces[mask]/f_mag[mask]) * v_mag[mask]
            
        mask_up = p_domain > 0
        d_npos[mask_up] += 1
        mask_grow = mask_up & (d_npos > N_MIN)
        d_dt[mask_grow] = np.minimum(d_dt[mask_grow] * F_INC, DT_MAX)
        d_alpha[mask_grow] *= F_ALPHA
        
        mask_down = p_domain <= 0
        d_npos[mask_down] = 0; d_dt[mask_down] *= F_DEC; d_alpha[mask_down] = ALPHA_START
        vel[mask_down[atom_indices]] = 0.0
        
        if np.sqrt(np.sum(forces**2) / (2 * N_ATOMS)) < TOL: break

        forces, forces_dt = get_forces_cell_list_numba(pos, d_dt[atom_indices], radius, K_SPRING, BOX_SIZE)
            
    return np.array(energy_history), np.array(dt_mean_history), np.array(time_history), time.time() - t0

def run_sweep_point(task):
    phi_idx, phi, gamma_target, grid, current_radius = task
    actual_gamma = (8 * current_radius * grid) / BOX_SIZE
    point_t0 = time.time()
    print(
        f"   -> Short point phi={phi:.2f}, target gamma={gamma_target:.2f}, "
        f"K={grid}x{grid}, actual gamma={actual_gamma:.3f}",
        flush=True,
    )
    e_hist_glob, dt_hist_glob, t_hist_glob, time_glob = run_global_fire(
        BASE_POSITIONS, MAX_STEPS, current_radius, FIRE_WALL_TIME
    )
    
    e_hist_async, dt_hist_async, t_hist_async, time_async = run_async_fire(
        BASE_POSITIONS, MAX_STEPS, current_radius, grid, FIRE_WALL_TIME
    )
    
    out_filename = os.path.join(
        OUTPUT_DIR,
        f"sweep_phi{phi:.2f}_gamma{gamma_target:.2f}_grid{grid}.npz",
    )
    np.savez_compressed(out_filename, 
                        e_hist_async=e_hist_async, e_hist_glob=e_hist_glob,
                        t_hist_async=t_hist_async, t_hist_glob=t_hist_glob,
                        time_async=time_async, time_glob=time_glob,
                        grid_divs=grid, n_atoms=N_ATOMS, phi=phi,
                        gamma=actual_gamma, target_gamma=gamma_target,
                        point_time=time.time() - point_t0,
                        point_wall_time=POINT_WALL_TIME,
                        fire_wall_time=FIRE_WALL_TIME)
    print(
        f"      saved {out_filename} in {time.time() - point_t0:.1f}s "
        f"(global {time_glob:.1f}s, async {time_async:.1f}s)",
        flush=True,
    )
    return out_filename

def parse_args():
    parser = argparse.ArgumentParser(description="Run the 3x3 Async FIRE sweep.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Fresh directory for this run's .npz outputs. Defaults to a timestamped run directory.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id used when creating the default output directory.",
    )
    return parser.parse_args()

def default_run_id():
    return os.environ.get("ASYNC_FIRE_RUN_ID") or time.strftime("run_%Y%m%d_%H%M%S")

def write_run_metadata(output_dir, run_id):
    metadata = {
        "run_id": run_id,
        "n_atoms": N_ATOMS,
        "phi_sweep": PHI_SWEEP,
        "gamma_targets": GAMMA_TARGETS,
        "point_wall_time": POINT_WALL_TIME,
        "fire_wall_time": FIRE_WALL_TIME,
        "n_workers": N_WORKERS,
        "output_dir": output_dir,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == '__main__':
    args = parse_args()
    run_id = args.run_id or default_run_id()
    OUTPUT_DIR = args.output_dir or os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(OUTPUT_DIR, exist_ok=False)
    write_run_metadata(OUTPUT_DIR, run_id)

    np.random.seed(42)
    BASE_POSITIONS = np.random.rand(N_ATOMS, 2)

    expected_budget = len(PHI_SWEEP) * len(GAMMA_TARGETS) * POINT_WALL_TIME
    print(f"Warming up Numba JIT for N={N_ATOMS}...", flush=True)
    print(f"Writing run data to {OUTPUT_DIR}", flush=True)
    print(
        f"Short sweep budget: {len(PHI_SWEEP)} phi x {len(GAMMA_TARGETS)} gamma x "
        f"2 solver runs x {FIRE_WALL_TIME:.1f}s per run = {expected_budget:.1f}s "
        f"(~{expected_budget / 60.0:.1f} min) plus JIT/render overhead.",
        flush=True,
    )
    dummy_r = np.sqrt((0.82 * BOX_SIZE**2) / (10 * np.pi))
    _ = get_forces_cell_list_numba(BASE_POSITIONS[:10], np.ones(10), dummy_r, K_SPRING, BOX_SIZE)
    _ = get_total_energy_cell_list_numba(BASE_POSITIONS[:10], dummy_r, K_SPRING, BOX_SIZE)

    try:
        tasks = []
        for phi_idx, phi in enumerate(PHI_SWEEP):
            current_radius = np.sqrt((phi * BOX_SIZE**2) / (N_ATOMS * np.pi))
            k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in GAMMA_TARGETS]
            
            print(f"\n[DENSITY PHASE {phi_idx+1}/{len(PHI_SWEEP)}] Evaluating phi = {phi:.2f}", flush=True)
            for gamma_target, grid in zip(GAMMA_TARGETS, k_values):
                tasks.append((phi_idx, phi, gamma_target, grid, current_radius))
        print(f"\nLaunching {len(tasks)} sweep points across {N_WORKERS} workers.", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=N_WORKERS) as pool:
            for out_filename in pool.imap_unordered(run_sweep_point, tasks):
                print(f"Completed {out_filename}", flush=True)
                
    except Exception as e:
        print(f"\nExecution interrupted: {e}", flush=True)
        raise
