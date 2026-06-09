import numpy as np
import time
import os
import argparse
from numba import njit, prange

# ==========================================
# 0. COMMAND LINE ARGUMENTS (FOR HPC SWEEPS)
# ==========================================
parser = argparse.ArgumentParser(description="Async FIRE HPC Runner")
parser.add_argument('--N', type=int, default=5000, help='Number of atoms')
parser.add_argument('--grid', type=int, default=8, help='Grid divisions (KxK)')
parser.add_argument('--phi', type=float, default=0.90, help='Packing fraction')
parser.add_argument('--steps', type=int, default=1000, help='Max integration steps')
args = parser.parse_args()

# ==========================================
# 1. SYSTEM & FIRE PARAMETERS
# ==========================================
N_ATOMS = args.N
GRID_DIVS = args.grid
TARGET_FRACTION = args.phi
MAX_STEPS = args.steps

BOX_SIZE = 1.0
RADIUS = np.sqrt((TARGET_FRACTION * BOX_SIZE**2) / (N_ATOMS * np.pi))
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
CAPTURE_STEP = int(MAX_STEPS * 0.3)  # Dynamically capture at 30% of max steps

# ==========================================
# 2. NUMBA JIT-COMPILED CORE PHYSICS 
# ==========================================
@njit(parallel=True)
def get_forces_mpt_numba(pos, atom_dt, radius, k_spring):
    N = len(pos)
    forces = np.zeros_like(pos)
    forces_dt = np.zeros_like(pos)
    cutoff = 2.0 * radius
    for i in prange(N):
        for j in range(N):
            if i == j: continue
            
            dx_raw = pos[i, 0] - pos[j, 0]
            dy_raw = pos[i, 1] - pos[j, 1]
            
            dx = dx_raw - np.round(dx_raw)
            dy = dy_raw - np.round(dy_raw)
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist < cutoff:
                f_mag = k_spring * (cutoff - dist)
                if dist > 1e-12:
                    fx = (dx / dist) * f_mag
                    fy = (dy / dist) * f_mag
                else:
                    fx = 0.0
                    fy = 0.0
                
                forces[i, 0] += fx
                forces[i, 1] += fy
                dt_ij = min(atom_dt[i], atom_dt[j])
                forces_dt[i, 0] += fx * dt_ij
                forces_dt[i, 1] += fy * dt_ij
    return forces, forces_dt

@njit(parallel=True)
def get_total_energy_numba(pos, radius, k_spring):
    N = len(pos)
    total_energy = 0.0
    cutoff = 2.0 * radius
    for i in prange(N):
        for j in range(i + 1, N):
            
            dx_raw = pos[i, 0] - pos[j, 0]
            dy_raw = pos[i, 1] - pos[j, 1]
            
            dx = dx_raw - np.round(dx_raw)
            dy = dy_raw - np.round(dy_raw)
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist < cutoff:
                overlap = cutoff - dist
                total_energy += 0.5 * k_spring * overlap**2
                
    return total_energy

def map_atoms_to_grid(pos, divs):
    ix = np.clip(np.floor(pos[:, 0] * divs).astype(int), 0, divs-1)
    iy = np.clip(np.floor(pos[:, 1] * divs).astype(int), 0, divs-1)
    return ix * divs + iy

# ==========================================
# 3. PURE FIRE ENGINES
# ==========================================
def run_global_fire(pos_init, max_steps):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    dt = DT_INIT
    alpha = ALPHA_START
    npos = 0
    
    energy_history = []
    dt_history = []
    
    t0 = time.time()
    for step in range(max_steps):
        if step % 10 == 0:
            e_curr = get_total_energy_numba(pos, RADIUS, K_SPRING)
            energy_history.append(e_curr)
            dt_history.append(dt)
            
        atom_dt = np.full(N_ATOMS, dt)
        forces, forces_dt = get_forces_mpt_numba(pos, atom_dt, RADIUS, K_SPRING)
        
        vel += 0.5 * forces_dt / MASS
        pos += vel * dt
        pos = np.mod(pos, BOX_SIZE)
        
        forces_new, forces_dt_new = get_forces_mpt_numba(pos, atom_dt, RADIUS, K_SPRING)
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
            
        if np.sqrt(np.sum(forces**2) / (2 * N_ATOMS)) < TOL:
            break
            
    return pos, np.array(energy_history), np.array(dt_history), time.time() - t0

def run_async_fire(pos_init, max_steps, capture_step):
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos)
    n_domains = GRID_DIVS * GRID_DIVS
    
    d_dt = np.full(n_domains, DT_INIT)
    d_alpha = np.full(n_domains, ALPHA_START)
    d_npos = np.zeros(n_domains, dtype=int)
    
    energy_history = []
    dt_mean_history = []
    snapshot_dt = None
    accumulated_dt = np.zeros(n_domains)
    
    atom_indices = map_atoms_to_grid(pos, GRID_DIVS)
    forces, forces_dt = get_forces_mpt_numba(pos, d_dt[atom_indices], RADIUS, K_SPRING)
    
    t0 = time.time()
    for step in range(max_steps):
        accumulated_dt += d_dt
        
        if step % 10 == 0:
            e_current = get_total_energy_numba(pos, RADIUS, K_SPRING)
            energy_history.append(e_current)
            dt_mean_history.append(np.mean(d_dt))
            
        if step == capture_step:
            snapshot_dt = np.copy(d_dt)
            
        vel += 0.5 * forces_dt / MASS
        pos += vel * d_dt[atom_indices][:, np.newaxis]
        pos = np.mod(pos, BOX_SIZE)
        
        atom_indices = map_atoms_to_grid(pos, GRID_DIVS)
        forces_new, forces_dt_new = get_forces_mpt_numba(pos, d_dt[atom_indices], RADIUS, K_SPRING)
        vel += 0.5 * forces_dt_new / MASS
        forces = forces_new; forces_dt = forces_dt_new
        
        p_domain = np.zeros(n_domains)
        np.add.at(p_domain, atom_indices, np.sum(forces * vel, axis=1))
        
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
        vel[np.isin(atom_indices, np.where(mask_down)[0])] = 0.0
        
        if np.sqrt(np.sum(forces**2) / (2 * N_ATOMS)) < TOL: 
            if snapshot_dt is None: snapshot_dt = np.copy(d_dt)
            break

        forces, forces_dt = get_forces_mpt_numba(pos, d_dt[atom_indices], RADIUS, K_SPRING)
            
    if snapshot_dt is None: snapshot_dt = np.copy(d_dt)
    
    return pos, snapshot_dt, np.array(energy_history), np.array(dt_mean_history), time.time() - t0, accumulated_dt

# ==========================================
# 4. EXECUTION & DATA EXPORT
# ==========================================
if __name__ == '__main__':
    np.random.seed(42)
    print(f"Initializing {N_ATOMS} particles at phi={TARGET_FRACTION} with a {GRID_DIVS}x{GRID_DIVS} grid...")
    initial_pos = np.random.rand(N_ATOMS, 2)

    print("0. Warming up Numba JIT compiler...")
    _ = get_forces_mpt_numba(initial_pos[:10], np.ones(10), RADIUS, K_SPRING)
    _ = get_total_energy_numba(initial_pos[:10], RADIUS, K_SPRING)

    print(f"\n1. Running Global FIRE (Baseline)...")
    pos_glob, e_hist_glob, dt_hist_glob, time_glob = run_global_fire(initial_pos, MAX_STEPS)
    print(f"Global FIRE Complete in {time_glob:.2f}s.")

    print(f"\n2. Running Async FIRE...")
    pos_async, dt_grid_snap, e_hist_async, dt_hist_async, time_async, accumulated_dt = run_async_fire(initial_pos, MAX_STEPS, CAPTURE_STEP)
    print(f"Async FIRE Complete in {time_async:.2f}s.")

    print("\n3. Serializing Data to Disk...")
    os.makedirs("results", exist_ok=True)
    filename = f"results/PRL_Dataset_N{N_ATOMS}_Grid{GRID_DIVS}.npz"
    np.savez_compressed(filename, 
                        pos_async=pos_async,
                        pos_glob=pos_glob,
                        dt_grid_snap=dt_grid_snap, 
                        e_hist_async=e_hist_async,
                        e_hist_glob=e_hist_glob,
                        dt_hist_async=dt_hist_async,
                        dt_hist_glob=dt_hist_glob,
                        time_async=time_async,
                        time_glob=time_glob,
                        accumulated_dt=accumulated_dt,
                        radius=RADIUS,
                        k_spring=K_SPRING,
                        grid_divs=GRID_DIVS,
                        n_atoms=N_ATOMS,
                        box_size=BOX_SIZE)

    print(f"\nSuccess! Data saved to '{filename}'.")
