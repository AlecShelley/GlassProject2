#!/usr/bin/env python3
"""
Async FIRE minimizer optimized for HPC.

All domain-level FIRE updates are Numba JIT-compiled, reducing per-step
cost from O(K^2 * N) to O(N). K=1 reduces to standard (global) FIRE.

Usage:
    python async_fire_hpc.py --N 1000000 --phi 0.84 --K 8 --D 3 --outdir results/
"""
import argparse
import numpy as np
import time, json, os, sys

ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', max(1, os.cpu_count() - 1)))
os.environ['NUMBA_NUM_THREADS'] = str(ncpu)
sys.stdout.reconfigure(line_buffering=True)

from numba import njit, prange

# ── Physics ──────────────────────────────────────────────────────────
K_SPRING   = 5000.0
BOX        = 1.0
SIZE_RATIO = 1.4

# ── FIRE parameters ─────────────────────────────────────────────────
DT_INIT    = 0.001
DT_MAX     = 0.01
N_MIN      = 5
F_INC      = 1.1
F_DEC      = 0.5
ALPHA0     = 0.1
F_ALPHA    = 0.99
F_TOL      = 1e-6
MAX_STEPS  = 500_000
STAG_LIMIT = 3


def get_radii(N, phi):
    r_small = np.sqrt(2 * phi * BOX**2 / (N * np.pi * (1 + SIZE_RATIO**2)))
    r_large = SIZE_RATIO * r_small
    radii = np.empty(N)
    radii[:N // 2] = r_small
    radii[N // 2:] = r_large
    return radii, 2 * r_large


def lj_scale(phi):
    return np.pi * (0.5**2 + 0.7**2) / (2 * phi)


# ── Numba kernels ────────────────────────────────────────────────────

@njit
def build_cell_list(pos, bs, cutoff):
    n = pos.shape[0]
    nc = max(3, int(np.floor(bs / cutoff)))
    cs = bs / nc
    head = np.full((nc, nc), -1, dtype=np.int32)
    nxt = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        cx = max(0, min(int(pos[i, 0] / cs), nc - 1))
        cy = max(0, min(int(pos[i, 1] / cs), nc - 1))
        nxt[i] = head[cx, cy]
        head[cx, cy] = i
    return head, nxt, nc


@njit(parallel=True)
def compute_forces(pos, radii, k, bs, mcut):
    n = pos.shape[0]
    forces = np.zeros((n, 2))
    head, nxt, nc = build_cell_list(pos, bs, mcut)
    cs = bs / nc
    for i in prange(n):
        cx = max(0, min(int(pos[i, 0] / cs), nc - 1))
        cy = max(0, min(int(pos[i, 1] / cs), nc - 1))
        fx = fy = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = head[(cx + dx) % nc, (cy + dy) % nc]
                while j != -1:
                    if i != j:
                        dvx = pos[i, 0] - pos[j, 0]
                        dvy = pos[i, 1] - pos[j, 1]
                        dvx -= bs * np.round(dvx / bs)
                        dvy -= bs * np.round(dvy / bs)
                        d2 = dvx * dvx + dvy * dvy
                        sig = radii[i] + radii[j]
                        if d2 < sig * sig and d2 > 1e-12:
                            d = np.sqrt(d2)
                            fm = k * (sig - d)
                            fx += (dvx / d) * fm
                            fy += (dvy / d) * fm
                    j = nxt[j]
        forces[i, 0] = fx
        forces[i, 1] = fy
    return forces


@njit(parallel=True)
def compute_energy(pos, radii, k, bs, mcut):
    n = pos.shape[0]
    head, nxt, nc = build_cell_list(pos, bs, mcut)
    cs = bs / nc
    energy = 0.0
    for i in prange(n):
        cx = max(0, min(int(pos[i, 0] / cs), nc - 1))
        cy = max(0, min(int(pos[i, 1] / cs), nc - 1))
        le = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = head[(cx + dx) % nc, (cy + dy) % nc]
                while j != -1:
                    if j > i:
                        dvx = pos[i, 0] - pos[j, 0]
                        dvy = pos[i, 1] - pos[j, 1]
                        dvx -= bs * np.round(dvx / bs)
                        dvy -= bs * np.round(dvy / bs)
                        d2 = dvx * dvx + dvy * dvy
                        sig = radii[i] + radii[j]
                        if d2 < sig * sig:
                            le += 0.5 * k * (sig - np.sqrt(d2))**2
                    j = nxt[j]
        energy += le
    return energy


@njit
def assign_domains(pos, K):
    n = pos.shape[0]
    ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        ix = min(max(int(pos[i, 0] * K), 0), K - 1)
        iy = min(max(int(pos[i, 1] * K), 0), K - 1)
        ids[i] = ix * K + iy
    return ids


@njit
def sync_domain_dt(d_dt, K):
    nd = K * K
    synced = np.empty(nd)
    for gx in range(K):
        for gy in range(K):
            m = d_dt[gx * K + gy]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    m = min(m, d_dt[((gx + dx) % K) * K + (gy + dy) % K])
            synced[gx * K + gy] = m
    return synced


@njit
def domain_fire_stats(forces, vel, domain_ids, nd):
    P = np.zeros(nd)
    v2 = np.zeros(nd)
    f2 = np.zeros(nd)
    for i in range(forces.shape[0]):
        dd = domain_ids[i]
        P[dd] += forces[i, 0] * vel[i, 0] + forces[i, 1] * vel[i, 1]
        v2[dd] += vel[i, 0]**2 + vel[i, 1]**2
        f2[dd] += forces[i, 0]**2 + forces[i, 1]**2
    return P, np.sqrt(v2), np.sqrt(f2)


@njit(parallel=True)
def fire_mix_velocities(vel, forces, domain_ids, d_alpha, v_norm, f_norm):
    for i in prange(vel.shape[0]):
        dd = domain_ids[i]
        fn = f_norm[dd]
        if fn > 1e-12:
            a = d_alpha[dd]
            vn = v_norm[dd]
            vel[i, 0] = (1 - a) * vel[i, 0] + a * (forces[i, 0] / fn) * vn
            vel[i, 1] = (1 - a) * vel[i, 1] + a * (forces[i, 1] / fn) * vn


@njit
def update_fire_params(P, d_dt, d_alpha, d_npos, d_neg, nd,
                       delayed_reset, alpha0, nmin, finc, fdec, dtmax, falpha):
    reset = np.zeros(nd, dtype=np.bool_)
    for dd in range(nd):
        if P[dd] > 0:
            d_npos[dd] += 1
            d_neg[dd] = 0
            if d_npos[dd] > nmin:
                d_dt[dd] = min(d_dt[dd] * finc, dtmax)
                d_alpha[dd] *= falpha
        else:
            d_neg[dd] += 1
            if d_neg[dd] >= max(1, delayed_reset):
                d_npos[dd] = 0
                d_dt[dd] *= fdec
                d_alpha[dd] = alpha0
                d_neg[dd] = 0
                reset[dd] = True
    return reset


@njit(parallel=True)
def zero_reset_vel(vel, domain_ids, reset):
    for i in prange(vel.shape[0]):
        if reset[domain_ids[i]]:
            vel[i, 0] = 0.0
            vel[i, 1] = 0.0


@njit(parallel=True)
def verlet_kick(vel, forces, atom_dt, n):
    for i in prange(n):
        dt = atom_dt[i]
        vel[i, 0] += 0.5 * forces[i, 0] * dt
        vel[i, 1] += 0.5 * forces[i, 1] * dt


@njit(parallel=True)
def verlet_drift(pos, vel, atom_dt, bs, n):
    for i in prange(n):
        dt = atom_dt[i]
        x = pos[i, 0] + vel[i, 0] * dt
        y = pos[i, 1] + vel[i, 1] * dt
        pos[i, 0] = x - bs * np.floor(x / bs)
        pos[i, 1] = y - bs * np.floor(y / bs)


@njit(parallel=True)
def compute_frms_fmax(forces, n):
    sum_f2 = 0.0
    fmax2 = 0.0
    for i in prange(n):
        fi2 = forces[i, 0]**2 + forces[i, 1]**2
        sum_f2 += fi2
        if fi2 > fmax2:
            fmax2 = fi2
    return np.sqrt(sum_f2 / (2 * n)), np.sqrt(fmax2)


# ── Main simulation ─────────────────────────────────────────────────

def run(N, phi, K_GRID, DELAYED_RESET, seed=42):
    radii, mcut = get_radii(N, phi)
    scale = lj_scale(phi)
    nd = K_GRID * K_GRID

    np.random.seed(seed)
    pos = np.random.rand(N, 2) * BOX
    vel = np.zeros((N, 2))

    d_dt = np.full(nd, DT_INIT)
    d_alpha = np.full(nd, ALPHA0)
    d_npos = np.zeros(nd, dtype=np.int64)
    d_neg = np.zeros(nd, dtype=np.int64)

    forces = compute_forces(pos, radii, K_SPRING, BOX, mcut)
    e_init = compute_energy(pos, radii, K_SPRING, BOX, mcut)
    prev_e = None
    stagnant = 0
    history = []
    status = 'max_steps'

    t0 = time.time()
    final_step = MAX_STEPS - 1

    for step in range(MAX_STEPS):
        domain_ids = assign_domains(pos, K_GRID)
        synced_dt = sync_domain_dt(d_dt, K_GRID)
        atom_dt = synced_dt[domain_ids]

        verlet_kick(vel, forces, atom_dt, N)
        verlet_drift(pos, vel, atom_dt, BOX, N)
        forces = compute_forces(pos, radii, K_SPRING, BOX, mcut)
        verlet_kick(vel, forces, atom_dt, N)

        P, v_norm, f_norm = domain_fire_stats(forces, vel, domain_ids, nd)
        fire_mix_velocities(vel, forces, domain_ids, d_alpha, v_norm, f_norm)
        reset = update_fire_params(P, d_dt, d_alpha, d_npos, d_neg, nd,
                                   DELAYED_RESET, ALPHA0, N_MIN,
                                   F_INC, F_DEC, DT_MAX, F_ALPHA)
        zero_reset_vel(vel, domain_ids, reset)

        # Convergence check (cheap — every 200 steps)
        if step % 200 == 0:
            f_rms, f_max = compute_frms_fmax(forces, N)
            if f_rms < F_TOL:
                status = 'converged'
                final_step = step
                break

        # Energy checks (expensive — every 10000 steps + early checks)
        if step in (4000, 10000) or (step > 0 and step % 10000 == 0):
            e = compute_energy(pos, radii, K_SPRING, BOX, mcut)
            if step % 200 != 0:
                f_rms, f_max = compute_frms_fmax(forces, N)

            if step <= 10000 and e > 0.1 * e_init:
                status = 'diverged'
                final_step = step
                history.append({
                    'step': int(step), 'energy': float(e * scale),
                    'f_rms': float(f_rms), 'fmax': float(f_max)
                })
                print(f"  step {step:7d}  DIVERGED  E/N={e * scale:.4e}")
                break

            if prev_e is not None:
                rel = abs(e - prev_e) / max(abs(prev_e), 1e-15)
                if rel < 1e-8:
                    stagnant += 1
                else:
                    stagnant = 0
                if stagnant >= STAG_LIMIT:
                    status = 'stagnated'
                    final_step = step
                    history.append({
                        'step': int(step), 'energy': float(e * scale),
                        'f_rms': float(f_rms), 'fmax': float(f_max)
                    })
                    print(f"  step {step:7d}  STAGNATED  E/N={e * scale:.6e}")
                    break
            prev_e = e

            history.append({
                'step': int(step), 'energy': float(e * scale),
                'f_rms': float(f_rms), 'fmax': float(f_max)
            })
            print(f"  step {step:7d}  E/N_LJ={e * scale:.6e}  |F|={f_rms:.2e}")

    elapsed = time.time() - t0
    e_final = compute_energy(pos, radii, K_SPRING, BOX, mcut)
    f_rms_final, f_max_final = compute_frms_fmax(forces, N)

    if status == 'converged':
        print(f"  step {final_step:7d}  CONVERGED  E/N_LJ={e_final * scale:.6e}  "
              f"|F|={f_rms_final:.2e}  ({elapsed:.1f}s)")

    return {
        'N': N, 'phi': phi, 'K': K_GRID, 'D': DELAYED_RESET,
        'seed': seed,
        'energy_unit': float(e_final),
        'energy_per_n_lj': float(e_final * scale),
        'f_rms': float(f_rms_final),
        'fmax': float(f_max_final),
        'steps': int(final_step),
        'time_s': float(elapsed),
        'status': status,
        'history': history,
        'params': {
            'k_spring': K_SPRING, 'max_steps': MAX_STEPS,
            'f_tol': F_TOL, 'dt_init': DT_INIT, 'dt_max': DT_MAX,
        },
    }


def jit_warmup():
    d = np.random.rand(10, 2)
    r = np.concatenate([np.full(5, 0.05), np.full(5, 0.07)])
    _ = compute_forces(d, r, K_SPRING, BOX, 0.14)
    _ = compute_energy(d, r, K_SPRING, BOX, 0.14)
    ids = assign_domains(d, 2)
    _ = sync_domain_dt(np.full(4, 0.001), 2)
    P, vn, fn = domain_fire_stats(np.zeros((10, 2)), np.zeros((10, 2)), ids, 4)
    fire_mix_velocities(np.zeros((10, 2)), np.ones((10, 2)) * 0.01, ids,
                        np.full(4, 0.1), vn + 1, fn + 1)
    rst = update_fire_params(P, np.full(4, 0.001), np.full(4, 0.1),
                             np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64),
                             4, 0, 0.1, 5, 1.1, 0.5, 0.01, 0.99)
    zero_reset_vel(np.zeros((10, 2)), ids, rst)
    verlet_kick(np.zeros((10, 2)), np.ones((10, 2)) * 0.01, np.full(10, 0.001), 10)
    verlet_drift(np.random.rand(10, 2), np.zeros((10, 2)), np.full(10, 0.001), 1.0, 10)
    _, _ = compute_frms_fmax(np.ones((10, 2)) * 0.01, 10)


def main():
    ap = argparse.ArgumentParser(description='Async FIRE minimizer (HPC)')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--phi', type=float, required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--D', type=int, required=True)
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    outfile = os.path.join(
        args.outdir,
        f'fire_N{args.N}_phi{args.phi:.2f}_K{args.K}_D{args.D}.json'
    )
    if os.path.exists(outfile):
        print(f"SKIP: {outfile} already exists")
        return

    os.makedirs(args.outdir, exist_ok=True)

    print("JIT warmup...")
    jit_warmup()
    print("Ready.\n")

    print(f"N={args.N}  phi={args.phi}  K={args.K}  D={args.D}  "
          f"threads={ncpu}  seed={args.seed}")
    print("-" * 60)

    result = run(args.N, args.phi, args.K, args.D, seed=args.seed)

    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {outfile}")
    print(f"  E/N_LJ = {result['energy_per_n_lj']:.6e}  "
          f"status={result['status']}  steps={result['steps']}  "
          f"time={result['time_s']:.1f}s")


if __name__ == '__main__':
    main()
