#!/usr/bin/env python3
"""
FIRE Parameter Sweep — 5x5 Render Script
Reads the 25 .npz files from the compute job and generates the dashboard.

Usage:
  python cell_listfix_plot.py [output_dir]

  output_dir: path to the directory containing sweep_phi*.npz files
              (defaults to "parameter_sweep" in the current directory)

Run locally after copying results from $SCRATCH, or on a Sherlock login node.
"""

import numpy as np
import matplotlib
import os
import sys

if not os.environ.get("DISPLAY") and "WAYLAND_DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ==========================================
# PARAMETERS (must match compute script)
# ==========================================
N_ATOMS = 10000000
BOX_SIZE = 1.0
PHI_SWEEP = [0.80, 0.82, 0.84, 0.86, 0.90]
GAMMA_TARGETS = [0.06, 0.11, 0.17, 0.23, 0.34]


def render_sweep_results(output_dir="parameter_sweep"):
    print("\n" + "=" * 70)
    print("Generating the 5x5 Parameter Sweep Dashboard...")
    print(f"Reading from: {output_dir}/")

    fig, axes = plt.subplots(
        len(PHI_SWEEP), len(GAMMA_TARGETS),
        figsize=(24, 20), sharex="col", sharey="row",
    )
    fig.suptitle(
        f"Async FIRE vs Global FIRE: Energy Convergence vs. Wall-Clock Time (N={N_ATOMS:,})",
        fontsize=22, y=0.95,
    )

    files_found = 0
    for i, phi in enumerate(PHI_SWEEP):
        current_radius = np.sqrt((phi * BOX_SIZE**2) / (N_ATOMS * np.pi))
        k_values = [max(1, int(np.round(g * BOX_SIZE / (8 * current_radius)))) for g in GAMMA_TARGETS]
        k_values = sorted(list(set(k_values)))

        while len(k_values) < len(GAMMA_TARGETS):
            k_values.append(k_values[-1] + 2)

        for j, grid in enumerate(k_values[:len(GAMMA_TARGETS)]):
            ax = axes[i, j]
            filename = os.path.join(output_dir, f"sweep_phi{phi:.2f}_grid{grid}.npz")

            if os.path.exists(filename):
                data = np.load(filename)
                t_glob = data["t_hist_glob"]
                e_glob = data["e_hist_glob"]
                t_async = data["t_hist_async"]
                e_async = data["e_hist_async"]
                gamma_val = float(data["gamma"]) if "gamma" in data else (8 * current_radius * grid) / BOX_SIZE

                ax.plot(t_glob, e_glob, "k--", linewidth=2, label="Global Baseline")
                ax.plot(t_async, e_async, "r-", linewidth=2, label="Async FIRE")

                ax.set_yscale("log")
                ax.set_title(rf"$\phi$={phi:.2f} | K={grid}x{grid} ($\gamma$={gamma_val:.2f})", fontsize=14)
                ax.grid(alpha=0.3)

                if i == len(PHI_SWEEP) - 1:
                    ax.set_xlabel("Wall-Clock Time (s)", fontsize=12)
                if j == 0:
                    ax.set_ylabel("Total Potential Energy", fontsize=12)
                if i == 0 and j == 0:
                    ax.legend(loc="upper right")

                files_found += 1
            else:
                ax.text(0.5, 0.5, "Data Not Generated Yet",
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(rf"$\phi$={phi:.2f} | K={grid}x{grid}", fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    out_pdf = os.path.join(output_dir, "parameter_sweep_results_5x5.pdf")
    out_png = os.path.join(output_dir, "parameter_sweep_results_5x5.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)

    print(f"Rendered {files_found}/25 panels")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "parameter_sweep"
    render_sweep_results(data_dir)
