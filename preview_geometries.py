"""
Preview deformed interface geometries without running CFD.

The interface displacement is purely prescribed (Dirichlet), so we can
evaluate the deformed surface directly from the sampler — no solver needed.

Usage:
    python preview_geometries.py [--n N] [--seed S] [--out preview.png]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_displacement import sample_displacement, Z0, R_INNER, HEIGHT

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_MESH = os.path.join(BASE_DIR, "base_mesh")


def _load_interface_grid(base_mesh_dir):
    """Return sorted unique (z_vals, theta_vals) and the index grid."""
    vtp = os.path.join(base_mesh_dir, "fluid", "mesh-surfaces", "interface.vtp")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp)
    reader.Update()
    pts = v2n(reader.GetOutput().GetPoints().GetData())

    z     = pts[:, 2]
    theta = np.arctan2(pts[:, 1], pts[:, 0])

    z_unique = np.unique(np.round(z, 6))
    t_unique = np.unique(np.round(theta, 6))
    return z_unique, t_unique, z, theta


def _deformed_r(z_nodes, theta_nodes, params):
    """Radial displacement field for all interface nodes given params dict."""
    A       = params["A"]
    sigma_z = params["sigma_z"]
    sigma_t = params["sigma_theta"]
    rho     = params["rho"]

    dz_vec  = z_nodes - Z0
    d_theta = np.arctan2(np.sin(theta_nodes), np.cos(theta_nodes))

    sz2, st2 = sigma_z**2, sigma_t**2
    cov      = rho * sigma_z * sigma_t
    det      = sz2 * st2 - cov**2
    inv_szz  =  st2 / det
    inv_stt  =  sz2 / det
    inv_szt  = -cov / det

    exponent = -0.5 * (inv_szz * dz_vec**2
                       + 2.0 * inv_szt * dz_vec * d_theta
                       + inv_stt * d_theta**2)
    d_r = A * np.exp(exponent)
    return R_INNER + d_r


def preview(n, seed, out_path):
    z_unique, t_unique, z_nodes, theta_nodes = _load_interface_grid(BASE_MESH)

    nz, nt = len(z_unique), len(t_unique)
    # Map each node to grid position
    zi = np.searchsorted(z_unique, np.round(z_nodes, 6))
    ti = np.searchsorted(t_unique, np.round(theta_nodes, 6))

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4 * ncols + 1, 3.5 * nrows + 1.2))
    fig.suptitle(
        f"Deformed interface  (z fixed at {Z0} cm, θ=0)\n"
        f"σ_z ∈ [{0.5},{2.0}] cm   σ_θ ∈ [{0.3},{1.5}] rad   "
        f"A ∈ [0.05, 0.776] cm   ρ ∈ [−0.7, 0.7]",
        fontsize=10, y=1.01,
    )
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.45, wspace=0.35)

    rng = np.random.default_rng(seed)
    vmax = R_INNER * 1.65   # headroom for large bumps

    for k in range(n):
        params, _, _ = sample_displacement(BASE_MESH, rng)

        r_grid = np.full((nz, nt), np.nan)
        r_full = _deformed_r(z_nodes, theta_nodes, params)
        for idx, (iz, it) in enumerate(zip(zi, ti)):
            r_grid[iz, it] = r_full[idx]

        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        T_deg = np.degrees(t_unique)
        im = ax.pcolormesh(
            z_unique, T_deg, (r_grid - R_INNER).T,
            cmap="plasma", shading="nearest",
            vmin=0.0, vmax=vmax - R_INNER,
        )
        ax.axhline(0, color="white", lw=0.6, ls="--")  # θ=0 line
        ax.axvline(Z0, color="cyan", lw=0.6, ls="--")  # z=z0=7.5
        ax.set_xlabel("z (cm)", fontsize=8)
        ax.set_ylabel("θ (°)", fontsize=8)
        ax.tick_params(labelsize=7)
        A_str  = f"A={params['A']:.3f}"
        sz_str = f"σz={params['sigma_z']:.2f}"
        st_str = f"σθ={params['sigma_theta']:.2f}"
        rho_str = f"ρ={params['rho']:.2f}"
        ax.set_title(f"{A_str}  {sz_str}\n{st_str}  {rho_str}", fontsize=8)
        fig.colorbar(im, ax=ax, label="Δr (cm)", pad=0.02)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",  type=str,
                        default=os.path.join(BASE_DIR, "preview.png"))
    args = parser.parse_args()
    preview(args.n, args.seed, args.out)
