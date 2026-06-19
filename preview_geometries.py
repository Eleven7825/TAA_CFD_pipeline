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
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_displacement import sample_displacement, Z0, R_INNER, HEIGHT

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_MESH = os.path.join(BASE_DIR, "base_mesh")

CMAP  = "plasma"
VMIN  = 0.0
VMAX  = 0.6 * 2 * R_INNER   # 0.6 × diameter


def _load_interface_grid(base_mesh_dir):
    vtp = os.path.join(base_mesh_dir, "fluid", "mesh-surfaces", "interface.vtp")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp)
    reader.Update()
    pts   = v2n(reader.GetOutput().GetPoints().GetData())
    z     = pts[:, 2]
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    z_unique = np.unique(np.round(z, 6))
    t_unique = np.unique(np.round(theta, 6))
    zi = np.searchsorted(z_unique, np.round(z, 6))
    ti = np.searchsorted(t_unique, np.round(theta, 6))
    return z_unique, t_unique, z, theta, zi, ti


def _build_r_grid(z_nodes, theta_nodes, zi, ti, nz, nt, params):
    """Evaluate deformed radius on the (nz × nt) grid."""
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
    r_full = R_INNER + d_r

    r_grid = np.full((nz, nt), np.nan)
    for idx, (iz, it) in enumerate(zip(zi, ti)):
        r_grid[iz, it] = r_full[idx]
    return r_grid


def _plot_2d(ax, z_unique, t_unique, r_grid, params):
    T_deg = np.degrees(t_unique)
    dr = r_grid - R_INNER
    im = ax.pcolormesh(z_unique, T_deg, dr.T,
                       cmap=CMAP, shading="nearest", vmin=VMIN, vmax=VMAX)
    ax.axhline(0,  color="white", lw=0.6, ls="--", label="θ=0")
    ax.axvline(Z0, color="cyan",  lw=0.6, ls="--", label=f"z={Z0}")
    ax.set_xlabel("z (cm)", fontsize=8)
    ax.set_ylabel("θ (°)",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(_param_str(params), fontsize=8)
    plt.colorbar(im, ax=ax, label="Δr (cm)", pad=0.02)


def _plot_3d(ax, z_unique, t_unique, r_grid, params):
    # Close the cylinder by appending the first theta column at theta+2π
    t_wrap = np.append(t_unique, t_unique[0] + 2 * np.pi)
    r_wrap = np.hstack([r_grid, r_grid[:, [0]]])   # (nz, nt+1)

    Z_g, T_g = np.meshgrid(z_unique, t_wrap, indexing="ij")  # (nz, nt+1)
    X_g = r_wrap * np.cos(T_g)
    Y_g = r_wrap * np.sin(T_g)

    # Color by Δr
    dr_norm = (r_wrap - R_INNER) / VMAX
    dr_norm = np.clip(dr_norm, 0, 1)
    fcolors = matplotlib.colormaps[CMAP](dr_norm)

    ax.plot_surface(X_g, Y_g, Z_g, facecolors=fcolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)

    ax.set_xlabel("x", fontsize=7, labelpad=1)
    ax.set_ylabel("y", fontsize=7, labelpad=1)
    ax.set_zlabel("z (cm)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_box_aspect([1, 1, 3])
    ax.view_init(elev=20, azim=-60)
    ax.set_title(_param_str(params), fontsize=8)


def _param_str(params):
    return (f"A={params['A']:.3f}  σz={params['sigma_z']:.2f}\n"
            f"σθ={params['sigma_theta']:.2f}  ρ={params['rho']:.2f}")


def preview(n, seed, out_path):
    z_unique, t_unique, z_nodes, theta_nodes, zi, ti = _load_interface_grid(BASE_MESH)
    nz, nt = len(z_unique), len(t_unique)

    # Layout: pairs_per_row samples per row, each sample = [2D | 3D]
    pairs_per_row = min(n, 3)
    n_rows = (n + pairs_per_row - 1) // pairs_per_row
    n_cols = pairs_per_row * 2   # each sample takes 2 columns

    fig = plt.figure(figsize=(n_cols * 3.6 + 0.5, n_rows * 4.5 + 1.2))
    fig.suptitle(
        f"Deformed pipe interface  —  z₀={Z0} cm (fixed), θ₀=0 (fixed)\n"
        f"σ_z ∈ [0.5, 2.0] cm   σ_θ ∈ [0.3, 1.5] rad   "
        f"A ∈ [0.05, {VMAX:.3f}] cm   ρ ∈ [−0.7, 0.7]",
        fontsize=10,
    )
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.35)

    rng = np.random.default_rng(seed)

    for k in range(n):
        params, _, _ = sample_displacement(BASE_MESH, rng)
        r_grid = _build_r_grid(z_nodes, theta_nodes, zi, ti, nz, nt, params)

        row = k // pairs_per_row
        col = (k % pairs_per_row) * 2

        ax2d = fig.add_subplot(gs[row, col])
        _plot_2d(ax2d, z_unique, t_unique, r_grid, params)

        ax3d = fig.add_subplot(gs[row, col + 1], projection="3d")
        _plot_3d(ax3d, z_unique, t_unique, r_grid, params)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",  type=str,
                        default=os.path.join(BASE_DIR, "preview.png"))
    args = parser.parse_args()
    preview(args.n, args.seed, args.out)
