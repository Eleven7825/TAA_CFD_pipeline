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
from generate_displacement import sample_displacement, Z0_MEAN, Z0_SIGMA, R_INNER, HEIGHT

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_MESH = os.path.join(BASE_DIR, "base_mesh")

CMAP3D = "coolwarm"             # diverging for 3D (white→visible on dark bg)
PANE_COLOR = (0.12, 0.12, 0.12, 1.0)   # dark background for 3D panes
VLIM  = 0.6 * 2 * R_INNER      # 0.6 × diameter  ≈ 0.776 cm
VMIN  = -VLIM
VMAX  =  VLIM


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


def _bump_dr_grid(z_nodes, theta_nodes, bp):
    """Evaluate one bump's d_r on the 1-D node arrays."""
    dz_vec  = z_nodes - bp["z0"]
    d_theta = np.arctan2(np.sin(theta_nodes - bp["theta0"]),
                         np.cos(theta_nodes - bp["theta0"]))
    sz2 = bp["sigma_z"]**2
    st2 = bp["sigma_theta"]**2
    cov = bp["rho"] * bp["sigma_z"] * bp["sigma_theta"]
    det = sz2 * st2 - cov**2
    exponent = -0.5 * ((st2 / det) * dz_vec**2
                       + 2.0 * (-cov / det) * dz_vec * d_theta
                       + (sz2 / det) * d_theta**2)
    return bp["A"] * np.exp(exponent)


def _build_r_grid(z_nodes, theta_nodes, zi, ti, nz, nt, params):
    """Evaluate deformed radius on the (nz × nt) grid, summing all bumps."""
    from generate_displacement import R_MIN
    d_r_total = np.zeros(len(z_nodes))
    for bp in params["bumps"]:
        d_r_total += _bump_dr_grid(z_nodes, theta_nodes, bp)
    r_full = np.clip(R_INNER + d_r_total, R_MIN, None)

    r_grid = np.full((nz, nt), np.nan)
    for idx, (iz, it) in enumerate(zip(zi, ti)):
        r_grid[iz, it] = r_full[idx]
    return r_grid


def _plot_3d(ax, z_unique, t_unique, r_grid, params):
    # Close the cylinder by appending the first theta column at theta+2π
    t_wrap = np.append(t_unique, t_unique[0] + 2 * np.pi)
    r_wrap = np.hstack([r_grid, r_grid[:, [0]]])   # (nz, nt+1)

    Z_g, T_g = np.meshgrid(z_unique, t_wrap, indexing="ij")  # (nz, nt+1)
    X_g = r_wrap * np.cos(T_g)
    Y_g = r_wrap * np.sin(T_g)

    # Color by Δr (diverging: cool = stenosis, warm = aneurysm)
    dr_norm = (r_wrap - R_INNER - VMIN) / (VMAX - VMIN)
    dr_norm = np.clip(dr_norm, 0, 1)
    fcolors = matplotlib.colormaps[CMAP3D](dr_norm)

    ax.plot_surface(X_g, Y_g, Z_g, facecolors=fcolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True,
                    shade=True)

    # Dark pane backgrounds so the cylinder is visible
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(PANE_COLOR)
        pane.set_edgecolor("none")
    ax.set_facecolor(PANE_COLOR)

    # Light tick/label colours for contrast against dark background
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
        axis.set_tick_params(colors="white")

    ax.set_xlabel("x (cm)", fontsize=7, labelpad=1)
    ax.set_ylabel("y (cm)", fontsize=7, labelpad=1)
    ax.set_zlabel("z (cm)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, colors="white")
    # Physical aspect: x/y span ≈ 2*R_INNER, z span = HEIGHT
    r_max = np.nanmax(r_grid)
    ax.set_box_aspect([2 * r_max, 2 * r_max, HEIGHT])
    ax.view_init(elev=20, azim=-60)
    ax.set_title(_param_str(params), fontsize=8)


def _param_str(params):
    n = params["n_bumps"]
    lines = [f"{n} bump{'s' if n > 1 else ''}"]
    for i, bp in enumerate(params["bumps"]):
        lines.append(f"  [{i+1}] A={bp['A']:.2f} z₀={bp['z0']:.1f} θ₀={np.degrees(bp['theta0']):.0f}°"
                     f" σz={bp['sigma_z']:.1f} σθ={bp['sigma_theta']:.1f}")
    return "\n".join(lines)


def preview(n, seed, out_path):
    z_unique, t_unique, z_nodes, theta_nodes, zi, ti = _load_interface_grid(BASE_MESH)
    nz, nt = len(z_unique), len(t_unique)

    # Layout: one 3D view per sample
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 3.2 + 0.5, n_rows * 8.0 + 1.2))
    fig.suptitle(
        f"Deformed pipe interface  —  "
        f"z₀ ~ N({Z0_MEAN}, {Z0_SIGMA}²) cm   θ₀ ~ U(−π, π)\n"
        f"σ_z ∈ [0.5, 2.0] cm   σ_θ ∈ [0.3, 1.5] rad   "
        f"A ∈ [{VMIN:.3f}, {VMAX:.3f}] cm   ρ ∈ [−0.7, 0.7]"
        f"   (blue = stenosis, red = aneurysm)",
        fontsize=10,
    )
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.25, wspace=0.15)

    rng = np.random.default_rng(seed)

    for k in range(n):
        params, _, _ = sample_displacement(BASE_MESH, rng)
        r_grid = _build_r_grid(z_nodes, theta_nodes, zi, ti, nz, nt, params)

        row = k // n_cols
        col = k % n_cols

        ax3d = fig.add_subplot(gs[row, col], projection="3d")
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
