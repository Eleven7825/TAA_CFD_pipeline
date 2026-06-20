"""
Sample superimposed 2-D multivariate-Gaussian radial bumps on the fluid-solid
interface and write interface_displacement.dat for the svMultiPhysics lElas
mesh solver.

N_bumps ~ Uniform(1, N_BUMPS_MAX) bumps are summed per sample.
Each bump has independent (A, sigma_z, sigma_theta, rho, z0, theta0).
Positive A → aneurysm (outward).  Negative A → stenosis (inward).
The total radial displacement is clamped so the lumen never collapses.
"""

import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

# Geometry constants (must match fsg_full_coarse.json)
HEIGHT  = 15.0
R_INNER = 0.647        # cm
DIAMETER = 2.0 * R_INNER

# Axial centre: Gaussian around pipe mid-point, std = Z0_SIGMA cm
Z0_MEAN  = HEIGHT / 2.0   # = 7.5 cm
Z0_SIGMA = 1.0             # cm  — controls how off-centre the bump can be

# Sampling ranges
# Negative A → stenosis; positive A → aneurysm.
# Lower bound capped at -0.6*R_INNER so the lumen never collapses.
A_RANGE       = (-0.6 * R_INNER, 0.6 * DIAMETER)  # cm  ≈ (-0.39, 0.78)
SIGMA_Z_RANGE = (0.5,  2.0)             # cm, axial width
SIGMA_T_RANGE = (0.3,  1.5)             # rad, angular width
RHO_RANGE     = (-0.7, 0.7)             # correlation between z and theta

# Multi-bump: number of bumps per sample drawn from Uniform(1, N_BUMPS_MAX)
N_BUMPS_MAX = 3

# Minimum allowed inner radius after summing bumps (prevents lumen collapse)
R_MIN = 0.2 * R_INNER


def _read_interface(base_mesh_dir):
    """Return (GlobalNodeID, points) arrays for the interface surface."""
    vtp = os.path.join(base_mesh_dir, "fluid", "mesh-surfaces", "interface.vtp")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp)
    reader.Update()
    data = reader.GetOutput()
    ids  = v2n(data.GetPointData().GetArray("GlobalNodeID"))
    pts  = v2n(data.GetPoints().GetData())
    return ids, pts


def _single_bump_dr(z, theta, rng):
    """Sample one bump and return (d_r, bump_params)."""
    A       = rng.uniform(*A_RANGE)
    sigma_z = rng.uniform(*SIGMA_Z_RANGE)
    sigma_t = rng.uniform(*SIGMA_T_RANGE)
    rho     = rng.uniform(*RHO_RANGE)

    z0     = float(np.clip(rng.normal(Z0_MEAN, Z0_SIGMA), sigma_z, HEIGHT - sigma_z))
    theta0 = rng.uniform(-np.pi, np.pi)

    d_theta = np.arctan2(np.sin(theta - theta0), np.cos(theta - theta0))
    dz_vec  = z - z0

    sz2, st2 = sigma_z ** 2, sigma_t ** 2
    cov_zt   = rho * sigma_z * sigma_t
    det      = sz2 * st2 - cov_zt ** 2
    inv_szz  =  st2 / det
    inv_stt  =  sz2 / det
    inv_szt  = -cov_zt / det

    exponent = -0.5 * (inv_szz * dz_vec**2
                       + 2.0 * inv_szt * dz_vec * d_theta
                       + inv_stt * d_theta**2)
    d_r = A * np.exp(exponent)
    params = {"A": A, "sigma_z": sigma_z, "sigma_theta": sigma_t, "rho": rho,
              "z0": z0, "theta0": theta0}
    return d_r, params


def sample_displacement(base_mesh_dir, rng):
    """
    Sample N_bumps superimposed Gaussian bumps, compute nodal displacements.

    N_bumps ~ Uniform(1, N_BUMPS_MAX).  Each bump has independent parameters.
    The summed radial displacement is clamped so r >= R_MIN (no lumen collapse).

    Returns
    -------
    params : dict  {"bumps": [list of per-bump param dicts], "n_bumps": int}
    ids    : (N,)  GlobalNodeID of interface nodes
    disp   : (N,3) displacement vectors [dx, dy, dz]
    """
    ids, pts = _read_interface(base_mesh_dir)

    z     = pts[:, 2]
    theta = np.arctan2(pts[:, 1], pts[:, 0])

    n_bumps = int(rng.integers(1, N_BUMPS_MAX + 1))
    d_r_total = np.zeros(len(ids))
    bump_params = []
    for _ in range(n_bumps):
        d_r, bp = _single_bump_dr(z, theta, rng)
        d_r_total += d_r
        bump_params.append(bp)

    # Clamp so lumen never collapses
    r_deformed  = np.clip(R_INNER + d_r_total, R_MIN, None)
    d_r_clamped = r_deformed - R_INNER

    dx = d_r_clamped * np.cos(theta)
    dy = d_r_clamped * np.sin(theta)
    dz = np.zeros(len(ids))
    disp = np.column_stack([dx, dy, dz])

    params = {"bumps": bump_params, "n_bumps": n_bumps}
    return params, ids, disp


def write_displacement_file(path, ids, disp):
    """
    Write interface_displacement.dat consumed by svMultiPhysics General BC.

    Uses 3 time points [0, 1, 2] so period=2.0.  After one step (dt=1.0),
    com_mod.time=1.0 and igbc() computes fmod(1.0, 2.0)=1.0, which correctly
    interpolates the full displacement.  With period=1.0 (2 points), fmod
    would give 0.0, returning zero displacement every time.

    Format (read_files.cpp::read_temp_spat_values):
        ndof  num_timesteps  num_nodes
        0.0   1.0   2.0                # time points (on separate lines)
        <GlobalNodeID>
        0.0  0.0  0.0                  # displacement at t=0
        dx   dy   dz                   # displacement at t=1
        dx   dy   dz                   # displacement at t=2 (held constant)
        ... repeat for all nodes
    """
    n = len(ids)
    with open(path, "w") as f:
        f.write(f"3 3 {n}\n")
        f.write("0.0\n1.0\n2.0\n")
        for nid, (dx, dy, dz) in zip(ids, disp):
            f.write(f"{nid}\n")
            f.write(f"0.0 0.0 0.0\n")
            f.write(f"{dx:.10f} {dy:.10f} {dz:.10f}\n")
            f.write(f"{dx:.10f} {dy:.10f} {dz:.10f}\n")


def generate(base_mesh_dir, sample_dir, seed):
    """
    Top-level function: sample, write file, return params.

    Parameters
    ----------
    base_mesh_dir : str
    sample_dir    : str  destination directory (must exist)
    seed          : int

    Returns
    -------
    params : dict
    """
    rng = np.random.default_rng(seed)
    params, ids, disp = sample_displacement(base_mesh_dir, rng)
    out = os.path.join(sample_dir, "interface_displacement.dat")
    write_displacement_file(out, ids, disp)
    return params
