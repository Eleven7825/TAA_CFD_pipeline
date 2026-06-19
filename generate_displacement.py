"""
Sample a 2-D multivariate-Gaussian aneurysm bump on the fluid-solid interface
and write interface_displacement.dat for the svMultiPhysics lElas mesh solver.

The displacement is purely radial (outward), always centred at theta=0 and
z=HEIGHT/2.  Shape is controlled by a full 2x2 covariance matrix in (z, theta)
space, parameterised as (sigma_z, sigma_theta, rho).
"""

import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

# Geometry constants (must match fsg_full_coarse.json)
HEIGHT  = 15.0
R_INNER = 0.647        # cm
DIAMETER = 2.0 * R_INNER

# Aneurysm centre — fixed at pipe mid-length, theta=0
Z0 = HEIGHT / 2.0     # = 7.5 cm

# Sampling ranges
A_RANGE       = (0.05, 0.6 * DIAMETER)  # cm, amplitude (up to 0.6 × diameter)
SIGMA_Z_RANGE = (1.0,  5.0)             # cm, axial width
SIGMA_T_RANGE = (0.3,  1.5)             # rad, angular width
RHO_RANGE     = (-0.7, 0.7)             # correlation between z and theta


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


def sample_displacement(base_mesh_dir, rng):
    """
    Sample random aneurysm parameters, compute nodal displacements.

    The bump shape is a 2-D Gaussian in (z, theta) with full covariance:

        Sigma = [[sigma_z^2,              rho*sigma_z*sigma_t],
                 [rho*sigma_z*sigma_t,    sigma_t^2          ]]

        d_r = A * exp(-0.5 * v^T Sigma^{-1} v),   v = [z - z0, theta]

    Parameters
    ----------
    base_mesh_dir : str  path to base_mesh/
    rng           : np.random.Generator

    Returns
    -------
    params : dict  {A, sigma_z, sigma_theta, rho}
    ids    : (N,)  GlobalNodeID of interface nodes
    disp   : (N,3) displacement vectors [dx, dy, dz]
    """
    ids, pts = _read_interface(base_mesh_dir)

    A       = rng.uniform(*A_RANGE)
    sigma_z = rng.uniform(*SIGMA_Z_RANGE)
    sigma_t = rng.uniform(*SIGMA_T_RANGE)
    rho     = rng.uniform(*RHO_RANGE)

    z     = pts[:, 2]
    theta = np.arctan2(pts[:, 1], pts[:, 0])

    # Angular distance from theta=0, wrapped to [-pi, pi]
    d_theta = np.arctan2(np.sin(theta), np.cos(theta))

    # Full 2x2 covariance and its inverse
    sz2, st2 = sigma_z ** 2, sigma_t ** 2
    cov_zt   = rho * sigma_z * sigma_t
    det      = sz2 * st2 - cov_zt ** 2          # always > 0 for |rho| < 1
    inv_szz  =  st2 / det
    inv_stt  =  sz2 / det
    inv_szt  = -cov_zt / det

    dz_vec = z - Z0
    exponent = -0.5 * (inv_szz * dz_vec**2
                       + 2.0 * inv_szt * dz_vec * d_theta
                       + inv_stt * d_theta**2)

    d_r = A * np.exp(exponent)

    dx = d_r * np.cos(theta)
    dy = d_r * np.sin(theta)
    dz = np.zeros(len(ids))
    disp = np.column_stack([dx, dy, dz])

    params = {"A": A, "sigma_z": sigma_z, "sigma_theta": sigma_t, "rho": rho}
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
