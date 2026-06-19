"""
Sample a 2-D multivariate-Gaussian aneurysm bump on the fluid-solid interface
and write interface_displacement.dat for the svMultiPhysics lElas mesh solver.

The displacement is purely radial (outward), centred at theta=0, random z.
"""

import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

# Pipe height (must match fsg_full_coarse.json)
HEIGHT = 15.0

# Sampling ranges
A_RANGE       = (0.05, 0.15)   # cm, amplitude
SIGMA_Z_RANGE = (2.0,  5.0)    # cm, axial width
SIGMA_T_RANGE = (0.5,  1.5)    # rad, angular width
Z0_FRAC       = (0.2,  0.8)    # fraction of HEIGHT for aneurysm centre


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

    Parameters
    ----------
    base_mesh_dir : str  path to base_mesh/
    rng           : np.random.Generator

    Returns
    -------
    params : dict  {A, z0, sigma_z, sigma_theta}
    ids    : (N,)  GlobalNodeID of interface nodes
    disp   : (N,3) displacement vectors [dx, dy, dz]
    """
    ids, pts = _read_interface(base_mesh_dir)

    A       = rng.uniform(*A_RANGE)
    z0      = rng.uniform(Z0_FRAC[0] * HEIGHT, Z0_FRAC[1] * HEIGHT)
    sigma_z = rng.uniform(*SIGMA_Z_RANGE)
    sigma_t = rng.uniform(*SIGMA_T_RANGE)

    z     = pts[:, 2]
    theta = np.arctan2(pts[:, 1], pts[:, 0])

    # Wrap angular distance to [-pi, pi]
    d_theta = np.arctan2(np.sin(theta), np.cos(theta))

    d_r = A * np.exp(-0.5 * ((z - z0) / sigma_z) ** 2
                     -0.5 * (d_theta    / sigma_t) ** 2)

    dx = d_r * np.cos(theta)
    dy = d_r * np.sin(theta)
    dz = np.zeros(len(ids))
    disp = np.column_stack([dx, dy, dz])

    params = {"A": A, "z0": z0, "sigma_z": sigma_z, "sigma_theta": sigma_t}
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
        0.0   1.0   2.0                # time points
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
