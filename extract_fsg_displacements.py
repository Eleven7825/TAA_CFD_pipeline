#!/usr/bin/env python3
"""
Extract inner-surface displacement fields from svFSGe FSG tube meshes.

Each coupling-iteration solid mesh `tube_NNN.vtu` from a real fluid+mesh FSG run
carries a `Displacement` point array and an `ids_interface` mask (672 inner
luminal nodes).  Because the svFSGe tube was meshed by the same `cylinder.py`
generator as our `base_mesh` (r=0.647, h=15, n_cir=32, n_axi=20), its undeformed
interface nodes coincide *exactly* with the base-mesh interface nodes — so the
FSG `Displacement` at those nodes is precisely the displacement from the
reference cylinder, with a built-in node correspondence.  This replaces the
LDDMM registration that previously supplied that correspondence.

For each tube we mask the interface nodes, then reorder their displacement into
base-mesh GlobalNodeID order via an exact nearest-neighbour match on the
undeformed positions.  The stacked result feeds POD (coefficients_convert.py)
and the linear-elasticity fluid-mesh solve (run_sample.py).

Usage:
    python extract_fsg_displacements.py \
        --run_dir /home/shiyi/svFSGe/partitioned_2026-02-23_01-01-05.242473 \
        --output fsg_displacements.npz
"""

import argparse
import glob
import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from scipy.spatial import cKDTree

from generate_displacement import _read_interface

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_MESH = os.path.join(BASE_DIR, "base_mesh")

# Max allowed gap between a base-mesh interface node and its nearest tube
# interface node (cm).  The meshes are identical, so this should be ~0.
MATCH_TOL = 1e-6


def extract_interface_displacement(tube_vtu_path, ref_pts, match_tol=MATCH_TOL):
    """
    Read one tube_NNN.vtu and return the inner-surface displacement reordered to
    the base-mesh interface node order defined by `ref_pts` (672, 3).

    Returns disp (672, 3) where disp[i] is the FSG displacement at the interface
    node whose undeformed position is ref_pts[i].
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(tube_vtu_path)
    reader.Update()
    d = reader.GetOutput()
    pd = d.GetPointData()

    pts = v2n(d.GetPoints().GetData()).astype(np.float64)
    disp = v2n(pd.GetArray("Displacement")).astype(np.float64)
    ids_int = v2n(pd.GetArray("ids_interface"))

    mask = ids_int > 0.5
    tube_ref = pts[mask]        # undeformed interface positions (672, 3)
    tube_disp = disp[mask]      # interface displacement (672, 3)

    if tube_ref.shape[0] != ref_pts.shape[0]:
        raise ValueError(
            f"{os.path.basename(tube_vtu_path)}: interface has "
            f"{tube_ref.shape[0]} nodes, base mesh has {ref_pts.shape[0]}"
        )

    # Exact correspondence: nearest tube node to each base-mesh node.
    tree = cKDTree(tube_ref)
    dist, idx = tree.query(ref_pts, k=1)
    if dist.max() > match_tol:
        raise ValueError(
            f"{os.path.basename(tube_vtu_path)}: max NN distance "
            f"{dist.max():.3e} exceeds tol {match_tol:.0e} — tube mesh does not "
            f"match base_mesh interface (different discretization?)"
        )
    if len(np.unique(idx)) != ref_pts.shape[0]:
        raise ValueError(
            f"{os.path.basename(tube_vtu_path)}: NN match is not a bijection "
            f"({len(np.unique(idx))} unique of {ref_pts.shape[0]})"
        )

    return tube_disp[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Extract inner-surface displacement fields from FSG tube meshes"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="svFSGe partitioned_* run directory (contains partitioned/tube_*.vtu)",
    )
    parser.add_argument(
        "--base_mesh", type=str, default=DEFAULT_BASE_MESH,
        help="TAA base_mesh directory (defines interface node order)",
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(BASE_DIR, "fsg_displacements.npz"),
        help="Output NPZ path",
    )
    args = parser.parse_args()

    pdir = os.path.join(args.run_dir, "partitioned")
    if not os.path.isdir(pdir):
        # Allow passing the partitioned/ dir directly too.
        pdir = args.run_dir
    tube_files = sorted(glob.glob(os.path.join(pdir, "tube_*.vtu")))
    if not tube_files:
        raise FileNotFoundError(f"No tube_*.vtu found in {pdir}")

    ids, ref_pts = _read_interface(args.base_mesh)
    ids = np.asarray(ids)
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    print(f"Base-mesh interface: {ref_pts.shape[0]} nodes")
    print(f"Found {len(tube_files)} tube meshes in {pdir}\n")

    disp_list = []
    used_files = []
    for tube in tube_files:
        name = os.path.basename(tube)
        try:
            disp = extract_interface_displacement(tube, ref_pts)
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            continue
        mag = np.linalg.norm(disp, axis=1)
        print(f"  {name}: 672 nodes  |  |disp| max={mag.max():.5f}  mean={mag.mean():.5f} cm")
        disp_list.append(disp)
        used_files.append(name)

    if not disp_list:
        raise RuntimeError("No tube meshes could be processed.")

    disp_arr = np.stack(disp_list).astype(np.float64)   # (N, 672, 3)
    np.savez(
        args.output,
        disp=disp_arr,
        ids=ids,
        ref_xyz=ref_pts,
        source_files=np.array(used_files),
    )
    print(f"\nSaved {disp_arr.shape} displacement array to {args.output}")
    print(f"  ({len(used_files)} samples, {disp_arr.shape[1]} interface nodes)")


if __name__ == "__main__":
    main()
