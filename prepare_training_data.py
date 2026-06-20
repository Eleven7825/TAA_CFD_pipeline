#!/usr/bin/env python3
"""
Prepare per-sample training NPZ files for ShapeOperatorLearning.

For each sample:
  1. Load WSS from CFD result (Gaussian-displaced node positions)
  2. Load LDDMM-registered node positions (1-shoot-16.vtk)
  3. Interpolate WSS from CFD nodes onto LDDMM nodes via inverse-distance weighting
  4. Save as processed_TAA_data_{idx}.npz with keys:
       transformed_values  (N_points, 3)  -- WSS at LDDMM positions
       ref_xyz             (N_points, 3)  -- LDDMM node positions
"""

import os
import glob
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
from tqdm import tqdm


def load_vtk_points(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float64)


def idw_interpolate(source_pts, source_vals, target_pts, k=8, power=2):
    """
    Inverse-distance weighted interpolation from source_pts to target_pts.

    Args:
        source_pts : (M, 3) source point cloud
        source_vals: (M, D) values at source points
        target_pts : (N, 3) query points
        k          : number of nearest neighbours
        power      : distance decay exponent

    Returns:
        (N, D) interpolated values
    """
    tree = cKDTree(source_pts)
    dists, idxs = tree.query(target_pts, k=k)

    # Avoid division by zero for exact matches
    dists = np.where(dists == 0, 1e-12, dists)
    weights = 1.0 / dists ** power
    weights /= weights.sum(axis=1, keepdims=True)

    return (weights[:, :, None] * source_vals[idxs]).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate CFD WSS onto LDDMM mesh nodes and save training NPZ files"
    )
    parser.add_argument("--samples_dir", type=str, default="./samples",
                        help="Directory containing sample_XXXXX subdirectories")
    parser.add_argument("--matchings_dir", type=str, default="./matchings",
                        help="Directory containing LDDMM matching results")
    parser.add_argument("--output_dir", type=str, default="./training_data",
                        help="Output directory for processed_TAA_data_N.npz files")
    parser.add_argument("--shoot_frame", type=str, default="1-shoot-16.vtk",
                        help="LDDMM output frame to use as target geometry")
    parser.add_argument("--case_range", type=int, nargs=2, default=[0, 999],
                        help="Inclusive range of sample indices to process")
    parser.add_argument("--idw_k", type=int, default=8,
                        help="Number of nearest neighbours for IDW interpolation")
    parser.add_argument("--idw_power", type=float, default=2.0,
                        help="Distance decay exponent for IDW")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    case_start, case_end = args.case_range
    skipped = 0
    saved = 0

    for idx in tqdm(range(case_start, case_end + 1), desc="Processing samples"):
        sample_name = f"sample_{idx:05d}"

        # --- CFD result ---
        result_path = os.path.join(args.samples_dir, sample_name, "result.npz")
        if not os.path.exists(result_path):
            skipped += 1
            continue

        result = np.load(result_path, allow_pickle=True)
        cfd_coords = result["coords"]   # (672, 3)  Gaussian-displaced positions
        cfd_wss    = result["wss"]      # (672, 3)  wall shear stress

        if cfd_wss.shape[1] < 3:
            print(f"  {sample_name}: unexpected WSS shape {cfd_wss.shape}, skipping")
            skipped += 1
            continue

        # --- LDDMM registered geometry ---
        lddmm_vtk = os.path.join(args.matchings_dir, sample_name, args.shoot_frame)
        if not os.path.exists(lddmm_vtk):
            skipped += 1
            continue

        lddmm_pts = load_vtk_points(lddmm_vtk)   # (672, 3)

        # --- Interpolate WSS from CFD positions to LDDMM positions ---
        wss_at_lddmm = idw_interpolate(
            cfd_coords, cfd_wss, lddmm_pts,
            k=args.idw_k, power=args.idw_power
        )  # (672, 3)

        # --- Save ---
        out_path = os.path.join(args.output_dir, f"processed_TAA_data_{idx}.npz")
        np.savez(
            out_path,
            transformed_values=wss_at_lddmm.astype(np.float64),
            ref_xyz=lddmm_pts.astype(np.float64),
        )
        saved += 1

    print(f"\nDone. Saved {saved} files to {args.output_dir}/  (skipped {skipped})")


if __name__ == "__main__":
    main()
