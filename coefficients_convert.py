#!/usr/bin/env python3
"""
Compute SVD geometry coefficients from fshapesTk matching outputs.

Reads the geodesic shoot endpoints (1-shoot-1.vtk = cylinder, 1-shoot-16.vtk = registered
target) for each TAA sample, computes the displacement field, and applies SVD reduction
to produce a coefficient_data.npz consumed by the ShapeOperatorLearning training pipeline.
"""

import os
import numpy as np
import argparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm
from svd_utils import SVD_reduce


def load_vtk_points(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VTK file not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    if pts.shape[1] != 3:
        raise ValueError(f"Expected 3D points, got shape {pts.shape}")
    return pts


def main():
    parser = argparse.ArgumentParser(
        description="Compute SVD coefficients from TAA registration matchings"
    )
    parser.add_argument("--template_vtk", type=str,
                        default="./vtk/cylinder.vtk",
                        help="Template (source) VTK — cylinder start of geodesic")
    parser.add_argument("--matchings_dir", type=str,
                        default="./matchings",
                        help="Directory containing sample_XXXXX matching subdirectories")
    parser.add_argument("--target_vtk_name", type=str,
                        default="1-shoot-16.vtk",
                        help="Filename of the registered target inside each matching dir")
    parser.add_argument("--output_file", type=str,
                        default="coefficient_data.npz",
                        help="Output NPZ file path")
    parser.add_argument("--mode", type=int, default=3,
                        help="Number of SVD modes to keep")
    parser.add_argument("--case_range", type=int, nargs=2, default=[0, 999],
                        help="Inclusive range of sample indices to process")
    args = parser.parse_args()

    print(f"Loading template from: {args.template_vtk}")
    template_points = load_vtk_points(args.template_vtk)
    print(f"Template shape: {template_points.shape}")

    dx_list = []
    valid_cases = []
    case_start, case_end = args.case_range

    for idx in tqdm(range(case_start, case_end + 1), desc="Computing displacements"):
        sample_name = f"sample_{idx:05d}"
        target_vtk = os.path.join(args.matchings_dir, sample_name, args.target_vtk_name)
        if not os.path.exists(target_vtk):
            continue
        try:
            target_points = load_vtk_points(target_vtk)
            if template_points.shape != target_points.shape:
                print(f"  Shape mismatch for {sample_name}, skipping")
                continue
            dx_list.append(target_points - template_points)
            valid_cases.append(idx)
        except Exception as e:
            print(f"  Error on {sample_name}: {e}")

    if not valid_cases:
        raise RuntimeError("No valid cases found — check matchings_dir and case_range.")

    print(f"Loaded {len(valid_cases)} cases")

    dx = np.array(dx_list)                      # (n_cases, n_points, 3)
    dx1 = dx[:, :, 0].T                         # (n_points, n_cases)
    dx2 = dx[:, :, 1].T
    dx3 = dx[:, :, 2].T

    print(f"Applying SVD (mode={args.mode})...")
    Ux, coeff_x, Uy, coeff_y, Uz, coeff_z = SVD_reduce(dx1, dx2, dx3, args.mode)

    coefficients = np.concatenate((coeff_x, coeff_y, coeff_z), axis=0).T  # (n_cases, 3*mode)

    np.savez(
        args.output_file,
        coefficients=coefficients,
        case_numbers=np.array(valid_cases),
        l_max=args.mode,
    )

    print(f"Saved {coefficients.shape} coefficient matrix to {args.output_file}")
    print(f"Cases: {len(valid_cases)}  |  Features per case: {coefficients.shape[1]}")


if __name__ == "__main__":
    main()
