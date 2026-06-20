"""
Convert interface.vtp surface meshes (quads) to legacy VTK ASCII format (triangles)
required by fshapesTk's import_fshape_vtk.

Outputs:
  vtk/cylinder.vtk         -- base cylinder template (from base_mesh/)
  vtk/sample_XXXXX.vtk     -- per-sample bulged interface (from samples/)
"""

import os
import glob
import numpy as np
import vtk

VTK_DIR = os.path.join(os.path.dirname(__file__), "vtk")
os.makedirs(VTK_DIR, exist_ok=True)


def read_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def triangulate(polydata):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    return tri.GetOutput()


def write_legacy_vtk(polydata, out_path):
    """Write a triangulated vtkPolyData as legacy ASCII VTK with a zero signal."""
    pts = polydata.GetPoints()
    n_pts = pts.GetNumberOfPoints()
    n_cells = polydata.GetNumberOfCells()

    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("fshape interface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_pts} float\n")
        for i in range(n_pts):
            x, y, z = pts.GetPoint(i)
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        f.write(f"\nPOLYGONS {n_cells} {4 * n_cells}\n")
        for i in range(n_cells):
            cell = polydata.GetCell(i)
            ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            f.write("3 " + " ".join(str(v) for v in ids) + "\n")

        f.write(f"\nPOINT_DATA {n_pts}\n")
        f.write("SCALARS signal float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _ in range(n_pts):
            f.write("0.0\n")


def convert(vtp_path, out_path):
    pd = read_vtp(vtp_path)
    pd = triangulate(pd)
    write_legacy_vtk(pd, out_path)
    print(f"  {os.path.relpath(vtp_path)} -> {os.path.relpath(out_path)}")


if __name__ == "__main__":
    # Base cylinder template
    base_vtp = os.path.join(
        os.path.dirname(__file__),
        "base_mesh", "fluid", "mesh-surfaces", "interface.vtp"
    )
    convert(base_vtp, os.path.join(VTK_DIR, "cylinder.vtk"))

    # Per-sample bulged interfaces
    sample_dirs = sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), "samples", "sample_*")
    ))
    for sd in sample_dirs:
        name = os.path.basename(sd)          # e.g. sample_00001
        vtp = os.path.join(sd, "fluid", "mesh-surfaces", "interface.vtp")
        if not os.path.exists(vtp):
            print(f"  SKIP {name}: no interface.vtp")
            continue
        convert(vtp, os.path.join(VTK_DIR, f"{name}.vtk"))

    print(f"\nDone. VTK files written to {VTK_DIR}/")
