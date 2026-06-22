#!/usr/bin/env python
"""
Single-sample pipeline driver — one SLURM array task.

Usage:
    python run_sample.py --sample_id 0 [--seed 42] [--out_dir samples/]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from vtk_functions import read_geo, write_geo
from generate_displacement import generate, write_displacement_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SOLVER    = "/build-petsc/svMultiPhysics-build/bin/svmultiphysics"
BASE_MESH = os.path.join(BASE_DIR, "base_mesh")
MESH_XML  = os.path.join(BASE_DIR, "mesh.xml")          # static template
PETSC_DIR = os.path.join(BASE_DIR, "in_petsc")
PRES_DAT  = os.path.join(BASE_DIR, "steady_pressure.dat")


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------
STEADY_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Continue_previous_simulation> 0 </Continue_previous_simulation>
  <Number_of_spatial_dimensions> 3 </Number_of_spatial_dimensions>
  <Number_of_time_steps> 10 </Number_of_time_steps>
  <Time_step_size> 0.01 </Time_step_size>
  <Spectral_radius_of_infinite_time_step> 0.5 </Spectral_radius_of_infinite_time_step>

  <Save_results_to_VTK_format> 1 </Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files> steady </Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files> 1 </Increment_in_saving_VTK_files>
  <Start_saving_after_time_step> 1 </Start_saving_after_time_step>
  <Save_results_in_folder> steady </Save_results_in_folder>

  <Increment_in_saving_restart_files> 1 </Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format> 0 </Convert_BIN_to_VTK_format>

  <Verbose> 0 </Verbose>
  <Warning> 0 </Warning>
  <Debug> 0 </Debug>
</GeneralSimulationParameters>


<Add_mesh name="fluid">
  <Mesh_file_path> {mesh_vtu} </Mesh_file_path>

  <Add_face name="f_interface">
    <Face_file_path> {interface_vtp} </Face_file_path>
  </Add_face>

  <Add_face name="f_start">
    <Face_file_path> {start_vtp} </Face_file_path>
  </Add_face>

  <Add_face name="f_end">
    <Face_file_path> {end_vtp} </Face_file_path>
  </Add_face>
</Add_mesh>


<Add_equation type="fluid">
  <Coupled> true </Coupled>
  <Min_iterations> 1 </Min_iterations>
  <Max_iterations> 10 </Max_iterations>
  <Tolerance> 1e-6 </Tolerance>

  <Density> 1.06e-6 </Density>
  <Viscosity model="Constant">
    <Value> 4e-6 </Value>
  </Viscosity>

  <Output type="Spatial">
    <WSS> true </WSS>
    <Velocity> true </Velocity>
    <Pressure> true </Pressure>
  </Output>

  <LS type="GMRES">
    <Linear_algebra type="petsc">
      <Preconditioner> petsc-rcs </Preconditioner>
      <Configuration_file> {petsc_cfg} </Configuration_file>
    </Linear_algebra>
    <Max_iterations> 500 </Max_iterations>
    <Tolerance> 1e-8 </Tolerance>
    <Krylov_space_dimension> 75 </Krylov_space_dimension>
  </LS>

  <Add_BC name="f_start">
    <Type> Dir </Type>
    <Profile> Parabolic </Profile>
    <Value> -1000.0 </Value>
    <Zero_out_perimeter> false </Zero_out_perimeter>
  </Add_BC>

  <Add_BC name="f_interface">
    <Type> Dir </Type>
    <Time_dependence> Steady </Time_dependence>
    <Value> 0.0 </Value>
    <Zero_out_perimeter> false </Zero_out_perimeter>
  </Add_BC>

  <Add_BC name="f_end">
    <Type> Neu </Type>
    <Time_dependence> Unsteady </Time_dependence>
    <Temporal_values_file_path> {pressure_dat} </Temporal_values_file_path>
  </Add_BC>

  <Add_BC name="f_end">
    <Type> Dir </Type>
    <Time_dependence> Steady </Time_dependence>
    <Value> 0.0 </Value>
    <Effective_direction> (1, 1, 0) </Effective_direction>
    <Impose_on_state_variable_integral> true </Impose_on_state_variable_integral>
    <Zero_out_perimeter> false </Zero_out_perimeter>
  </Add_BC>

</Add_equation>

</svMultiPhysicsFile>
"""


def write_steady_xml(sample_dir):
    fluid_dir  = os.path.join(sample_dir, "fluid")
    surf_dir   = os.path.join(fluid_dir, "mesh-surfaces")
    xml = STEADY_XML_TEMPLATE.format(
        mesh_vtu      = os.path.join(fluid_dir, "mesh-complete.mesh.vtu"),
        interface_vtp = os.path.join(surf_dir,  "interface.vtp"),
        start_vtp     = os.path.join(surf_dir,  "start.vtp"),
        end_vtp       = os.path.join(surf_dir,  "end.vtp"),
        petsc_cfg     = os.path.join(PETSC_DIR, "bcgs.inp"),
        pressure_dat  = PRES_DAT,
    )
    path = os.path.join(sample_dir, "steady_local.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


# ---------------------------------------------------------------------------
# Mesh warping
# ---------------------------------------------------------------------------
def _warp(vtk_data, disp_array):
    """Return a new vtkDataSet warped by disp_array (N,3 numpy)."""
    arr = n2v(disp_array.astype(np.float64))
    arr.SetName("_warp_disp")
    vtk_data.GetPointData().AddArray(arr)
    vtk_data.GetPointData().SetActiveVectors("_warp_disp")
    warp = vtk.vtkWarpVector()
    warp.SetInputData(vtk_data)
    warp.Update()
    out = warp.GetOutput()
    out.GetPointData().RemoveArray("_warp_disp")
    return out


def warp_fluid_mesh(sample_dir, mesh_vtu_path):
    """
    Read the lElas output, extract the Displacement field, warp the base
    fluid mesh and the three boundary surfaces into sample_dir/fluid/.
    """
    # Load displacement from lElas output
    disp_reader = vtk.vtkXMLUnstructuredGridReader()
    disp_reader.SetFileName(mesh_vtu_path)
    disp_reader.Update()
    disp_data = disp_reader.GetOutput()
    disp = v2n(disp_data.GetPointData().GetArray("Displacement"))

    # Warp volume mesh
    fluid_dir = os.path.join(sample_dir, "fluid")
    surf_dir  = os.path.join(fluid_dir, "mesh-surfaces")
    os.makedirs(surf_dir, exist_ok=True)

    base_vol = read_geo(os.path.join(BASE_MESH, "fluid", "mesh-complete.mesh.vtu")).GetOutput()
    warped_vol = _warp(base_vol, disp)
    write_geo(os.path.join(fluid_dir, "mesh-complete.mesh.vtu"), warped_vol)

    # Warp surfaces — map volume displacement to surface nodes via GlobalNodeID
    vol_ids = v2n(base_vol.GetPointData().GetArray("GlobalNodeID"))
    id_to_idx = {int(nid): i for i, nid in enumerate(vol_ids)}

    for surface in ("interface", "start", "end"):
        src = os.path.join(BASE_MESH, "fluid", "mesh-surfaces", f"{surface}.vtp")
        surf_data = read_geo(src).GetOutput()
        surf_ids  = v2n(surf_data.GetPointData().GetArray("GlobalNodeID"))
        surf_disp = np.array([disp[id_to_idx[int(nid)]] for nid in surf_ids])
        warped_surf = _warp(surf_data, surf_disp)
        write_geo(os.path.join(surf_dir, f"{surface}.vtp"), warped_surf)

    return warped_vol


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------
def extract_results(sample_dir, warped_vol):
    """
    Load steady_010.vtu, extract WSS + Pressure at the 672 interface nodes.
    Returns dict ready for np.savez.
    """
    # Interface node indices in the fluid volume (ids_interface == 1)
    iface_flag = v2n(warped_vol.GetPointData().GetArray("ids_interface"))
    iface_idx  = np.where(iface_flag == 1)[0]

    result_vtu = os.path.join(sample_dir, "steady", "steady_010.vtu")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(result_vtu)
    reader.Update()
    data = reader.GetOutput()

    pressure = v2n(data.GetPointData().GetArray("Pressure"))[iface_idx]
    wss      = v2n(data.GetPointData().GetArray("WSS"))[iface_idx]

    pts = v2n(warped_vol.GetPoints().GetData())[iface_idx]
    vol_ids = v2n(warped_vol.GetPointData().GetArray("GlobalNodeID"))
    node_ids = vol_ids[iface_idx]

    return {"pressure": pressure, "wss": wss, "coords": pts, "node_ids": node_ids}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_precomputed_displacement(sample_dir, disp_npz, disp_index):
    """
    Write interface_displacement.dat from a precomputed displacement field
    (e.g. extracted from an FSG tube, possibly augmented), bypassing the Gaussian
    sampler.  Uses the same .dat writer as generate_displacement.

    Returns a params dict describing the source.
    """
    npz = np.load(disp_npz, allow_pickle=True)
    ids = npz["ids"]
    disp = npz["disp"][disp_index]              # (672, 3)
    out = os.path.join(sample_dir, "interface_displacement.dat")
    write_displacement_file(out, ids, disp)
    return {"source": os.path.basename(disp_npz), "disp_index": int(disp_index)}


def run(sample_id, seed, out_dir, disp_npz=None, disp_index=None, mesh_only=False):
    sample_dir = os.path.join(out_dir, f"sample_{sample_id:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Write interface_displacement.dat — either Gaussian-sampled or precomputed
    if disp_npz is not None:
        params = write_precomputed_displacement(sample_dir, disp_npz, disp_index)
        print(f"[{sample_id}] precomputed disp: {params}")
    else:
        params = generate(BASE_MESH, sample_dir, seed=seed)
        print(f"[{sample_id}] params: {params}")

    # 2. Run lElas mesh deformation (cwd = sample_dir; mesh.xml uses ../../ paths)
    ret = subprocess.run([SOLVER, MESH_XML], cwd=sample_dir,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        print(ret.stdout.decode(), ret.stderr.decode(), file=sys.stderr)
        raise RuntimeError(f"Mesh solver failed for sample {sample_id}")

    # 3. Warp fluid mesh using lElas Displacement output
    mesh_vtu = os.path.join(sample_dir, "mesh", "mesh_001.vtu")
    warped_vol = warp_fluid_mesh(sample_dir, mesh_vtu)

    # mesh_only: stop after generating the fluid mesh (CFD runs in a later stage)
    if mesh_only:
        print(f"[{sample_id}] mesh-only done → {sample_dir}/fluid/mesh-complete.mesh.vtu")
        return

    # 4. Write per-sample steady XML and run CFD
    steady_xml = write_steady_xml(sample_dir)
    ret = subprocess.run([SOLVER, steady_xml], cwd=sample_dir,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        print(ret.stdout.decode(), ret.stderr.decode(), file=sys.stderr)
        raise RuntimeError(f"CFD solver failed for sample {sample_id}")

    # 5. Extract interface WSS + Pressure from final time step
    results = extract_results(sample_dir, warped_vol)
    results["params"] = params

    np.savez(os.path.join(sample_dir, "result.npz"), **results)
    print(f"[{sample_id}] done → {sample_dir}/result.npz")

    # 6. Remove large intermediate files to save disk
    if not getattr(args, 'keep', False):
        for pattern in ("mesh/stFile_*.bin", "steady/stFile_*.bin",
                        "steady/steady_00*.vtu"):
            import glob
            for f in glob.glob(os.path.join(sample_dir, pattern)):
                os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out_dir",   type=str,
                        default=os.path.join(BASE_DIR, "samples"))
    parser.add_argument("--keep",      action="store_true",
                        help="keep intermediate VTUs and restart files")
    parser.add_argument("--no_registration", type=str, default=None,
                        help="NPZ with precomputed displacement fields disp (N,672,3); "
                             "use disp[--disp_index] as the interface BC instead of "
                             "sampling a Gaussian bump.")
    parser.add_argument("--disp_index", type=int, default=None,
                        help="Row index into --no_registration disp array.")
    parser.add_argument("--mesh_only", action="store_true",
                        help="Stop after generating the fluid mesh (skip CFD).")
    args = parser.parse_args()

    if args.no_registration is not None and args.disp_index is None:
        parser.error("--no_registration requires --disp_index")

    run(args.sample_id, seed=args.seed + args.sample_id, out_dir=args.out_dir,
        disp_npz=args.no_registration, disp_index=args.disp_index,
        mesh_only=args.mesh_only)
