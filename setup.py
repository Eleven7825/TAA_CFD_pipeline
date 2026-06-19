#!/usr/bin/env python
"""Run once to generate the shared base mesh used by all samples."""

import os
import shutil
from cylinder import generate_mesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MESH = os.path.join(BASE_DIR, "base_mesh")
GEO_FILE  = os.path.join(BASE_DIR, "in_geo", "fsg_full_coarse.json")
MESH_TMP  = os.path.join(BASE_DIR, "mesh_tube_fsi")

if os.path.isdir(BASE_MESH):
    print(f"base_mesh/ already exists at {BASE_MESH} — skipping generation.")
else:
    print("Generating base mesh from fsg_full_coarse.json …")
    os.chdir(BASE_DIR)
    generate_mesh(GEO_FILE)
    shutil.move(MESH_TMP, BASE_MESH)
    print(f"Base mesh written to {BASE_MESH}")
