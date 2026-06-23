# TAA_CFD_pipeline

Synthetic CFD data generation for Thoracic Aortic Aneurysm (TAA) geometries,
intended for training neural operators (see `ShapeOperatorLearning`).

The pipeline proceeds in five stages:

```
1. Base mesh          →  2. Sample generation  →  3. CFD simulation
        ↓
4. LDDMM registration  →  5. Training data preparation
```

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python 3 + `vtk`, `numpy`, `scipy`, `tqdm` | geometry processing & data prep |
| MATLAB R2026a + fshapesTk | LDDMM registration |
| CUDA-enabled GPU | fshapesTk CUDA kernels |
| svMultiPhysics | CFD solver |

MATLAB must be able to find `libcudart.so.11.0`. The CUDA MEX binaries in
`fshapesTk/Bin/kernels/binaries/` have been patched (via `patchelf`) to load
it from `~/miniconda3/envs/lbm/lib/` automatically.

---

## Stage 1 — Base mesh

Run once to build the shared cylinder mesh:

```bash
python setup.py
```

Output: `base_mesh/` — structured hexahedral cylinder (inner radius 0.647 cm,
height 15 cm, 20 × 32 × 25 fluid elements).

---

## Stage 2 — Sample generation

Each sample applies a parameterised Gaussian radial bump to the fluid-solid
interface, creating aneurysm (positive amplitude) or stenosis (negative)
variants.

```bash
python generate_displacement.py   # writes samples/sample_XXXXX/interface_displacement.dat
```

Geometry parameters per sample (sampled randomly):
- Amplitude *A* ∈ [−0.39, 1.29] cm
- Axial width *σ_z* ∈ [0.5, 2.0] cm
- Angular width *σ_θ* ∈ [0.3, 1.5] rad
- Correlation *ρ* ∈ [−0.7, 0.7]

---

## Stage 3 — CFD simulation

Run the FSI solver for each sample (single sample):

```bash
python run_sample.py --sample <N>
```

Or submit a SLURM array job for a batch:

```bash
sbatch submit.sh   # array 0-999 by default
```

Each completed sample produces:
- `samples/sample_XXXXX/result.npz` — interface WSS `(672, 3)` and pressure `(672,)` at Gaussian-displaced nodes
- `samples/sample_XXXXX/steady/steady_010.vtu` — full volumetric CFD solution

---

## Stage 4 — LDDMM geometric registration

Registers the base cylinder to each sample's bulged interface using LDDMM
(fshapesTk). This produces a diffeomorphic deformation field whose SVD
coefficients encode the geometry for the neural operator.

### 4a — Convert interface meshes to legacy VTK format

Required once (or whenever new samples are added):

```bash
python convert_interfaces_to_vtk.py
```

Output: `vtk/cylinder.vtk` (template) and `vtk/sample_XXXXX.vtk` (one per sample).

### 4b — Run LDDMM registration in MATLAB

Full batch (all samples):

```bash
/home/shiyi/matlab/bin/matlab -sd /home/shiyi/TAA_CFD_pipeline \
    -batch "script_TAA_matching_geom"
```

Or a specific range (e.g. first 600 samples) in the background:

```bash
nohup /home/shiyi/matlab/bin/matlab -sd /home/shiyi/TAA_CFD_pipeline \
    -batch "script_TAA_matching_600" \
    > ./matchings/run_600.log 2>&1 &
```

Key registration parameters (edit `script_TAA_matching_geom.m` to tune):

| Parameter | Value | Description |
|---|---|---|
| `kernel_size_mom` | `[0.3, 0.2]` | Deformation kernel sizes (cm) |
| `nb_euler_steps` | 15 | Geodesic integration steps |
| `kernel_size_geom` | 0.3 | Varifold geometric kernel (cm) |
| `bfgs.maxit` | 30 | Max BFGS iterations per sample |

Each completed matching produces `matchings/sample_XXXXX/1-shoot-16.vtk` —
the cylinder mesh deformed to the registered target geometry.

---

## Stage 5 — Training data preparation

### 5a — Interpolate CFD WSS onto LDDMM mesh nodes

The CFD result lives on Gaussian-displaced nodes; the neural operator expects
values at LDDMM-registered nodes. This step bridges the two via
inverse-distance weighted interpolation.

```bash
python prepare_training_data.py \
    --case_range 0 599 \
    --output_dir ./training_data
```

Output per sample: `training_data/processed_TAA_data_N.npz`

| Key | Shape | Description |
|---|---|---|
| `transformed_values` | `(672, 3)` | WSS (Fx, Fy, Fz) at LDDMM nodes |
| `ref_xyz` | `(672, 3)` | LDDMM node positions |

Samples with NaN or all-zero WSS are skipped automatically.

### 5b — Compute SVD geometry coefficients

```bash
python coefficients_convert.py \
    --case_range 0 599 \
    --mode 8 \
    --output_file ./coefficient_data_m8.npz
```

Output: `coefficient_data_m8.npz` — SVD coefficient matrix `(N_cases, 3×mode)`
that encodes the geometry of each sample as input to the neural operator.
Also saves `POD_mode_frac.png` showing the energy fraction per SVD mode.

---

## Alternative pipeline — FSG direct displacement (no LDDMM)

When the geometry comes from an **svFSGe FSG run** (true fluid + mesh solver),
LDDMM registration (Stage 4) is unnecessary. Each `tube_NNN.vtu` was meshed by
the same `cylinder.py` generator as `base_mesh`, so its inner-surface nodes
coincide *exactly* with the base-mesh interface nodes — the FSG `Displacement`
array already *is* the displacement-from-cylinder, with a built-in node
correspondence. This replaces Stages 4–5 with a registration-free,
interpolation-free path.

```
tube_NNN.vtu  ─[1]─► fsg_displacements.npz ─[2]─► augmented_displacements.npz
                                                   ├─[3]─► coefficient_data_fsg_m8.npz (+ _basis)
                                                   └─[4]─► interface_displacement.dat
                                                            └─[5 CFD]─► result.npz
                                                                         └─[6 dl][7]─► training_data_fsg/
```

| Step | Command | Output |
|---|---|---|
| 1 | `python extract_fsg_displacements.py --run_dir <svFSGe partitioned_* run>` | `fsg_displacements.npz` `(N,672,3)` |
| 2 | `python augment_displacements.py --n_aug 500 --alpha 0.3 --scale_range 0.5 1.5` | `augmented_displacements.npz` `(N',672,3)` |
| 3 | `python coefficients_convert.py --no_registration augmented_displacements.npz --mode 8 --output_file coefficient_data_fsg_m8.npz` | coeffs `(N',24)` + `_basis.npz` |
| 4 | `python run_sample.py --sample_id K --no_registration augmented_displacements.npz --disp_index K --mesh_only` | per-sample `interface_displacement.dat` (+ lElas fluid mesh) |
| 5 | (HPC) `sbatch --array=... submit_cfd_array.sh` → `run_sample.py --hpc --skip-geom` | `samples/sample_NNNNN/result.npz` |
| 6 | `rsync` results back locally → `fsg_results/` | downloaded `result.npz` |
| 7 | `python prepare_training_data_direct.py` | `training_data_fsg/processed_TAA_data_*.npz` + `coefficient_data_fsg_m8_aligned.npz` |

Key differences from the LDDMM path:

- **Stage 4 (LDDMM) is skipped.** Geometry correspondence is exact by
  construction; `extract_fsg_displacements.py` asserts it (max NN distance ≈ 0).
- **`coefficients_convert.py --no_registration`** reads displacement fields
  directly from an NPZ instead of `matchings/1-shoot-16.vtk`; the SVD/output
  format is identical.
- **`run_sample.py --no_registration --disp_index`** writes the lElas BC from a
  precomputed displacement instead of sampling a Gaussian bump; `--mesh_only`
  stops after the fluid mesh. On the cluster, `--hpc --skip-geom` consumes an
  existing `interface_displacement.dat`.
- **`prepare_training_data_direct.py`** replaces `prepare_training_data.py`:
  because CFD WSS and the POD encoding share the same 672 base-mesh nodes, it
  only reorders into the canonical node order — **no IDW interpolation**.

Augmentation (step 2) synthesizes new fields as `s · Σ_k w_k d_k` with Dirichlet
weights (`--alpha`, smaller = sparser/more amplitude-diverse) and a uniform
scale (`--scale_range`). FSG coupling sub-iterations lie on one growth
trajectory, so the POD is near rank-1 and augmentation mainly varies amplitude.

## Directory structure

```
TAA_CFD_pipeline/
├── base_mesh/                  # shared cylinder mesh (gitignored)
├── samples/                    # per-sample CFD inputs & results (gitignored)
│   └── sample_XXXXX/
│       ├── result.npz          # WSS + pressure at interface
│       └── steady/steady_010.vtu
├── vtk/                        # converted legacy VTK meshes (gitignored)
├── matchings/                  # LDDMM registration outputs (gitignored)
│   └── sample_XXXXX/
│       ├── 1-shoot-1.vtk       # cylinder start of geodesic
│       └── 1-shoot-16.vtk      # registered target geometry
├── training_data/              # prepared training NPZ files (gitignored)
├── training_data_fsg/          # FSG direct-displacement training NPZs (gitignored)
├── fsg_results/                # result.npz downloaded from cluster (gitignored)
├── convert_interfaces_to_vtk.py
├── script_TAA_matching_geom.m  # full-batch LDDMM script
├── script_TAA_matching_600.m   # 600-sample batch script
├── prepare_training_data.py    # CFD→LDDMM WSS interpolation
├── coefficients_convert.py     # SVD geometry coefficient computation
├── svd_utils.py                # SVD reduction utility
│   # --- FSG direct-displacement path (no LDDMM) ---
├── extract_fsg_displacements.py  # tube_*.vtu inner-surface displacement
├── augment_displacements.py      # random linear-combination augmentation
└── prepare_training_data_direct.py  # CFD result → training NPZs (no interpolation)
```

---

## Quick-start for new data

To add more samples end-to-end:

```bash
# 1. Generate displacements for new samples
python generate_displacement.py

# 2. Run CFD
sbatch submit.sh

# 3. Convert interfaces
python convert_interfaces_to_vtk.py

# 4. Register (MATLAB, background)
nohup /home/shiyi/matlab/bin/matlab -sd $(pwd) \
    -batch "script_TAA_matching_geom" > matchings/run.log 2>&1 &

# 5. Prepare training data
python prepare_training_data.py --case_range 0 999 --output_dir ./training_data

# 6. Recompute SVD coefficients over the full set
python coefficients_convert.py --case_range 0 999 --mode 8 \
    --output_file ./coefficient_data_m8.npz
```
