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
├── convert_interfaces_to_vtk.py
├── script_TAA_matching_geom.m  # full-batch LDDMM script
├── script_TAA_matching_600.m   # 600-sample batch script
├── prepare_training_data.py    # CFD→LDDMM WSS interpolation
├── coefficients_convert.py     # SVD geometry coefficient computation
└── svd_utils.py                # SVD reduction utility
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
