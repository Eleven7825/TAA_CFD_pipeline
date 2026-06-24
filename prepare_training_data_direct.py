#!/usr/bin/env python3
"""
Assemble ShapeOperatorLearning training NPZs from the FSG CFD results.

Unlike the LDDMM path (prepare_training_data.py), the FSG CFD result already
stores WSS / pressure / coords at the *same* 672 base-mesh interface nodes used
by the POD geometry encoding — so no inverse-distance interpolation is needed.
We only reorder each result into the canonical interface node order (`ids` from
augmented_displacements.npz, i.e. interface.vtp order, the same order the POD
basis Ux/Uy/Uz lives in) so that node i is consistent across the geometry basis
and the WSS/pressure labels.

Sample sample_{id_offset+k} corresponds to displacement field disp[k] and to
coefficient row k, so we also re-emit the geometry coefficients with
case_numbers = id_offset + k to pair file-for-file with the labels.

Outputs (per sample): training_data_fsg/processed_TAA_data_{sample_id}.npz
    transformed_values  (672, 3)  WSS at the deformed interface nodes
    pressure_values     (672, 1)  pressure
    ref_xyz             (672, 3)  deformed node positions
Plus an aligned coefficient_data_fsg_m8_aligned.npz with matching case_numbers.

Usage:
    python prepare_training_data_direct.py \
        --results_dir fsg_results --disp_npz augmented_displacements.npz \
        --coeff_npz coefficient_data_fsg_m8.npz \
        --output_dir training_data_fsg --id_offset 30000
"""

import argparse
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description="Assemble FSG CFD results into training NPZs")
    p.add_argument("--results_dir", default=os.path.join(BASE_DIR, "fsg_results"),
                   help="Dir with sample_<id>/result.npz downloaded from the cluster")
    p.add_argument("--disp_npz", default=os.path.join(BASE_DIR, "augmented_displacements.npz"))
    p.add_argument("--coeff_npz", default=os.path.join(BASE_DIR, "coefficient_data_fsg_m8.npz"))
    p.add_argument("--output_dir", default=os.path.join(BASE_DIR, "training_data_fsg"))
    p.add_argument("--id_offset", type=int, default=30000,
                   help="sample_{id_offset+k} <-> disp[k] <-> coeff row k")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    aug = np.load(args.disp_npz, allow_pickle=True)
    ids = np.asarray(aug["ids"]).astype(int)        # canonical node order
    n_samples = aug["disp"].shape[0]

    saved, skipped, missing = 0, 0, 0
    sample_ids = []
    for k in range(n_samples):
        sample_id = args.id_offset + k
        rpath = os.path.join(args.results_dir, f"sample_{sample_id:05d}", "result.npz")
        if not os.path.exists(rpath):
            missing += 1
            continue

        r = np.load(rpath, allow_pickle=True)
        nid = r["node_ids"].astype(int)
        # reorder result rows into canonical `ids` order
        pos = {n: i for i, n in enumerate(nid)}
        try:
            order = np.array([pos[n] for n in ids])
        except KeyError:
            print(f"  sample_{sample_id}: node_ids mismatch, skipping")
            skipped += 1
            continue

        wss = r["wss"][order].astype(np.float64)            # (672, 3)
        pressure = r["pressure"][order].astype(np.float64)  # (672,)
        coords = r["coords"][order].astype(np.float64)      # (672, 3)

        # Reject NaN/inf and physically implausible magnitudes (a diverged CFD
        # solve on an extreme geometry can emit |WSS| ~1e70 and poison training).
        if (not np.isfinite(wss).all() or not np.isfinite(pressure).all()
                or np.all(wss == 0)
                or np.abs(wss).max() > 1.0 or np.abs(pressure).max() > 10.0):
            print(f"  sample_{sample_id}: non-finite / all-zero / out-of-range "
                  f"(|WSS|max={np.abs(wss).max():.3g}), skipping")
            skipped += 1
            continue

        out = os.path.join(args.output_dir, f"processed_TAA_data_{sample_id}.npz")
        np.savez(
            out,
            transformed_values=wss,
            pressure_values=pressure[:, None],
            ref_xyz=coords,
        )
        sample_ids.append(sample_id)
        saved += 1

    print(f"Saved {saved} training NPZs to {args.output_dir}/  "
          f"(skipped {skipped}, missing {missing})")

    # Re-emit geometry coefficients with case_numbers aligned to sample IDs.
    coeff = np.load(args.coeff_npz, allow_pickle=True)
    coefficients = coeff["coefficients"]
    aligned_case_numbers = args.id_offset + np.asarray(coeff["case_numbers"]).astype(int)
    # keep only rows whose label file was actually written
    keep = np.isin(aligned_case_numbers, np.array(sample_ids))
    stem = os.path.splitext(args.coeff_npz)[0]
    aligned_path = f"{stem}_aligned.npz"
    np.savez(
        aligned_path,
        coefficients=coefficients[keep],
        case_numbers=aligned_case_numbers[keep],
        l_max=coeff["l_max"],
    )
    print(f"Saved aligned coefficients {coefficients[keep].shape} "
          f"(case_numbers {aligned_case_numbers[keep].min()}..{aligned_case_numbers[keep].max()}) "
          f"to {aligned_path}")


if __name__ == "__main__":
    main()
