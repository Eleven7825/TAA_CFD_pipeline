#!/usr/bin/env python3
"""
Augment the FSG displacement dataset by random linear combinations.

The extracted FSG fields (fsg_displacements.npz) are coupling sub-iterations of a
single growth trajectory, so they are highly correlated — essentially one
aneurysm-growth shape mode sampled at increasing amplitude.  To enrich the
training set we synthesize new displacement fields as

    new = s * sum_k w_k * d_k

with convex weights  w ~ Dirichlet(1)  over the N original fields (an
interpolation that stays a physical aneurysm-growth shape) and a global scale
s ~ Uniform[s_min, s_max]  that varies / extrapolates the amplitude.  Because the
inputs lie along one growth mode, the augmentation mainly diversifies amplitude;
`--scale_range` is the lever for how far to extrapolate beyond the observed
trajectory.

Usage:
    python augment_displacements.py \
        --input fsg_displacements.npz \
        --output augmented_displacements.npz \
        --n_aug 500 --scale_range 0.5 1.5 --seed 0
"""

import argparse
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def augment(d, n_aug, scale_range, rng, alpha=1.0):
    """
    d : (N, P, 3) original displacement fields.
    alpha : Dirichlet concentration.  alpha=1 → uniform on the simplex (blends
        cluster near the mean shape); alpha<1 → sparse, near-one-hot weights
        (blends spread across the full amplitude range of the trajectory).
    Returns (aug (n_aug, P, 3), weights (n_aug, N), scales (n_aug,)).
    """
    n = d.shape[0]
    weights = rng.dirichlet(alpha * np.ones(n), size=n_aug)  # (n_aug, N), rows sum to 1
    scales = rng.uniform(scale_range[0], scale_range[1], size=n_aug)
    # einsum: weighted sum of fields, then per-sample scale.
    aug = np.einsum("ak,kpc->apc", weights, d)
    aug = aug * scales[:, None, None]
    return aug.astype(np.float64), weights.astype(np.float64), scales.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Random linear-combination augmentation of FSG displacements"
    )
    parser.add_argument("--input", type=str,
                        default=os.path.join(BASE_DIR, "fsg_displacements.npz"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(BASE_DIR, "augmented_displacements.npz"))
    parser.add_argument("--n_aug", type=int, default=500,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--scale_range", type=float, nargs=2, default=[0.5, 1.5],
                        help="Uniform range for the global amplitude scale")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Dirichlet concentration; <1 gives sparser blends "
                             "spanning the full amplitude range (default 0.3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_include_originals", action="store_true",
                        help="Exclude the original FSG fields from the output set")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    d = data["disp"].astype(np.float64)            # (N, P, 3)
    ids = data["ids"]
    ref_xyz = data["ref_xyz"]
    n_orig = d.shape[0]
    print(f"Loaded {n_orig} original FSG fields, {d.shape[1]} nodes")

    rng = np.random.default_rng(args.seed)
    aug, weights, scales = augment(d, args.n_aug, args.scale_range, rng, alpha=args.alpha)

    if args.no_include_originals:
        disp = aug
        is_original = np.zeros(len(aug), dtype=bool)
        # pad provenance so all arrays align with `disp`
        out_weights = weights
        out_scales = scales
    else:
        disp = np.concatenate([d, aug], axis=0)
        is_original = np.concatenate([
            np.ones(n_orig, dtype=bool), np.zeros(len(aug), dtype=bool)
        ])
        # originals get identity provenance (one-hot weight, scale 1)
        orig_w = np.eye(n_orig)
        out_weights = np.concatenate([orig_w, weights], axis=0)
        out_scales = np.concatenate([np.ones(n_orig), scales], axis=0)

    mag = np.linalg.norm(disp, axis=2)
    print(f"Output set: {disp.shape[0]} samples ({n_orig} original + {len(aug)} augmented)")
    print(f"  |disp| per-node max overall: {mag.max():.5f} cm")
    print(f"  per-sample max |disp| range: [{mag.max(axis=1).min():.5f}, {mag.max(axis=1).max():.5f}] cm")

    np.savez(
        args.output,
        disp=disp.astype(np.float64),
        ids=ids,
        ref_xyz=ref_xyz,
        is_original=is_original,
        weights=out_weights.astype(np.float64),
        scales=out_scales.astype(np.float64),
    )
    print(f"\nSaved {disp.shape} to {args.output}")


if __name__ == "__main__":
    main()
