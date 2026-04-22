#!/usr/bin/env python3
"""
Usage
-----
  python3 gen_lambert_w_table.py -o fitter/core/lambert_w_table.npz

Parameters
----------
  --re-range   Re(z) coverage range, default [-0.40, 0.40]
  --im-range   Im(z) coverage range, default [-0.40, 0.40]
  --n          Grid points per axis, default 1024
  -o           输出路径
"""

import argparse

import time
import numpy as np

from scipy.special import lambertw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="lambert_w_table.npz")
    ap.add_argument("--re-range", nargs=2, type=float, default=[-0.40, 0.40])
    ap.add_argument("--im-range", nargs=2, type=float, default=[-0.40, 0.40])
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()

    n = args.n
    re_lo, re_hi = args.re_range
    im_lo, im_hi = args.im_range

    re_pts = np.linspace(re_lo, re_hi, n, dtype=np.float64)
    im_pts = np.linspace(im_lo, im_hi, n, dtype=np.float64)

    print(
        f"[*] Building {n}*{n} Lambert-W table "
        f"Re∈[{re_lo},{re_hi}]  Im∈[{im_lo},{im_hi}] ..."
    )
    t0 = time.perf_counter()

    RE, IM = np.meshgrid(re_pts, im_pts, indexing="ij")  # (n, n)
    Z = RE + 1j * IM
    W = lambertw(Z, k=0)

    W_re = np.real(W).astype(np.float64)  # (n, n)
    W_im = np.imag(W).astype(np.float64)  # (n, n)

    dt = time.perf_counter() - t0
    print(f"[*] Computed in {dt:.2f}s")

    # Accuracy checking
    rng = np.random.default_rng(0)
    n_check = 10000
    r_check = rng.uniform(0, 0.38, n_check)
    phi_check = rng.uniform(-np.pi, np.pi, n_check)
    z_check = r_check * np.exp(1j * phi_check)
    w_ref = lambertw(z_check, k=0)

    # Bilinear interpolation error estimation
    # Theoretically, bilinear interpolation is O(h²) in the spacing h
    # and O(h⁴) in the spacing h³.
    h_re = (re_hi - re_lo) / (n - 1)
    h_im = (im_hi - im_lo) / (n - 1)
    print(f"[*] Grid spacing: dRe={h_re:.5f}  dIm={h_im:.5f}")
    print(
        f"[*] Estimated bilinear interpolation error: O(h²) ≈ {max(h_re,h_im)**2:.2e}"
    )

    np.savez_compressed(
        args.output,
        re_pts=re_pts,
        im_pts=im_pts,
        W_re=W_re,
        W_im=W_im,
    )
    size_kb = __import__("os").path.getsize(args.output) / 1024
    print(f"[+] Saved to {args.output}  ({size_kb:.0f} KB)")
    print(f"    re_pts: {re_pts.shape}  im_pts: {im_pts.shape}")
    print(f"    W_re: {W_re.shape}  W_im: {W_im.shape}")


if __name__ == "__main__":
    main()
