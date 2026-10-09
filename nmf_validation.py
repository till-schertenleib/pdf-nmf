"""
Validation of the NMF weight scaling with synthetic two-end-member mixtures.

Builds G_mix(x) = (1-x) G_A + x G_B from two measured PDFs,
runs the same 2-component NMF pipeline as nmf.py, and compares the recovered fraction of the
B-like component with:
  * x                                  (the true mixing fraction), and
  * the "signal share" predicted for each normalisation:
        f_pred(x) = x*s_B / (x*s_B + (1-x)*s_A),  where s = L2 norm or max|.| of the mean-removed G.

If the pipeline works correctly, f_l2 and f_max follow f_pred; they equal x only if s_A == s_B.

Usage:
    python nmf_validation.py --a <path_to_A.gr> --b <path_to_B.gr> [--rmin 1.5] [--rmax 6.5] [--steps 11]
"""
import argparse
import numpy as np
from nmf import load_pdf_data, shift_data, fit_nmf, normalise_components, NORMS


def scale(g, norm):
    g = g - g.mean()
    return {"l2": np.linalg.norm(g), "max": np.max(np.abs(g)), "raw": np.nan}[norm]


def main():
    parser = argparse.ArgumentParser(description="Synthetic-mixture validation test of NMF weight scaling.")
    parser.add_argument("--a", required=True, help="Path to end member A PDF (.gr file, x = 0).")
    parser.add_argument("--b", required=True, help="Path to end member B PDF (.gr file, x = 1).")
    parser.add_argument("--rmin", type=float, default=1.5, help="Minimum r-value in Angstroms.")
    parser.add_argument("--rmax", type=float, default=6.5, help="Maximum r-value in Angstroms.")
    parser.add_argument("--steps", type=int, default=11, help="Number of mixing fractions between 0 and 1.")
    args = parser.parse_args()

    r, G = load_pdf_data([args.a, args.b], rmin=args.rmin, rmax=args.rmax)
    GA, GB = G
    x = np.linspace(0, 1, args.steps)
    X = np.array([(1 - xi) * GA + xi * GB for xi in x])

    Xs, shift = shift_data(X)
    W, H, model = fit_nmf(Xs, 2)
    rel = np.linalg.norm(X - (W @ H - shift)) / np.linalg.norm(X)

    # Identify which NMF component is B-like (largest correlation with G_B)
    kB = int(np.argmax([np.corrcoef(h, GB)[0, 1] for h in H]))

    print(f"r = {args.rmin}-{args.rmax} Å, shift = {shift:.4f}, rel. err (G-space) = {rel:.2e}")
    print(f"end-member scales  L2: A={scale(GA,'l2'):.3f}  B={scale(GB,'l2'):.3f} | "
          f"max: A={scale(GA,'max'):.3f}  B={scale(GB,'max'):.3f}\n")

    f = {n: normalise_components(W, H, shift, n)[2][:, kB] for n in NORMS}
    pred = {n: x * scale(GB, n) / (x * scale(GB, n) + (1 - x) * scale(GA, n)) for n in ("l2", "max")}

    print("   x    f_raw   f_l2  pred_l2   f_max  pred_max")
    for i, xi in enumerate(x):
        print(f"{xi:5.2f}  {f['raw'][i]:6.3f}  {f['l2'][i]:6.3f}  {pred['l2'][i]:6.3f}"
              f"  {f['max'][i]:6.3f}  {pred['max'][i]:6.3f}")

    print("\nRMS deviation from x       : " +
          "  ".join(f"{n}={np.sqrt(np.mean((f[n] - x) ** 2)):.3f}" for n in NORMS))
    print("RMS deviation from f_pred  : " +
          "  ".join(f"{n}={np.sqrt(np.mean((f[n] - pred[n]) ** 2)):.3f}" for n in ("l2", "max")))


if __name__ == "__main__":
    main()

