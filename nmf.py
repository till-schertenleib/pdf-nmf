"""
NMF of a series of PDFs, following nmfMapping / Liu et al., J. Appl. Cryst. 54, 768-775 (2021),
https://doi.org/10.1107/S160057672100265X

Weight scaling (v2):
    NMF has a scale ambiguity (W[:,k]*H[k] == (a*W[:,k])*(H[k]/a)), so raw W are not comparable
    between components. After the fit we:
      1. remove the baseline each component carries because of the global shift:
             Ht_k = H_k - <H_k>_r
      2. normalise:  G_hat_k = Ht_k / s_k   (s_k = L2 norm, max|Ht_k|, or 1 for 'raw')
      3. absolute weights A_ik = W_ik * s_k,   fractions f_ik = A_ik / sum_k A_ik
    Every sample is then reconstructed (exactly, up to the NMF fit error) as:
             G_i(r) = sum_k A_ik * G_hat_k(r) + d_i ,   d_i = sum_k W_ik <H_k> - shift
    d_i is a constant offset (~ mean of G_i over the fitted r-range).
    With --norm raw, f_ik equals the old W/rowsum weights of nmf.py v1 (regression check).
"""
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from diffpy.utils.parsers.loaddata import loadData

# Billinge colors:
bg_blue = '#0B3C5D'
bg_red = '#B82601'
bg_green = '#1c6b0a'

# Nature colors:
orange = (230/255, 159/255, 0/255)
skyblue = (86/255, 180/255, 233/255)
green = (0/255, 158/255, 115/255)
yellow = (240/255, 228/255, 66/255)
blue = (0/255, 114/255, 178/255)
verm = (213/255, 94/255, 0/255)
purple = (204/255, 121/255, 167/255)

NORMS = ("l2", "max", "raw")


def load_pdf_data(file_paths, rmin=None, rmax=None):
    """
    Loads PDF data from a list of files and filters by rmin and rmax.
    Assumes files are 2D arrays where the first column is r and the second is G(r).
    All files must share the same r-grid (checked).
    """
    g_r_list = []
    r_full = None
    mask = None

    for path in file_paths:
        data = loadData(path)
        r_vals = data[:, 0]
        g_vals = data[:, 1]

        if r_full is None:
            r_full = r_vals
            mask = np.ones_like(r_vals, dtype=bool)
            if rmin is not None:
                mask &= (r_vals >= rmin)
            if rmax is not None:
                mask &= (r_vals <= rmax)
        elif len(r_vals) != len(r_full) or not np.allclose(r_vals, r_full):
            raise ValueError(f"r-grid of {path} differs from {file_paths[0]}; regrid the data first.")

        g_r_list.append(g_vals[mask])

    return r_full[mask], np.array(g_r_list)


def shift_data(data_matrix):
    """Global shift so that the matrix is non-negative (as in nmfMapping). Returns (shifted, shift)."""
    shift_value = max(0.0, -float(np.min(data_matrix)))
    if shift_value > 0:
        print(f"Warning: Data contains negative values. Shifting data by {shift_value:.4f}")
    return data_matrix + shift_value, shift_value


def fit_nmf(shifted, n, init='nndsvda', random_state=42, max_iter=5000):
    """Single NMF fit. Returns W, H, model."""
    model = NMF(n_components=n, init=init, max_iter=max_iter, random_state=random_state)
    W = model.fit_transform(shifted)
    H = model.components_
    return W, H, model


def run_nmf_analysis(data_matrix, max_components=5):
    """
    Runs NMF for 1 to max_components and tracks the reconstruction error.
    Returns absolute Frobenius norms (shifted space), relative errors in shifted space and
    in G-space, the models {n: (W, H)} and the applied shift.
    """
    shifted, shift_value = shift_data(data_matrix)
    errors, rel_shift, rel_g = [], [], []
    models = {}

    for n in range(1, max_components + 1):
        W, H, model = fit_nmf(shifted, n)
        recon = W @ H
        errors.append(model.reconstruction_err_)
        rel_shift.append(np.linalg.norm(shifted - recon) / np.linalg.norm(shifted))
        rel_g.append(np.linalg.norm(data_matrix - (recon - shift_value)) / np.linalg.norm(data_matrix))
        models[n] = (W, H)
        if model.n_iter_ >= model.max_iter:
            print(f"Warning: NMF with {n} components did not converge.")
        print(f"Components: {n} | Frobenius: {errors[-1]:.4f} | rel. err shifted: {rel_shift[-1]:.4f}"
              f" | rel. err G(r): {rel_g[-1]:.4f}")

    return errors, rel_shift, rel_g, models, shift_value


def normalise_components(W, H, shift_value, norm="l2"):
    """
    Removes the per-component baseline and normalises the components (see module docstring).
    Returns G_hat (n x npts), A (absolute weights), f (fractions), offset d_i, scale s_k, baseline b_k.
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}")
    b = H.mean(axis=1)                     # baseline carried by each component
    Ht = H - b[:, None]                    # G-like (mean-free) components
    if norm == "l2":
        s = np.linalg.norm(Ht, axis=1)
    elif norm == "max":
        s = np.max(np.abs(Ht), axis=1)
    else:
        s = np.ones(H.shape[0])
    s[s == 0] = 1.0
    G_hat = Ht / s[:, None]
    A = W * s[None, :]
    row = A.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    f = A / row
    offset = W @ b - shift_value
    return G_hat, A, f, offset, s, b


def match_components(H_ref, H):
    """Order of rows of H that best matches H_ref (max. correlation, Hungarian assignment)."""
    n = H_ref.shape[0]
    corr = np.corrcoef(H_ref, H)[:n, n:]
    _, cols = linear_sum_assignment(-corr)
    return cols


def seed_stability(data_matrix, n, H_ref, norm, n_seeds):
    """Refits with random initialisations; returns mean and std of fractions (components matched to H_ref)."""
    shifted, shift_value = shift_data(data_matrix)
    fracs = []
    for seed in range(n_seeds):
        W, H, _ = fit_nmf(shifted, n, init='random', random_state=seed)
        order = match_components(H_ref, H)
        W, H = W[:, order], H[order]
        fracs.append(normalise_components(W, H, shift_value, norm)[2])
    fracs = np.array(fracs)
    return fracs.mean(axis=0), fracs.std(axis=0)


def plot_reconstructions(r, G_obs, W_raw, H_raw, filenames, outdir, n_components, shift_value=0.0):
    """
    Generates Rietveld-style plots comparing the original PDF to the NMF reconstruction.
    """
    G_recon = np.dot(W_raw, H_raw) - shift_value

    plot_dir = os.path.join(outdir, f"reconstruction_plots_{n_components}c")
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(len(filenames)):
        sample_name = os.path.basename(filenames[i]).replace('.gr', '')

        y_obs = G_obs[i]
        y_calc = G_recon[i]
        residual = y_obs - y_calc

        offset = np.min(y_obs) - (np.max(y_obs) - np.min(y_obs)) * 0.3

        plt.figure(figsize=(8, 6))
        plt.plot(r, y_obs, linestyle='None', marker='o', label='Data', mfc='None', color=blue, markersize=8)
        plt.plot(r, y_calc, '-', label='NMF Fit', color=verm, linewidth=2)
        plt.plot(r, residual + offset, '-', label='Difference', color=green, linewidth=2)

        plt.axhline(offset, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

        plt.title(f'NMF Reconstruction: {sample_name} ({n_components} Components)')
        plt.xlabel('r (Å)')
        plt.ylabel('G(r)')
        plt.legend(loc='upper right')
        plt.tight_layout()

        save_path = os.path.join(plot_dir, f"{sample_name}_recon.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


def save_weights_csv(filepath, filenames, W, results, shift_value, frac_std=None):
    """
    One row per sample: raw W, and for every normalisation the absolute weights A, fractions f
    and the offset d (G_i = sum_k A_ik G_hat_k + d_i). Optional seed-stability std of the fractions.
    """
    n = W.shape[1]
    header = ["sample"] + [f"W{k+1}" for k in range(n)]
    for norm in results:
        header += [f"A{k+1}_{norm}" for k in range(n)] + [f"f{k+1}_{norm}" for k in range(n)]
    header += ["offset", "shift"]
    if frac_std is not None:
        header += [f"f{k+1}_std" for k in range(n)]

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, name in enumerate(filenames):
            row = [os.path.basename(name)] + [f"{x:.6g}" for x in W[i]]
            for norm, (_, A, f, _, _, _) in results.items():
                row += [f"{x:.6g}" for x in A[i]] + [f"{x:.6g}" for x in f[i]]
            offset = next(iter(results.values()))[3]   # identical for every normalisation
            row += [f"{offset[i]:.6g}", f"{shift_value:.6g}"]
            if frac_std is not None:
                row += [f"{x:.4g}" for x in frac_std[i]]
            w.writerow(row)


def save_latex_table(filepath, W_frac, scale_factors, filenames, shift_value, norm="l2"):
    """
    Generates a compilable LaTeX table containing the NMF component fractions
    and the absolute weight sum (sum_k A_ik) for the chosen normalisation.
    """
    n_samples, n_components = W_frac.shape
    with open(filepath, 'w', encoding="utf-8") as f:
        f.write("\\begin{table}[htpb]\n")
        f.write("\\centering\n")

        f.write(f"\\caption{{NMF component fractions $f_{{ik}} = A_{{ik}}/\\sum_k A_{{ik}}$ "
                f"(components mean-removed, normalisation: {norm}). "
                f"$\\sum_k A_{{ik}}$ is the total weight of sample $i$. Global shift used in the fit: {shift_value:.4f}.}}\n")

        col_format = "l" + "c" + "c" * n_components
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\n")

        headers = ["Sample", "$\\sum_k A_{ik}$"] + [f"Comp {i+1}" for i in range(n_components)]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\hline\n")

        for i in range(n_samples):
            row_name = os.path.basename(filenames[i]).replace('_', '\\_')
            scale_str = f"{scale_factors[i]:.4f}"
            weights = [f"{w:.4f}" for w in W_frac[i]]
            f.write(f"{row_name} & {scale_str} & " + " & ".join(weights) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(description="Perform NMF on a set of PDF data files.")
    parser.add_argument("--files", nargs='+', required=True, help="List of paths to the PDF files")
    parser.add_argument("--max_comp", type=int, default=6, help="Maximum number of NMF components to test.")
    parser.add_argument("--rmin", type=float, default=None, help="Minimum r-value.")
    parser.add_argument("--rmax", type=float, default=None, help="Maximum r-value.")
    parser.add_argument("--outdir", type=str, default="nmf_results", help="Main output directory name.")
    parser.add_argument("--norm", choices=NORMS, default="l2",
                        help="Normalisation used for the LaTeX table and normalised components "
                             "(all normalisations are always written to the CSV). Default is 'l2'.")
    parser.add_argument("--seeds", type=int, default=0,
                        help="If >0, refit each model with this many random initialisations and "
                             "report the std of the fractions (stability check).")
    parser.add_argument("--no_show", action="store_true", help="Do not open the summary plot window.")

    args = parser.parse_args()

    print("Loading data...")
    r, g_matrix = load_pdf_data(args.files, rmin=args.rmin, rmax=args.rmax)

    print("Running NMF factorization...")
    errors, rel_shift, rel_g, models, applied_shift = run_nmf_analysis(g_matrix, max_components=args.max_comp)

    # ---------------------------------------------------------
    # 1. Output Data Generation
    # ---------------------------------------------------------
    print(f"Saving outputs to '{args.outdir}/' ...")
    os.makedirs(args.outdir, exist_ok=True)

    # Save the reconstruction errors to a text file
    error_file = os.path.join(args.outdir, "reconstruction_errors.txt")
    error_data = np.column_stack((np.arange(1, args.max_comp + 1), errors, rel_shift, rel_g))
    np.savetxt(error_file, error_data, header="Components Frobenius_Norm RelErr_shifted RelErr_G",
               fmt=["%d", "%.6f", "%.6f", "%.6f"], comments="")

    for n in range(1, args.max_comp + 1):
        W, H = models[n]

        # --- Baseline removal + normalisation of components (all variants) ---
        results = {norm: normalise_components(W, H, applied_shift, norm) for norm in NORMS}
        G_hat, A, W_frac, offset, s, b = results[args.norm]
        # ---------------------------------------------------------------

        frac_std = None
        if args.seeds > 0:
            _, frac_std = seed_stability(g_matrix, n, H, args.norm, args.seeds)
            print(f"{n} comps: max std of fractions over {args.seeds} random inits = {frac_std.max():.4f}")

        sub_dir = os.path.join(args.outdir, f"{n}_components")
        os.makedirs(sub_dir, exist_ok=True)

        # Save components: raw (shifted space, as in v1) and normalised G-like components
        for i in range(n):
            comp_file = os.path.join(sub_dir, f"component_{i+1}.cgr")
            np.savetxt(comp_file, np.column_stack((r, H[i])),
                       header="r G(r)  raw NMF component (shifted space)", fmt="%.6f", comments="# ")
            norm_file = os.path.join(sub_dir, f"component_{i+1}_norm_{args.norm}.cgr")
            np.savetxt(norm_file, np.column_stack((r, G_hat[i])),
                       header=f"r G_hat(r)  mean-removed component, norm={args.norm}, scale s={s[i]:.6g}, "
                              f"baseline b={b[i]:.6g}", fmt="%.6e", comments="# ")

        # Machine-readable weights (all normalisations) and the LaTeX table (chosen normalisation)
        save_weights_csv(os.path.join(sub_dir, f"weights_{n}_components.csv"),
                         args.files, W, results, applied_shift, frac_std)
        tex_file = os.path.join(sub_dir, f"weights_{n}_components.tex")
        save_latex_table(tex_file, W_frac, A.sum(axis=1), args.files, applied_shift, args.norm)

        # Generate reconstruction plots
        plot_reconstructions(r, g_matrix, W, H, args.files, sub_dir, n, shift_value=applied_shift)

    # ---------------------------------------------------------
    # 2. Plotting (Reconstruction Error & max_comp Components)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot A: Reconstruction Error
    ax1.plot(range(1, args.max_comp + 1), rel_g, marker='o', linestyle='--', color='k')
    ax1.set_title('NMF Reconstruction Error vs. Number of Components')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel(r'Relative error $\|G - G_{NMF}\| / \|G\|$')
    ax1.set_xticks(range(1, args.max_comp + 1))
    ax1.grid(True, alpha=0.5)

    # Plot B: normalised components for max_comp
    W_max, H_max = models[args.max_comp]
    G_hat_max = normalise_components(W_max, H_max, applied_shift, args.norm)[0]
    ax2.set_title(f'Normalised NMF Components (n={args.max_comp}, norm={args.norm})')
    ax2.set_xlabel('r (Å)')
    ax2.set_ylabel('Intensity (Offset)')

    # Calculate a suitable dynamic offset to prevent overlap
    offset_step = 1.2 * (np.max(G_hat_max) - np.min(G_hat_max)) if np.ptp(G_hat_max) > 0 else 1.0

    for i in range(args.max_comp):
        ax2.plot(r, G_hat_max[i] + i * offset_step, label=f'Component {i+1}')

    ax2.legend(loc='upper right')
    plt.tight_layout()

    # Save overview figures under both standard names for compatibility
    overview_plot = os.path.join(args.outdir, "reconstruction_error_and_components.png")
    plt.savefig(overview_plot, dpi=300)
    plt.savefig(os.path.join(args.outdir, "summary.png"), dpi=200)
    print(f"Overview figure saved to '{overview_plot}'")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()