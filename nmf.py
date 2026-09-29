import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from diffpy.utils.parsers.loaddata import loadData

# billinge colors:
bg_blue = '#0B3C5D'
bg_red = '#B82601'
bg_green = '#1c6b0a'

# nature colors
orange = (230/255, 159/255, 0/255)
skyblue = (86/255, 180/255, 233/255)
green = (0/255, 158/255, 115/255)
yellow = (240/255, 228/255, 66/255)
blue = (0/255, 114/255, 178/255)
verm = (213/255, 94/255, 0/255)
purple = (204/255, 121/255, 167/255)


def load_pdf_data(file_paths, rmin=None, rmax=None):
    """
    Loads PDF data from a list of files and filters by rmin and rmax. 
    Assumes files are 2D arrays where the first column is r and the second is G(r).
    """
    g_r_list = []
    r_grid = None
    mask = None
    
    for path in file_paths:
        data = loadData(path)
        r_vals = data[:, 0]
        g_vals = data[:, 1]
        
        if r_grid is None:
            mask = np.ones_like(r_vals, dtype=bool)
            if rmin is not None:
                mask &= (r_vals >= rmin)
            if rmax is not None:
                mask &= (r_vals <= rmax)
            r_grid = r_vals[mask]
            
        g_r_list.append(g_vals[mask])
        
    return r_grid, np.array(g_r_list)

def run_nmf_analysis(data_matrix, max_components=5):
    """
    Runs NMF for 1 to max_components and tracks the Frobenius reconstruction error.
    """
    errors = []
    models = {}
    shift_value = 0.0
    
    min_val = np.min(data_matrix)
    if min_val < 0:
        shift_value = -min_val
        print(f"Warning: Data contains negative values. Shifting data by {shift_value:.4f}")
        # Note: data_matrix here is a local reference, so the original g_matrix remains unshifted
        data_matrix = data_matrix + shift_value 

    for n in range(1, max_components + 1):
        model = NMF(n_components=n, init='nndsvda', max_iter=5000, random_state=42)
        W = model.fit_transform(data_matrix) 
        H = model.components_              
        
        err = model.reconstruction_err_
        errors.append(err)
        models[n] = (W, H)
        
        print(f"Components: {n} | Reconstruction Error (Frobenius Norm): {err:.4f}")

    return errors, models, shift_value

def plot_reconstructions(r, G_obs, W_raw, H_raw, filenames, outdir, n_components, shift_value=0.0):
    """
    Generates Rietveld-style plots comparing the original PDF to the NMF reconstruction.
    """
    # Calculate the reconstructed dataset and shift it back down to the original baseline
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


def save_latex_table(filepath, W_frac, scale_factors, filenames, shift_value):
    """
    Generates a compilable LaTeX table containing the NMF component weights 
    and the required scaling factors for reconstruction.
    """
    n_samples, n_components = W_frac.shape
    with open(filepath, 'w') as f:
        f.write("\\begin{table}[htpb]\n")
        f.write("\\centering\n")
        
        # Record the shift value directly in the caption for future reference
        f.write(f"\\caption{{NMF Component Weights. To reconstruct the raw PDF, use a global baseline shift of {shift_value:.4f}.}}\n")
        
        # Define column alignments: Sample, Scale Factor, then Components
        col_format = "l" + "c" + "c" * n_components
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\n")
        
        # Header row
        headers = ["Sample", "Scale Factor"] + [f"Comp {i+1}" for i in range(n_components)]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\hline\n")
        
        # Data rows
        for i in range(n_samples):
            row_name = os.path.basename(filenames[i]).replace('_', '\\_')
            scale_str = f"{scale_factors[i][0]:.4f}"
            weights = [f"{w:.4f}" for w in W_frac[i]]
            f.write(f"{row_name} & {scale_str} & " + " & ".join(weights) + " \\\\\n")
            
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def main():
    parser = argparse.ArgumentParser(description="Perform NMF on a set of PDF data files.")
    parser.add_argument(
        "--files", 
        nargs='+', 
        required=True, 
        help="List of paths to the PDF files"
    )
    parser.add_argument(
        "--max_comp", 
        type=int, 
        default=6, 
        help="Maximum number of NMF components to test."
    )
    parser.add_argument(
        "--rmin", 
        type=float, 
        default=None, 
        help="Minimum r-value."
    )
    parser.add_argument(
        "--rmax", 
        type=float, 
        default=None, 
        help="Maximum r-value."
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        default="nmf_results", 
        help="Main output directory name."
    )
    
    args = parser.parse_args()
    
    print("Loading data...")
    r, g_matrix = load_pdf_data(args.files, rmin=args.rmin, rmax=args.rmax)
    
    print("Running NMF factorization...")
    errors, models, applied_shift = run_nmf_analysis(g_matrix, max_components=args.max_comp)

    # ---------------------------------------------------------
    # 1. Output Data Generation
    # ---------------------------------------------------------
    print(f"Saving outputs to '{args.outdir}/' ...")
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save the reconstruction errors to a text file
    error_file = os.path.join(args.outdir, "reconstruction_errors.txt")
    error_data = np.column_stack((np.arange(1, args.max_comp + 1), errors))
    np.savetxt(error_file, error_data, header="Components Frobenius_Norm", fmt=["%d", "%.6f"], comments="")
    
    for n in range(1, args.max_comp + 1):
        W, H = models[n]
        
        # --- Calculate sum-to-one fractional weights and scale factors ---
        row_sums = np.sum(W, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        
        W_frac = W / row_sums
        # ---------------------------------------------------------------
        
        sub_dir = os.path.join(args.outdir, f"{n}_components")
        os.makedirs(sub_dir, exist_ok=True)
        
        # Save components
        for i in range(n):
            comp_file = os.path.join(sub_dir, f"component_{i+1}.cgr")
            np.savetxt(comp_file, np.column_stack((r, H[i])), 
                       header="r G(r)", fmt="%.6f", comments="# ")
            
        # Generate the LaTeX weights table (now passing row_sums and applied_shift)
        tex_file = os.path.join(sub_dir, f"weights_{n}_components.tex")
        save_latex_table(tex_file, W_frac, row_sums, args.files, applied_shift)
        
        # Generate reconstruction plots 
        plot_reconstructions(r, g_matrix, W, H, args.files, sub_dir, n, shift_value=applied_shift)


    # ---------------------------------------------------------
    # 2. Plotting (Reconstruction Error & max_comp Components)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot A: Reconstruction Error
    ax1.plot(range(1, args.max_comp + 1), errors, marker='o', linestyle='--', color='k')
    ax1.set_title('NMF Reconstruction Error vs. Number of Components')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Frobenius Norm')
    ax1.set_xticks(range(1, args.max_comp + 1))
    ax1.grid(True, alpha=0.5)
    
    # Plot B: Components for max_comp
    W_max, H_max = models[args.max_comp]
    ax2.set_title(f'Extracted NMF Components (n={args.max_comp})')
    ax2.set_xlabel('r (Å)')
    ax2.set_ylabel('Intensity (Offset)')
    
    # Calculate a suitable dynamic offset to prevent overlap
    # We use 1.2x the maximum amplitude of the components to ensure visual separation
    offset_step = np.max(H_max) * 1.2 if np.max(H_max) > 0 else 1.0
    
    for i in range(args.max_comp):
        y_offset = H_max[i] + (i * offset_step)
        ax2.plot(r, y_offset, label=f'Component {i+1}')
        
    ax2.legend(loc='upper right')
    plt.tight_layout()
    
    overview_plot = os.path.join(args.outdir, "reconstruction_error_and_components.png")
    plt.savefig(overview_plot, dpi=300)
    print(f"Overview figure saved to '{overview_plot}'")
    plt.show()

if __name__ == "__main__":
    main()