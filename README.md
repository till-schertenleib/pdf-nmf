# pdf-nmf

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Non-Negative Matrix Factorization (NMF) for Pair Distribution Function (PDF) Analysis.**

`pdf-nmf` is a Python tool for decomposing series of total-scattering atomic pair distribution functions ($G(r)$) into constituent chemical or structural components and their relative fractional weights across samples. It builds upon the methodology described in [Liu et al. (2021)](#scientific-background--references), with enhanced baseline removal, scale ambiguity resolution, and stability checks.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Using `conda` / `mamba` (Recommended)](#using-conda--mamba-recommended)
  - [Using `pip`](#using-pip)
- [Quick Start](#quick-start)
- [Command-Line Options](#command-line-options)
- [Input Data Format](#input-data-format)
- [Output Structure](#output-structure)
- [Mathematical Framework (v2 Scaling & Interpretation)](#mathematical-framework-v2-scaling--interpretation)
- [Synthetic Mixture Validation](#synthetic-mixture-validation)
- [Scientific Background & References](#scientific-background--references)
- [License](#license)

---

## Features

- **Automatic Baseline Shift Handling**: NMF requires non-negative data ($V \ge 0$). Because experimental $G(r)$ oscillates around zero, `pdf-nmf` shifts the dataset non-negatively prior to factorization.
- **Resolution of NMF Scale Ambiguity (v2)**: Standard NMF is invariant to scalar rescalings $W_{ik} H_{kj} = (W_{ik} s_k)(H_{kj} / s_k)$, making raw weights incomparable across components. `pdf-nmf` removes the artificial baseline offset each component inherits from the global shift and normalises components (unit $L_2$ norm by default).
- **Physical Signal-Share Weights**: Provides absolute weights $A_{ik}$ and normalized signal fractions $f_{ik} = A_{ik} / \sum_k A_{ik}$ that directly reflect each component's share of the PDF signal.
- **Initialisation Stability Testing**: Refit models over $N$ random initial starts (`--seeds N`) with Hungarian assignment matching to assess component stability and calculate standard deviations.
- **Systematic Model Selection**: Evaluates 1 through $N$ components (`--max_comp`) tracking both Frobenius error and relative reconstruction errors ($\|G - G_{\text{NMF}}\| / \|G\|$).
- **Dual Component Export**: Saves both raw shifted components (`component_k.cgr`) and baseline-subtracted normalised components (`component_k_norm_{norm}.cgr`).
- **Comprehensive Tables**: Exports detailed machine-readable CSV tables containing raw weights, absolute weights, signal fractions for all normalisations, and sample offsets, alongside publication-ready LaTeX tables.
- **Diagnostic & Overview Plots**: Generates Rietveld-style data vs. reconstruction difference plots for each sample, as well as dual-panel summary overviews.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/till-schertenleib/pdf-nmf.git
cd pdf-nmf
```

### Using `conda` / `mamba` (Recommended)

Using **Conda** or **Mamba** (e.g. Miniforge) is recommended for DiffPy workflows:

#### Option A: Quick setup with `environment.yml`
```bash
conda env create -f environment.yml
conda activate pdf-nmf
```
*(If using `mamba`, substitute `mamba env create -f environment.yml`)*

#### Option B: Manual setup
```bash
conda create -n pdf-nmf python=3.10
conda activate pdf-nmf
conda install -c conda-forge diffpy.utils scikit-learn scipy matplotlib numpy
pip install -e .
```

---

### Using `pip`

Requires Python 3.9+:

1. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```powershell
     py -3 -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```
   *(Alternatively: `pip install -r requirements.txt`)*

---

## Quick Start

### Using the CLI command (`pdf-nmf`):

```bash
pdf-nmf --files data/*.gr --max_comp 5 --rmin 1.5 --rmax 20.0 --outdir results --norm l2
```

### Running the Python script directly:

```bash
python nmf.py --files data/sample_01.gr data/sample_02.gr data/sample_03.gr --max_comp 4 --norm l2 --seeds 20
```

---

## Command-Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--files` | `str` [list] | *(Required)* | Paths to PDF files (supports glob patterns e.g. `path/*.gr`). |
| `--max_comp` | `int` | `6` | Maximum number of NMF components to calculate (from 1 to `max_comp`). |
| `--rmin` | `float` | `None` | Minimum $r$-value (Å) to include. |
| `--rmax` | `float` | `None` | Maximum $r$-value (Å) to include. |
| `--outdir` | `str` | `nmf_results` | Output directory where all results, CSVs, and plots are saved. |
| `--norm` | `str` | `l2` | Normalisation method: `l2` (unit $L_2$ norm, recommended), `max` (peak amplitude = 1), or `raw` (unnormalised v1 behavior). |
| `--seeds` | `int` | `0` | If $> 0$, refits each model with $N$ random seeds to compute fraction uncertainties (std). |
| `--no_show` | `flag` | `False` | Suppresses interactive plot popups (useful for headless / script runs). |

---

## Input Data Format

Files provided to `--files` must be standard two-column ASCII text files (`.gr` files produced by `PDFgetX3`, `PDFgetN3`, etc.):
- **Column 1**: Radial distance $r$ in Angstroms (Å).
- **Column 2**: Pair distribution function $G(r)$.
- Header lines starting with `#` are automatically parsed.
- **Important**: All files in a given run must share the same $r$-grid spacing.

---

## Output Structure

Running the analysis produces the following directory layout in `--outdir`:

```text
nmf_results/
├── reconstruction_errors.txt
├── summary.png
├── reconstruction_error_and_components.png
├── 1_components/
│   ├── component_1.cgr
│   ├── component_1_norm_l2.cgr
│   ├── weights_1_components.csv
│   ├── weights_1_components.tex
│   └── reconstruction_plots_1c/
│       ├── sample_01_recon.png
│       └── ...
├── 2_components/
│   ├── component_1.cgr
│   ├── component_1_norm_l2.cgr
│   ├── component_2.cgr
│   ├── component_2_norm_l2.cgr
│   ├── weights_2_components.csv
│   ├── weights_2_components.tex
│   └── reconstruction_plots_2c/
│       └── ...
└── ...
```

### Description of Output Files

- **`reconstruction_errors.txt`**: Tabulated component count, Frobenius norm, relative error in shifted space, and relative error in original $G(r)$ space.
- **`summary.png`**: Dual-panel overview with relative error curve and stacked normalised components for `max_comp`.
- **`{n}_components/component_{i}.cgr`**: Raw NMF component $i$ in shifted space ($r$ vs. $H_i(r)$).
- **`{n}_components/component_{i}_norm_{norm}.cgr`**: Mean-removed, normalized component profile ($r$ vs. $\hat{G}_i(r)$).
- **`{n}_components/weights_{n}_components.csv`**: Full machine-readable table with raw weights $W$, absolute weights $A$ for all normalisations, fractional weights $f$, sample offsets $d_i$, and seed stability standard deviations (if `--seeds > 0`).
- **`{n}_components/weights_{n}_components.tex`**: LaTeX table reporting fractional weights $f_{ik}$ and total weight $\sum_k A_{ik}$ for the chosen `--norm`.
- **`reconstruction_plots_{n}c/{sample}_recon.png`**: Rietveld-style plot showing experimental data, NMF reconstruction, and difference curve.

---

## Mathematical Framework (v2 Scaling & Interpretation)

### 1. Baseline Shift
NMF requires non-negative matrix elements ($V_{ij} \ge 0$). For a dataset of observed PDFs $G_i(r)$, a global constant shift is applied:
$$\text{shift} = \max\left(0, -\min_{i, r} G_i(r)\right), \quad V_i(r) = G_i(r) + \text{shift}$$

### 2. Factorization & Baseline Removal
The shifted matrix is factorized as $V \approx W H$. Because of the global shift, each extracted component $H_k(r)$ carries a non-zero baseline:
$$b_k = \langle H_k(r) \rangle_r$$
We subtract this baseline to recover zero-mean, $G(r)$-like component curves:
$$\tilde{H}_k(r) = H_k(r) - b_k$$

### 3. Normalisation & Absolute Weights
Components are normalised by scale factor $s_k$:
$$\hat{G}_k(r) = \frac{\tilde{H}_k(r)}{s_k}$$
where:
- `--norm l2`: $s_k = \|\tilde{H}_k(r)\|_2 = \sqrt{\sum_r \tilde{H}_k(r)^2}$ (default)
- `--norm max`: $s_k = \max_r |\tilde{H}_k(r)|$
- `--norm raw`: $s_k = 1$ (reproduces v1 behavior)

Absolute weights are scaled correspondingly:
$$A_{ik} = W_{ik} \cdot s_k$$

### 4. Exact Sample Reconstruction
Each sample's original PDF is reconstructed as:
$$G_i(r) = \sum_k A_{ik} \hat{G}_k(r) + d_i$$
where $d_i = \sum_k W_{ik} b_k - \text{shift}$ is a constant offset per sample (approximately the sample's mean over the fitted $r$-range).

### 5. Fractional Signal Shares & Physical Interpretation
The fractional weights are defined as:
$$f_{ik} = \frac{A_{ik}}{\sum_k A_{ik}}$$

> **Important Scientific Note**:
> The fractions $f_{ik}$ represent the **share of the total PDF signal** ($L_2$ norm) contributed by component $k$.
> They are **not** identical to physical phase or mole fractions. Because total-scattering PDF amplitude scales with atomic scattering power and density, an end member with larger scattering amplitude contributes disproportionately to the signal.
>
> For a two-component mixture $(1 - x) G_A + x G_B$, the expected signal share is:
> $$f_{\text{pred}, B}(x) = \frac{x \cdot s_B}{x \cdot s_B + (1 - x) \cdot s_A}$$
> $f_{iB}$ equals the mixing fraction $x$ only if both end members share identical PDF signal amplitudes ($s_A = s_B$).

---

## Synthetic Mixture Validation

The included script `nmf_validation.py` tests this behavior by creating synthetic mixtures of two measured end-member PDFs and comparing the recovered fractions against $x$ and $f_{\text{pred}}(x)$:

```bash
python nmf_validation.py --a path/to/end_member_A.gr --b path/to/end_member_B.gr --rmin 1.5 --rmax 6.5 --steps 11
```

---

## Scientific Background & References

This implementation is based on the methodology published by Billinge and coworkers:

1. **Methodology & Foundation**:
   > Liu, C.-H., Wright, C. J., Gu, R., Bandi, S., Wustrow, A., Todd, P. K., O'Nolan, D., Beauvais, M. L., Neilson, J. R., Chupas, P. J., Chapman, K. W. & Billinge, S. J. L. (2021).  
   > *Non-negative matrix factorization for pair distribution function analysis.*  
   > **J. Appl. Cryst.** 54, 768–775.  
   > [DOI: 10.1107/S160057672100265X](https://doi.org/10.1107/S160057672100265X)

2. **Web Implementation & Service**:
   > Thatcher, Z., Liu, C.-H., Yang, L., McBride, B. C., Thinh Tran, G., Wustrow, A., Karlsen, M. A., Neilson, J. R., Ravnsbaek, D. B. & Billinge, S. J. L. (2022).  
   > *PDFitc: a web-based tool center for pair distribution function analysis.*  
   > **Acta Cryst.** A78, 242–248.  
   > [DOI: 10.1107/S205252062200234X](https://doi.org/10.1107/S205252062200234X)

---

## License

This project is licensed under the [MIT License](LICENSE).
