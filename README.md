# pdf-nmf

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Non-Negative Matrix Factorization (NMF) for Pair Distribution Function (PDF) Analysis.**

`pdf-nmf` is a Python tool for decomposing series of total-scattering atomic pair distribution functions ($G(r)$) into constituent chemical or structural components and their relative fractional weights across samples. It implements the methodology described in [Liu et al. (2021)](#references) and serves as an accessible standalone implementation.

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
  - [Mathematical Note on Shifts & Weights](#mathematical-note-on-shifts--weights)
- [Scientific Background & References](#scientific-background--references)
- [License](#license)

---

## Features

- **Automatic Baseline Shift Handling**: NMF requires strictly non-negative data ($V \ge 0$). Since PDF $G(r)$ typically oscillates around zero and has negative values, `pdf-nmf` automatically determines the minimum baseline across your series and applies the necessary positive shift before factorization.
- **Systematic Model Selection**: Evaluates 1 through $N$ components (`--max_comp`) and tracks the Frobenius reconstruction error to help identify the optimal number of physical components.
- **Component Export**: Saves each extracted constituent component as a standard two-column `.cgr` file ($r$ vs. $G(r)$), ready to be loaded into PDFgui, DiffPy-CMI, or plotting tools.
- **Publication-Ready LaTeX Tables**: Generates compilable LaTeX tables containing sample names, overall scale factors, and fractional weights normalized to sum to 1.
- **Rietveld-Style Diagnostic Plots**: Produces data vs. fit vs. difference curves for each sample and component count.
- **Overview Visualizations**: Outputs a dual-panel figure showing both the reconstruction error curve and vertically offset component spectra.

---

## Installation

First, clone this repository:

```bash
git clone https://github.com/till-schertenleib/pdf-nmf.git
cd pdf-nmf
```

### Using `conda` / `mamba` (Recommended)

Using **Conda** or **Mamba** (e.g. Miniforge or Anaconda) is strongly recommended for crystallography and DiffPy workflows, as C-libraries and dependencies like `diffpy.utils` are maintained on `conda-forge`.

#### Option A: Quick setup with `environment.yml`
```bash
# Create the environment with all dependencies pre-configured
conda env create -f environment.yml

# Activate the environment
conda activate pdf-nmf
```
*(If using `mamba`, substitute `mamba env create -f environment.yml`)*

#### Option B: Manual step-by-step setup
```bash
# 1. Create and activate a new environment
conda create -n pdf-nmf python=3.10
conda activate pdf-nmf

# 2. Install scientific packages from conda-forge
conda install -c conda-forge diffpy.utils scikit-learn matplotlib numpy

# 3. Install pdf-nmf in editable mode
pip install -e .
```

---

### Using `pip`

If you prefer standard `pip` without Conda (requires Python 3.9+):

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
   *(Alternatively, install requirements directly: `pip install -r requirements.txt`)*

---

## Quick Start

Once installed, you can run `pdf-nmf` either using the installed CLI command or directly as a script:

### Using the CLI command (`pdf-nmf`):

```bash
pdf-nmf --files data/*.gr --max_comp 5 --rmin 1.5 --rmax 20.0 --outdir results
```

### Running the Python script directly:

```bash
python nmf.py --files data/sample_01.gr data/sample_02.gr data/sample_03.gr --max_comp 4
```

---

## Command-Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--files` | `str` [list] | *(Required)* | One or more paths to PDF files (supports glob patterns e.g. `path/*.gr`). |
| `--max_comp` | `int` | `6` | Maximum number of NMF components to calculate (from 1 to `max_comp`). |
| `--rmin` | `float` | `None` | Minimum $r$-value (in Å) to include in the analysis. |
| `--rmax` | `float` | `None` | Maximum $r$-value (in Å) to include in the analysis. |
| `--outdir` | `str` | `nmf_results` | Target directory where all outputs and plots will be saved. |

---

## Input Data Format

Files provided to `--files` must be standard two-column ASCII text files (such as `.gr` files output by `PDFgetX3`, `PDFgetN3`, or `PDFgetX2`):
- **Column 1**: Radial distance $r$ in Angstroms (Å).
- **Column 2**: Pair distribution function $G(r)$.
- Header lines beginning with `#` or text are automatically handled by the `diffpy.utils` parser.
- All files in a given run must share the same $r$-grid spacing.

---

## Output Structure

Running the analysis produces the following directory layout in `--outdir`:

```text
nmf_results/
├── reconstruction_errors.txt
├── overview_reconstruction_and_components.png
├── 1_components/
│   ├── component_1.cgr
│   ├── weights_1_components.tex
│   └── reconstruction_plots_1c/
│       ├── sample_01_recon.png
│       └── ...
├── 2_components/
│   ├── component_1.cgr
│   ├── component_2.cgr
│   ├── weights_2_components.tex
│   └── reconstruction_plots_2c/
│       └── ...
└── ...
```

### Description of Output Files

- **`reconstruction_errors.txt`**: Tabulated Frobenius reconstruction error vs. number of components.
- **`overview_reconstruction_and_components.png`**: Dual-panel overview with the error curve (elbow plot) and vertically stacked component profiles for `--max_comp`.
- **`{n}_components/component_{i}.cgr`**: Extracted component $i$ exported with columns `r` and `G(r)`.
- **`{n}_components/weights_{n}_components.tex`**: LaTeX table containing normalized fractional weights for each sample and the sample scaling factor.
- **`reconstruction_plots_{n}c/{sample}_recon.png`**: Plot for each sample showing experimental data (circles), NMF reconstruction fit (red line), and the difference curve (green line offset below).

### Mathematical Note on Shifts & Weights

1. **Non-negativity & Baseline Shift**: Because NMF requires positive values, data is shifted by:
   $$\text{shift} = -\min(G_{\text{obs}}) \quad (\text{if } \min(G_{\text{obs}}) < 0)$$
   Reconstruction for plotting transforms back via:
   $$G_{\text{recon}} = W \times H - \text{shift}$$
2. **Normalized Weights**: The matrix $W$ contains weights. The exported LaTeX table reports fractional weights normalized such that each row sums to 1:
   $$w_{\text{frac}, ij} = \frac{W_{ij}}{\sum_{k} W_{ik}}$$
   The overall row sum $\sum_{k} W_{ik}$ is reported as the **Scale Factor**.

---

## Scientific Background & References

This implementation is based on the methodology published by Billinge and coworkers:

1. **Methodology & Foundation**:
   > Liu, C.-H., Wright, C. J., Gu, R., Bandi, S., Wustrow, A., Todd, P. K., O'Nolan, D., Beauvais, M. L., Neilson, J. R., Chupas, P. J., Chapman, K. W. & Billinge, S. J. L. (2021).  
   > *Non-negative matrix factorization for pair distribution function analysis.*  
   > **J. Appl. Cryst.** 54, 768–775.  
   > [DOI: 10.1107/S160057672100293X](https://doi.org/10.1107/S160057672100293X)

2. **Web Implementation & Peer-Reviewed Service**:
   > For an interactive web application, visit [PDFitc.org](https://pdfitc.org).  
   > Thatcher, Z., Liu, C.-H., Yang, L., McBride, B. C., Thinh Tran, G., Wustrow, A., Karlsen, M. A., Neilson, J. R., Ravnsbaek, D. B. & Billinge, S. J. L. (2022).  
   > *PDFitc: a web-based tool center for pair distribution function analysis.*  
   > **Acta Cryst.** A78, 242–248.  
   > [DOI: 10.1107/S205252062200234X](https://doi.org/10.1107/S205252062200234X)

---

## License

This project is licensed under the [MIT License](LICENSE).
