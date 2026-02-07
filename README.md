# CausalCompass: Evaluating the Robustness of Time-Series Causal Discovery in Misspecified Scenarios

**CausalCompass** is a flexible and extensible benchmark suite for evaluating the robustness of **time-series causal discovery (TSCD)** methods under **misspecified modeling assumptions**.

```
![Performance Comparison](images/nonlinear_15_500_f10_auprc_avg_scen_01.png)
```

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Generation](#data-generation)
- [Benchmark Scenarios](#benchmark-scenarios)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

CausalCompass includes:

- **8 assumption-violation scenarios**: Confounders, nonstationarity, measurement error, standardization, missing data, mixed data, min-max normalization, and trend/seasonality.
- **2 vanilla models**: VAR (linear) and Lorenz-96 (nonlinear).
- **11 TSCD algorithms spanning six major methodological categories**:
  - **Granger causality-based**: VAR, LGC
  - **Constraint-based**: PCMCI
  - **Noise-based**: VARLiNGAM
  - **Score-based**: DYNOTEARS, NTS-NOTEARS
  - **Topology-based**: TSCI
  - **Deep learning–based**: cMLP, cLSTM, CUTS, CUTS+
- **Reproducible experimental protocols** with multiple random seeds.
- **Automated experiment execution** via shell scripts for all algorithms.
- **Result processing utilities:**
  - LaTeX table generation for paper-ready results
  - Origin-compatible data export for radar plots

---

## Data Generation

All datasets can be generated using the scripts in the `data_generation/` directory.

### Quick Start

Generate all datasets for a specific scenario:

```bash
cd data_generation

# Vanilla datasets
python vanilla.py

# Assumption violation scenarios
python confounder.py
python measurement_error.py
python missing.py
python mixed_data.py
python nonstationary.py
python standardized.py		# Includes z-score and min-max normalization
python trendseason.py
```

###Example: Generate Measurement Error Data

```python
from data_generation.measurement_error import simulate_var_with_measure_error

# Generate VAR data with measurement error
p = 10          # Number of variables
T = 1000        # Time steps
lag = 3         # Lag order
gamma = 1.2    # Error variance = 1.2 × data variance
seed = 0 	    # Random seed for reproducibility

data, beta, gc = simulate_var_with_measure_error(
    p=p, T=T, lag=lag, gamma=gamma, seed=seed
)

print(f"Data shape: {data.shape}")        # (1000, 10)
print(f"Ground truth GC: {gc.shape}")     # (10, 10)
```

### Generated Data Structure

Generated datasets will be saved in the following structure:
```
datasets/
├── vanilla/
├── confounder/
├── measurement_error/
├── missing/
├── mixed_data/
├── nonstationary/
├── standardized/
└── trendseason/
```

The generated datasets follow the naming convention:

```
[scenario]_[params]_[model]_p[p]_T[T]_[optional]_seed[seed].npz
```

Example: `confounder_rho0.5_VAR_p10_T1000_seed0.npz`

Each `.npz` file contains:

- `data`: Time series observations (T × D)
- `gc`: Ground truth Granger causality graph (D × D)
- Additional scenario-specific metadata

The `datasets/` directory contains sample datasets. Complete datasets will be provided upon acceptance.

---

## Benchmark Scenarios

### 1. Vanilla

Standard VAR and Lorenz-96 systems without assumption violations.

### 2. Confounders
Hidden confounders create spurious correlations between observed variables.

### 3. Measurement Error
Gaussian noise proportional to data variance is added to observations.

### 4. Missing Data
Random missing values with specified probability, interpolated using zero-order hold.

### 5. Mixed Data

Mixture of continuous and discrete variables.

### 6. Nonstationarity

Time-varying noise variance and time-varying coefficients.

### 7. Standardized Data

Z-score and min-max normalization applied to time series.

### 8. Trend and Seasonality

Trends and seasonal patterns added to observations.

---

## Running Experiments

### Automated Experiment Execution

Run all TSCD algorithms automatically using the provided shell scripts:

```
# Navigate to scripts directory
cd scripts

# Run all experiments (11 algorithms)
chmod +x run_all.sh
./run_all.sh

# Or run individual algorithms
chmod +x run_*.sh
./run_var.sh        # VAR
./run_lgc.sh        # LGC
./run_pcmci.sh      # PCMCI
./run_varlingam.sh  # VARLiNGAM
./run_dynotears.sh  # DYNOTEARS
./run_ntsnotears.sh # NTS-NOTEARS
./run_tsci.sh       # TSCI
./run_ngc.sh        # NGC (cMLP and cLSTM)
./run_cuts.sh       # CUTS
./run_cutsplus.sh   # CUTS+
```

The `run_all.sh` script orchestrates all 11 algorithms and handles:

- Automatic error detection and reporting
- Progress tracking with timestamps
- Failed script counting and exit code management

**Note**: Results are saved in JSON format with performance metrics (AUPRC, AUROC) and hyperparameter configurations.

## Result Analysis

### Generating LaTeX Tables

Convert experimental results to publication-ready LaTeX tables:

```
python result2latex.py
```

This generates:

- **Comparison tables** across all scenarios and methods
- **Performance metrics** (AUPRC/AUROC) with best results highlighted
- Separate tables for VAR and Lorenz-96 with different parameters

Output files: `table_VAR_p10_T1000.tex`, `table_Lorenz_p10_T1000_F10.tex`, etc.

### Generating Origin Data Files

Export results for radar plots and visualization:

```
python generate_origin_tables.py
```

These scripts generate `.txt` files compatible with Origin for creating:

- **Radar plots** comparing method performance across scenarios
- **Hyperparameter sensitivity** visualizations

## Project Structure

```
CausalCompass/
│
├── algs/                   # Algorithm implementations
│   ├── cuts/               # CUTS implementation
│   ├── cutsplus/           # CUTS+ implementation
│   ├── lgc/                # LGC implementation
│   ├── ngc/                # NGC implementation
│   ├── ntsnotears/         # NTS-NOTEARS implementation
│   ├── tsci/               # TSCI implementation
│   ├── var/                # VAR implementation
│   ├── varlingam/          # VARLiNGAM implementation
│   └── __init__.py         # Package initialization
│
├── data_generation/        # Data generation scripts
│   ├── vanilla.py          # VAR and Lorenz-96
│   ├── confounder.py       # Confounders scenario
│   ├── measurement_error.py # Measurement error scenario
│   ├── missing.py          # Missing data scenario
│   ├── mixed_data.py       # Mixed data scenario
│   ├── non_gaussian.py     # Non-Gaussian noise scenario
│   ├── nonstationary.py    # Nonstationarity scenario
│   ├── standardized.py     # z-score and min-max scenario
│   └── trendseason.py      # Trend and seasonality scenario
│
├── datasets/               # Generated datasets 
│   └── [scenario]/         # Organized by scenario
│
├── scripts/                # Experiment execution scripts
│   ├── run_all.sh         # Master script to run all experiments
│   ├── run_var.sh         # VAR experiments
│   ├── run_lgc.sh         # LGC experiments
│   ├── run_pcmci.sh       # PCMCI experiments
│   ├── run_varlingam.sh   # VARLiNGAM experiments
│   ├── run_dynotears.sh   # DYNOTEARS experiments
│   ├── run_ntsnotears.sh  # NTS-NOTEARS experiments
│   ├── run_tsci.sh        # TSCI experiments
│   ├── run_ngc.sh         # NGC experiments
│   ├── run_cuts.sh        # CUTS experiments
│   └── run_cutsplus.sh    # CUTS+ experiments
│
├── result2latex.py         # Generate LaTeX tables from results
├── generate_origin_tables.py              # Generate Origin data files
│
└── README.md               # This file
```

---

## Key Findings

Our comprehensive evaluation reveals:

1. **No universal winner**: No single TSCD method achieves optimal performance across all assumption-violation scenarios.

2. **Deep learning advantages**: Methods exhibiting superior overall robustness are predominantly deep learning-based approaches.

3. **Preprocessing sensitivity**: NTS-NOTEARS shows strong dependence on standardization, performing poorly on vanilla data but excelling after standardization.

4. **Hyperparameter tuning**: Deep learning methods require more careful hyperparameter selection in linear settings but demonstrate greater stability in nonlinear scenarios.

---

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@misc{causalcompass2026_anonymous,
  title={CausalCompass: Evaluating the Robustness of Time-Series Causal Discovery in Misspecified Scenarios},
  author={Anonymous},
  year={2026},
  note={Under review as a conference paper},
}
```

---

**Note:**  This citation is anonymized for the review process. The complete author list and final bibliographic information will be released upon acceptance.

## License

- The code in this repository is released under the [MIT License](./LICENSE).
- The datasets generated and provided by this repository are released under the [CC BY 4.0 License](./LICENSE-CC-BY-4.0).

---

## Contributing

Contributions are welcome! If you encounter bugs, have suggestions for improvements, or would like to extend CausalCompass with additional assumption-violation scenarios or evaluation protocols, please feel free to open an issue or submit a pull request. 

## Availability

The full codebase will be publicly released upon paper acceptance.

## Contact

For questions or issues, please open an issue in this repository.

**Note**: This repository is anonymized for review. Full author information will be added upon acceptance.
