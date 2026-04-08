<img src="https://raw.githubusercontent.com/huiyang-yi/CausalCompass/main/images/icon4.png" width="150" align="left" vspace="1" />

# CausalCompass: Evaluating the Robustness of Time-Series Causal Discovery in Misspecified Scenarios



<p align="center">
  <a href="https://arxiv.org/abs/2602.07915"><img src="https://img.shields.io/badge/arXiv-2602.07915-b31b1b.svg" alt="arXiv"></a>
  <a href="https://causalcompass.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/badge/View-Documentation-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/causalcompass/"><img src="https://img.shields.io/pypi/v/causalcompass" alt="PyPI"></a>
  <a href="https://github.com/huiyang-yi/CausalCompass"><img src="https://img.shields.io/badge/GitHub-CausalCompass-black?logo=github" alt="GitHub"></a>
  <a href="https://github.com/huiyang-yi/CausalCompass/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

**CausalCompass** is a Python package that provides a flexible and extensible benchmark suite for evaluating the robustness of **time-series causal discovery (TSCD)** methods under **misspecified modeling assumptions**. For more details, please refer to the [Document](https://causalcompass.readthedocs.io/).

<p align="center">
  <img src="https://raw.githubusercontent.com/huiyang-yi/CausalCompass/main/images/auprc_performance_radar_plot.png" width="700" alt="AUPRC Performance Radar Plot">
</p>

---

## Abstract

Causal discovery from time series is a fundamental task in machine learning. However, its widespread adoption is hindered by a reliance on untestable causal assumptions and by the lack of robustness-oriented evaluation in existing benchmarks. To address these challenges, we propose **CausalCompass**, a flexible and extensible benchmark suite designed to assess the robustness of time-series causal discovery (TSCD) methods under violations of modeling assumptions. To demonstrate the practical utility of CausalCompass, we conduct extensive benchmarking of representative TSCD algorithms across eight assumption-violation scenarios. Our experimental results indicate that no single method consistently attains optimal performance across all settings. Nevertheless, the methods exhibiting superior overall performance across diverse scenarios are almost invariably deep learning-based approaches. We further provide hyperparameter sensitivity analyses to deepen the understanding of these findings. We also find, somewhat surprisingly, that NTS-NOTEARS relies heavily on standardized preprocessing in practice, performing poorly in the vanilla setting but exhibiting strong performance after standardization. Finally, our work aims to provide a comprehensive and systematic evaluation of TSCD methods under assumption violations, thereby facilitating their broader adoption in real-world applications.

## Key Features

- **8 assumption-violation scenarios**: Confounders, nonstationarity, measurement error, standardization, missing data, mixed data, min-max normalization, and trend/seasonality
- **2 vanilla models**: VAR (linear) and Lorenz-96 (nonlinear)
- **11 TSCD algorithms spanning 6 major methodological categories**:
  - **Granger causality-based**: VAR, LGC
  - **Constraint-based**: PCMCI
  - **Noise-based**: VARLiNGAM
  - **Score-based**: DYNOTEARS, NTS-NOTEARS
  - **Topology-based**: TSCI
  - **Deep learning-based**: cMLP, cLSTM, CUTS, CUTS+

---

## Datasets

The `datasets/` directory contains sample datasets. Complete datasets can be generated using the provided scripts. For convenience and reproducibility, the complete datasets archive is publicly available at
[Google Drive](https://drive.google.com/file/d/1jpggkKcT6cBc4YQT5bQYPj68pD4ImOj3/view?usp=sharing).

The generated datasets follow the naming convention:

```
[scenario]_[params]_[model]_p[p]_T[T]_[optional]_seed[seed].npz
```

Example: `confounder_rho0.5_VAR_p10_T1000_seed0.npz`

## Installation

### Install with pip

```bash
# 1. Create a clean conda environment
conda create -n causalcompass-env python=3.10 -y
conda activate causalcompass-env

# 2. Install causalcompass from PyPI
pip install causalcompass

# 3. Verify installation
pip show causalcompass
python -c "import causalcompass; print(dir(causalcompass))"
```

------

## Usage Example

```python
from causalcompass.datasets.measurement_error import simulate_var_with_measure_error
from causalcompass.algorithms import PCMCI

# Step 1: Generate VAR data with measurement error
p, T, lag, seed = 10, 500, 3, 0
gamma = 1.2  # measurement error scale factor

data, beta, true_adj = simulate_var_with_measure_error(
    p=p, T=T, lag=lag, gamma=gamma, seed=seed
)
print(f"Data shape: {data.shape}")              # (500, 10)
print(f"Ground truth shape: {true_adj.shape}")  # (10, 10)

# Step 2: Initialize and run the algorithm
model = PCMCI(tau_max=3, pc_alpha=0.05, alpha=0.05)
predicted_adj = model.run(data)

# Step 3: Evaluate
all_metrics, no_diag_metrics = model.eval(
    true_adj,
    predicted_adj,
    shd_thresholds=[0, 0.01, 0.05, 0.1, 0.3],
)
print(f"AUROC: {all_metrics['auroc']:.3f}")
print(f"AUPRC: {all_metrics['auprc']:.3f}")
print(f"NSHD: {all_metrics['shd']:.3f}")  # shd stores normalized SHD (NSHD)
print(f"AUROC (no diag): {no_diag_metrics['auroc']:.3f}")
print(f"AUPRC (no diag): {no_diag_metrics['auprc']:.3f}")
print(f"NSHD (no diag): {no_diag_metrics['shd']:.3f}")  # shd stores normalized SHD (NSHD)
```

---

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@misc{yi2026causalcompass,
  title   = {{CausalCompass}: Evaluating the Robustness of Time-Series Causal Discovery in Misspecified Scenarios},
  author  = {Yi, Huiyang and Shen, Xiaojian and Wu, Yonggang and Chen, Duxin and Wang, He and Yu, Wenwu},
  year    = {2026},
  note    = {Under review as a conference paper}
}
```

---

**Note:** The final bibliographic information (e.g., venue and proceedings details) will be updated upon paper acceptance.

## License

- The code in this repository is released under the [MIT License](./LICENSE).
- The datasets generated and provided by this repository are released under the [CC BY 4.0 License](./LICENSE-CC-BY-4.0).

---

## Contributing

Contributions are welcome! If you encounter bugs, have suggestions for improvements, or would like to extend CausalCompass with additional assumption-violation scenarios or evaluation protocols, please feel free to open an issue or submit a pull request.

## Contact

For questions or issues, please:

- Open an issue in this repository
- Email: yihuiyang@seu.edu.cn
