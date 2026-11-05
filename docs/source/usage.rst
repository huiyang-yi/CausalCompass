Usage
=====

Usage Example 1: Single Algorithm on a Single Scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to run a causal discovery algorithm on a single scenario, covering the complete workflow of data generation, algorithm execution, and evaluation.

.. code-block:: python

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


Usage Example 2: All Algorithms on a Single Scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example benchmarks all 11 TSCD algorithms under the measurement error scenario.

.. code-block:: python

   import torch
   from causalcompass.datasets.measurement_error import simulate_var_with_measure_error
   from causalcompass.algorithms import (
       PCMCI, DyNotears, CMLP, CLSTM,
       VARLiNGAM, CUTS, CUTSPlus, NTSNotears,
       LGC, VAR, TSCI
   )

   # Generate measurement error data
   p, T, lag, seed = 10, 500, 3, 0
   gamma = 1.2
   data, beta, true_adj = simulate_var_with_measure_error(
       p=p, T=T, lag=lag, gamma=gamma, seed=seed
   )
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Define all 11 algorithms with their parameters
   algorithms = [
       ('PCMCI',      PCMCI,      {'tau_max': 3, 'pc_alpha': 0.05, 'alpha': 0.05}),
       ('VARLiNGAM',  VARLiNGAM,  {'tau_max': 3, 'varlingamalpha': 0.01}),
       ('DyNotears',  DyNotears,  {'tau_max': 3, 'wthre': 0.01, 'lambda_w': 0.1, 'lambda_a': 0.1}),
       ('NTSNotears', NTSNotears, {'tau_max': 3, 'wthre': 0.01, 'lambda_1': 0.001, 'lambda_2': 0.01, 'device': device}),
       ('TSCI',       TSCI,       {'theta': 0.5, 'fnn_tol': 0.01}),
       ('VAR',        VAR,        {'tau_max': 3, 'threshold': 0.01}),
       ('LGC',        LGC,        {'tau_max': 3, 'threshold': 0.01, 'lgc_alphas': [1e-4, 5e-3, 1e-2, 2e-2, 5e-2]}),
       ('cMLP',       CMLP,       {'lag': 3, 'hidden_dim': [100], 'lam': 0.005, 'lr': 0.01, 'max_iter': 50000, 'device': device}),
       ('cLSTM',      CLSTM,      {'context': 10, 'hidden_dim': 100, 'lam': 0.005, 'lr': 0.01, 'max_iter': 20000, 'device': device}),
       ('CUTS',       CUTS,       {'input_step': 10, 'batch_size': 32, 'weight_decay': 0.001, 'device': device}),
       ('CUTSPlus',   CUTSPlus,   {'input_step': 10, 'batch_size': 32, 'weight_decay': 0.001, 'device': device}),
   ]

   # Run and evaluate each algorithm
   for name, cls, params in algorithms:
       model = cls(seed=seed, **params)

       # CUTS and CUTSPlus require the true causal matrix in run()
       if name in ['CUTS', 'CUTSPlus']:
           predicted_adj = model.run(data, true_cm=true_adj)
       else:
           predicted_adj = model.run(data)

       all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
       print(f"[{name}] AUROC={all_metrics['auroc']:.3f}, AUPRC={all_metrics['auprc']:.3f}, NSHD={all_metrics['shd']:.3f} | "
             f"No-Diag AUROC={no_diag_metrics['auroc']:.3f}, AUPRC={no_diag_metrics['auprc']:.3f}, NSHD={no_diag_metrics['shd']:.3f}")  # shd stores normalized SHD (NSHD)
