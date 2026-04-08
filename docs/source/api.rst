.. automodule:: causalcompass

API Reference
=============

Import causalcompass as::

   import causalcompass

CausalCompass provides data generation functions for both VAR (linear) and Lorenz-96 (nonlinear) models across multiple assumption-violation scenarios. 
All data generation functions are located under ``causalcompass.datasets``.
All algorithms are located under ``causalcompass.algorithms``.

Data Generation
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: api
   :nosignatures:

   datasets.vanilla.simulate_var
   datasets.vanilla.simulate_lorenz_96
   datasets.measurement_error.simulate_var_with_measure_error
   datasets.measurement_error.simulate_lorenz_with_measure_error
   datasets.mixed_data.generate_mixed_var
   datasets.mixed_data.generate_mixed_lorenz_96
   datasets.standardized.generate_standardized_var
   datasets.standardized.generate_standardized_lorenz_96
   datasets.confounder.simulate_var_with_confounders
   datasets.confounder.simulate_lorenz_with_confounders
   datasets.missing.generate_missing_var
   datasets.missing.generate_missing_lorenz_96
   datasets.trendseason.simulate_var_with_trend_season
   datasets.trendseason.simulate_lorenz_with_trend_season
   datasets.nonstationary.simulate_nonstationary_var
   datasets.nonstationary.simulate_nonstationary_var_timevarying_coef
   datasets.nonstationary.simulate_nonstationary_lorenz_96

Algorithms
^^^^^^^^^^

.. autosummary::
   :toctree: api
   :nosignatures:

   algorithms.VAR
   algorithms.LGC
   algorithms.VARLiNGAM
   algorithms.PCMCI
   algorithms.DyNotears
   algorithms.NTSNotears
   algorithms.TSCI
   algorithms.CMLP
   algorithms.CLSTM
   algorithms.CUTS
   algorithms.CUTSPlus