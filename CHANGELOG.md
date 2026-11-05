# Changelog

## 0.2.0

- Fix the direction of VARLiNGAM binary graphs and threshold sweeps; raw
  scores retain their existing orientation (`[i, j]` means `X_j -> X_i`).
- Apply sigmoid exactly once for CUTS+ NSHD evaluation. Raw `run()` outputs
  and AUROC/AUPRC calculations are unchanged. CUTS+ `shd_thresholds` now
  operate on probabilities, and the `shd` field continues to store NSHD.
- Remove GP missing-data imputation from the public package, including the
  bundled CUTS interpolation helper. ZOH and linear remain supported. GP
  experiments reported in the paper remain part of the separate research
  materials; this change does not remove those experiments or results.
- Honor `missing_prob` in batch generation when no missing configuration is
  supplied; explicit configurations remain authoritative.
- Synchronize abstracts, citations, dataset links, documentation version and
  project metadata with the camera-ready materials.
- Include the CUTS/CUTS+ configuration and missing-data updates previously
  available on GitHub but not in the PyPI 0.1.2 distribution.
