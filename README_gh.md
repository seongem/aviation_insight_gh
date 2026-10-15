# Aviation Insight: Anomaly Detection in Flight Data

## 📌 Overview

This project evaluates multiple anomaly detection models against flight telemetry data to improve safety and operational insights. This repository
partially represents the code used as part of praxis study. Additionally, it provides some reference material about different aspects of
the analysis in the Reference papers sectin below.

## Citation

If you use any of this work in academic papers or other studies, please use the following citation:
**TBD**

## Reference papers

- [Flight Phase Identification and Other Topics](reports/papers/Flight%20Phase%20Identification.pdf) paper (see reports/papers)

## 📂 Project Structure

- `configs`: Contains configuration files used for the various experiments
- `data_loader`: Provides the data loader class for the NASA Dashlink Four Class Anomaly Dataset
- `manifests`: Provides the manifest files that manage the feature matrix, flights and phases
- `metadata/stats_v2`: Provides the metadata used for fleetwide statistics and training split statistics
- `models/`: Training checkpoints and final evaluated models
- `src/`: Core generation scripts and utilities for study
- `reports/`: Thesis documentation and performance visualizations
- `workflows/`: Contains supporting generation and plotting tools
- `README_md`: This file
- `evaluate_4c.py`: Evaluation script for all models
- `requirements.text`: Python supporting modules
- `train_4c.py`: Train script for all models
