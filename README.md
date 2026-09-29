# Aviation Insight: Anomaly Detection in Flight Data
**PRAXIS Thesis - AI Workstation: Theseus**

## 📌 Overview
This project evaluates multiple anomaly detection models against flight telemetry data to improve safety and operational insights. Developed on an **RTX 5090 (Blackwell)** workstation using a multi-tier storage strategy.

## 🏗️ Hardware Architecture (Theseus)
- **Tier 1 (OS/Apps):** 4TB NVMe
- **Tier 2 (Active Code):** 500GB SSD RAID 1 (`/mnt/fast_storage`)
- **Tier 3 (Bulk Data/Cache):** 4TB HDD RAID 1 (`/mnt/bulk_data`)

## 📂 Project Structure
- `src/`: Core Python package for model architectures and data loaders.
- `notebooks/`: Exploratory Data Analysis (EDA) and Colab Pro prototypes.
- `data/`: (Symlinked to HDD) Raw, interim, and processed flight datasets.
- `models/`: (Symlinked to HDD) Training checkpoints and final evaluated models.
- `reports/`: Thesis documentation and performance visualizations.

## 📂 Project Structure Details
aviation_insight/   
├── data/                # Symbolic link to /mnt/bulk_data/raw_datasets/aviation_insight   
│   ├── external/        # Data from third-party sources (e.g., FAA, NOAA)   
│   ├── interim/         # Intermediate data that has been transformed   
│   ├── processed/       # The final, canonical data sets for modeling   
│   └── raw/             # The original, immutable data dump   
├── docs/                # Thesis documents, PRAXIS requirements, and literature review   
├── models/              # Symbolic link to /mnt/bulk_data/models/aviation_insight   
│   ├── checkpoints/     # Weights saved during training (RTX 5090 outputs)   
│   └── final/           # Finalized models for evaluation   
├── notebooks/           # Jupyter notebooks for EDA and Colab prototyping   
│   ├── exploratory/     # "Scratchpad" notebooks   
│   └── reports/         # Notebooks used to generate thesis visualizations   
├── reports/             # Final PRAXIS report, LaTeX files, and figures   
│   └── figures/         # Generated graphics for the thesis   
├── src/                 # The actual Python package (reusable code)   
│   ├── __init__.py   
│   ├── data_loader.py   # Scripts to turn raw data into tensors   
│   ├── models/          # Model architectures (VAE, Isolation Forest, etc.)   
│   ├── training/        # Training loops and loss functions   
│   └── visualization.py # Functions for plotting anomaly scores   
├── tests/               # Unit tests for your anomaly detection logic   
├── workflows/           # DVC pipelines or shell scripts for batch processing   
├── .gitignore           # Keeps the junk out of your Gitea/GitLab   
├── dvc.yaml             # DVC pipeline definition   
└── requirements.txt     # Python dependencies   

## 🚀 MLOps Workflow
1. **Version Control:** Managed via local Gitea instance (Privacy First).
2. **Data Versioning:** DVC is used to track datasets; the cache is locked to the 4TB HDD RAID.
3. **Cloud Bridge:** Google Colab Pro connects via **Tailscale** to fetch code and **GCS** to pull DVC-tracked data.

## 🛠️ Setup
1. **Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
