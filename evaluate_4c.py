import os
import sys
import yaml
import json
import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import seaborn as sns

# Internal Project Imports
from data_loader.aviation_four_class_dataset import AviationFourClassDataset
from evaluation.anomaly_four_class_evaluator import AnomalyFourClassEvaluator
from src.utils.load_environment import LoadEnvironment
from src.utils.anomaly_support import visualize_and_save_top_anomalies
from src.utils.palette_manager import theme

# Model Architecture Imports
from models.conventional_autoencoder import ConvAutoencoder
from models.omni_anomaly import OmniAnomaly
from models.lstm_vae import LSTMVAE
from models.patch_tst import PatchTST
from models.anomaly_transformer import AnomalyTransformer
from models.ts2vec import TSEncoder

# -------------------------------------------------------------------------
# Logger Setup
# -------------------------------------------------------------------------
from tools.diagnostic_logger import DiagnosticFileHandler

logger = logging.getLogger("AviationInsight")
ISSUE_TOKEN = "EVALUATE_MODELS"


# -------------------------------------------------------------------------
# 1. Pipeline Evaluator Engine
# -------------------------------------------------------------------------
def evaluate_unsupervised_model(
    model_name,
    evaluator,
    val_loader,
    test_loader,
    percentile=95.0,
    metadata_df=None,
    top_n_reports=20,
):
    """
    Executes leak-free two-stage thresholding:
    Stage A: Calibrates tau on Validation nominals at specified percentile (e.g., 95th)
    Stage B: Benchmarks Test set using static calibrated tau and extracts full diagnostic reports.
    """
    # Stage A: Calibrate threshold tau on validation split
    val_metrics = evaluator.get_metrics(val_loader, auto_percentile=percentile)
    tau_calibrated = val_metrics["Calibrated_Tau"]

    # Stage B: Full evaluation on test set using calibrated tau
    test_metrics = evaluator.get_metrics(
        test_loader,
        threshold=tau_calibrated,
        metadata_df=metadata_df,
        top_n_reports=top_n_reports,
        compute_gradients=True,  # Required for sensor-level attributions
    )

    return {
        "Model": model_name,
        "AUROC": test_metrics["AUROC"],
        "PR-AUC": test_metrics["PR-AUC"],
        "Accuracy": (test_metrics["TP"] + test_metrics["TN"])
        / (test_metrics["TP"] + test_metrics["TN"] + test_metrics["FP"] + test_metrics["FN"]),
        "Binary F1": test_metrics["F1_Binary"],
        "Macro F1": test_metrics["F1_Macro"],
        "Binary F2": test_metrics["F2_Binary"],
        "Precision": test_metrics["Prec"],
        "Recall": test_metrics["Rec"],
        "FPR": test_metrics["FPR"],
        "FNR": test_metrics["FNR"],
        "HitRate1": test_metrics["HR1"],
        "Calibrated Tau": tau_calibrated,
        "TN": test_metrics["TN"],
        "FP": test_metrics["FP"],
        "FN": test_metrics["FN"],
        "TP": test_metrics["TP"],
        "cm_matrix": np.array([
            [test_metrics["TN"], test_metrics["FP"]],
            [test_metrics["FN"], test_metrics["TP"]],
        ]),
        # Structured Diagnostics & Attributions
        "Reports": test_metrics["Reports"],
        "top_indices": test_metrics["top_indices"],
        "y_scores": test_metrics["y_scores"],
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
        "y_scores_norm": test_metrics["y_scores_norm"],
        "y_errors": test_metrics["y_errors"],
        "sensor_attributions": test_metrics["sensor_attributions"],
    }


import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------------
# 2. Confusion Matrix Plotting & Summarization
# -------------------------------------------------------------------------
def display_and_save_confusion_matrices(results_dict, output_dir, mask_name, max_cols=3):
    os.makedirs(output_dir, exist_ok=True)

    table_rows = []
    for model_name, res in results_dict.items():
        table_rows.append({
            "Mask": mask_name,
            "Model": model_name,
            "TN (True Nom)": res["TN"],
            "FP (False Anom)": res["FP"],
            "FN (Missed Anom)": res["FN"],
            "TP (True Anom)": res["TP"],
            "Binary F1": round(res["Binary F1"], 4),
            "Macro F1": round(res["Macro F1"], 4),
            "PR-AUC": round(res["PR-AUC"], 4),
            "AUROC": round(res.get("AUROC", 0.0), 4),
        })

    df_cm = pd.DataFrame(table_rows).set_index(["Mask", "Model"])
    logger.info("CONFUSION MATRIX SUMMARY [%s]:\n%s", mask_name, df_cm.to_string())

    num_models = len(results_dict)
    if num_models == 0:
        return

    cols = min(num_models, max_cols)
    rows = math.ceil(num_models / cols)

    # Calculate optimal grid layout (e.g., 2 x 4 for 8 models)
    cols = min(num_models, max_cols)
    rows = math.ceil(num_models / cols)

    # Use theme width constraint (7.0") for perfect 8.5" x 11" page fit
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(theme.PAGE_WIDTH_INCHES, 3.2 * rows),
        squeeze=False,
    )
    model_items = list(results_dict.items())

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        if i < num_models:
            model_name, res = model_items[i]
            sns.heatmap(
                res["cm_matrix"],
                annot=True,
                fmt="d",
                cmap=theme.cmap_blue,  # <-- Okabe-Ito Blue colormap
                cbar=False,
                square=True,
                ax=ax,
                annot_kws={"size": 9, "weight": "bold"},
                xticklabels=["Pred Nom (0)", "Pred Anom (1)"],
                yticklabels=["True Nom (0)", "True Anom (1)"],
            )

            # Lock rotation horizontal to prevent squishing the heatmap box
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)
            ax.tick_params(axis="y", labelrotation=90, labelsize=8)

            # Re-align Y labels vertically centered
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=90, va="center", ha="right", fontsize=8
            )

            ax.set_title(
                f"{model_name}\n" rf"$\mid{{Macro F1: {res['Macro F1']:.4f} | PR-AUC:}}$"
                f" {res['PR-AUC']:.4f}",
                fontsize=9,
                fontweight="bold",
            )

            if r == rows - 1 or i >= num_models - cols:
                ax.set_xlabel("Predicted", fontsize=8)
            else:
                ax.set_xlabel("")

            if c == 0:
                ax.set_ylabel("Actual Ground Truth")
            else:
                ax.set_ylabel("")
        else:
            ax.axis("off")  # Cleanly hide extra subplots

    plt.suptitle(
        f"Evaluation Confusion Matrices — Feature Mask: {mask_name}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"confusion_matrices_{mask_name}.png")
    plt.savefig(save_path, dpi=theme.DPI, bbox_inches="tight")
    logger.info("Saved heatmaps for [%s] to: %s", mask_name, save_path)
    plt.close()


# -------------------------------------------------------------------------
# 3. Helper: Clean State Dict Keys & JSON Metric Reader
# -------------------------------------------------------------------------
def load_clean_state_dict(ckpt_path, device):
    # 1. Register NumPy scalar type for PyTorch safe unpickling
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    
    # 2. Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 3. Resolve nested state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
    else:
        raw_state_dict = checkpoint

    # 4. Clean DDP, torch.compile, SWA, and THOP profiling metadata keys
    cleaned_state_dict = {}
    for k, v in raw_state_dict.items():
        # Filter out SWA metadata
        if k == "n_averaged" or k.endswith(".n_averaged"):
            continue
            
        # Filter out THOP profiling buffers
        if k in ("total_ops", "total_params") or k.endswith(".total_ops") or k.endswith(".total_params"):
            continue
            
        new_key = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned_state_dict[new_key] = v
        
    return cleaned_state_dict, checkpoint


def load_checkpoint_metrics_json(ckpt_dir: Path, filename: str = "best_model_metrics.json") -> dict:
    """Helper to load companion compute and validation metrics saved by train_4c.py."""
    json_path = ckpt_dir / filename
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse JSON metrics at {json_path}: {e}")
    return {}


# -------------------------------------------------------------------------
# 4. Main Execution Routine
# -------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Evaluation Engine on Device: %s", device)

    # Load Environment
    env = LoadEnvironment()
    checkpoint_dir = Path(env.CHECKPOINT_DIR)
    plots_dir = os.path.join(env.OUTPUT_DIR, "evaluation_plots")

    # Load experiment configuration YAML
    config_path = os.path.join(env.ROOT_BASE_DIR, "configs", "experiments.yaml")

    logger.info("Loading Experiment Config from: %s", config_path)
    with open(config_path, "r") as f:
        exp_cfg = yaml.safe_load(f)

    percentile_tau = exp_cfg.get("evaluation", {}).get("percentile", 95.0)
    feature_masks = exp_cfg.get("feature_masks", ["enabled_all"])
    batch_size = exp_cfg.get("defaults", {}).get("batch_size", 512)
    num_workers = exp_cfg.get("defaults", {}).get("num_workers", 4)

    all_mask_summary_rows = []

    # Extract LOGO configurations
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)
    inf_logo_masks = exp_cfg.get("logo_masks", {}).get("inference_time", None)

    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]
    if not isinstance(inf_logo_masks, list):
        inf_logo_masks = [inf_logo_masks]

    for mask_name in feature_masks:
        for train_logo in train_logo_masks:
            train_logo_tag = f"_logo_{train_logo}" if train_logo else ""
            model_ckpt_dir = f"{mask_name}{train_logo_tag}"

            for inf_logo in inf_logo_masks:
                test_logo_tag = f"_testlogo_{inf_logo}" if inf_logo else ""
                run_identifier = f"{model_ckpt_dir}{test_logo_tag}"

                logger.info(
                    f"\n=== EVAL | Feature Set: {mask_name} | Train LOGO: {train_logo or 'NONE'} | Test LOGO: {inf_logo or 'NONE'} ==="
                )

                # 1. Validation Dataset (Used exclusively to calibrate Tau / threshold)
                val_ds = AviationFourClassDataset(
                    manifest_dir=env.DATABASE_DIR / "manifests",
                    database_dir=env.DATABASE_DIR,
                    stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                    split="val",
                    feature_set=mask_name,
                    logo_mask_col=inf_logo,
                )

                # 2. Test Dataset (Used for final benchmark evaluation)
                test_ds = AviationFourClassDataset(
                    manifest_dir=env.DATABASE_DIR / "manifests",
                    database_dir=env.DATABASE_DIR,
                    stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                    split="test",
                    feature_set=mask_name,
                    logo_mask_col=inf_logo,
                )

                # 3. Train Dataset (Used for TS2Vec kNN for memory bank)
                train_ds = AviationFourClassDataset(
                    manifest_dir=env.DATABASE_DIR / "manifests",
                    database_dir=env.DATABASE_DIR,
                    stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                    split="train",
                    feature_set=mask_name,
                    logo_mask_col=inf_logo,
                )

                val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

                in_channels = val_ds.num_channels
                seq_len = val_ds.window_size
                input_shape = (seq_len, in_channels)

                logger.info(
                    f"Verified Dataset | Mask: {mask_name} | Channels: {in_channels} | Window Size: {seq_len}"
                )

                m_cfg = exp_cfg.get("models", {})

                patch_len = m_cfg.get("patchtst", {}).get("patch_len", 16)
                stride = m_cfg.get("patchtst", {}).get("stride", 8)
                num_patch = (seq_len - patch_len) // stride + 1

                models_config = {
                    "TS2Vec (Masking Drift)": {
                        "model_cls": TSEncoder,
                        "kwargs": {
                            "input_dims": in_channels,
                            "output_dims": m_cfg.get("ts2vec", {}).get("output_dims", 320),
                            "hidden_dims": m_cfg.get("ts2vec", {}).get("hidden_dims", 64),
                            "depth": m_cfg.get("ts2vec", {}).get("depth", 10),
                        },
                        "ckpt_base": "ts2vec",
                        "type": "ts2vec",
                    },
                    "TS2Vec (kNN-Memory Bank)": {
                        "model_cls": TSEncoder,
                        "kwargs": {
                            "input_dims": in_channels,
                            "output_dims": m_cfg.get("ts2vec", {}).get("output_dims", 320),
                            "hidden_dims": m_cfg.get("ts2vec", {}).get("hidden_dims", 64),
                            "depth": m_cfg.get("ts2vec", {}).get("depth", 10),
                        },
                        "ckpt_base": "ts2vec",
                        "type": "ts2vec_knn",
                    },
                    "LSTM-VAE": {
                        "model_cls": LSTMVAE,
                        "kwargs": {
                            "input_dim": in_channels,
                            "hidden_dim": m_cfg.get("lstm_vae", {}).get("hidden_dim", 128),
                            "latent_dim": m_cfg.get("lstm_vae", {}).get("latent_dim", 32),
                            "num_layers": m_cfg.get("lstm_vae", {}).get("num_layers", 2),
                            "dropout": m_cfg.get("lstm_vae", {}).get("dropout", 0.1),
                        },
                        "ckpt_base": "lstm_vae",
                        "type": "lstm_vae",
                    },
                    "OmniAnomaly": {
                        "model_cls": OmniAnomaly,
                        "kwargs": {
                            "n_features": in_channels,
                            "window_size": seq_len,
                            "latent_dim": m_cfg.get("omni", {}).get("latent_dim", 32),
                            "rnn_hidden": m_cfg.get("omni", {}).get("rnn_hidden", 128),
                        },
                        "ckpt_base": "omni",
                        "type": "omni",
                    },
                    "ConvAutoencoder": {
                        "model_cls": ConvAutoencoder,
                        "kwargs": {
                            "input_shape": input_shape,
                        },
                        "ckpt_base": "conv_ae",
                        "type": "conv_ae",
                    },
                    "PatchTST": {
                        "model_cls": PatchTST,
                        "kwargs": {
                            "c_in": in_channels,
                            "target_dim": seq_len,
                            "patch_len": patch_len,
                            "stride": stride,
                            "num_patch": num_patch,
                            "head_type": "pretrain",
                            "d_model": m_cfg.get("patchtst", {}).get("d_model", 64),
                            "n_layers": m_cfg.get("patchtst", {}).get("e_layers", 3),
                            "n_heads": m_cfg.get("patchtst", {}).get("n_heads", 4),
                            "dropout": m_cfg.get("patchtst", {}).get("dropout", 0.1),
                            "attn_dropout": m_cfg.get("patchtst", {}).get("attn_dropout", 0.1),
                            "head_dropout": m_cfg.get("patchtst", {}).get("head_dropout", 0.2),
                        },
                        "ckpt_base": "patchtst",
                        "type": "patchtst",
                    },
                    "Anomaly Transformer": {
                        "model_cls": AnomalyTransformer,
                        "kwargs": {
                            "win_size": seq_len,
                            "enc_in": in_channels,
                            "c_out": in_channels,
                            "d_model": m_cfg.get("anomaly_transformer", {}).get("d_model", 128),
                            "n_heads": m_cfg.get("anomaly_transformer", {}).get("n_heads", 8),
                            "e_layers": m_cfg.get("anomaly_transformer", {}).get("e_layers", 3),
                            "d_ff": m_cfg.get("anomaly_transformer", {}).get("d_ff", 256),
                            "dropout": m_cfg.get("anomaly_transformer", {}).get("dropout", 0.1),
                        },
                        "ckpt_base": "anomaly_transformer",
                        "type": "anomaly_transformer",
                    },
                }

                mask_results = {}

                # Iterative Model Benchmarking
                for model_name, cfg in models_config.items():
                    target_dir = checkpoint_dir / f"{cfg['ckpt_base']}_{model_ckpt_dir}"
                    mask_ckpt_path = target_dir / "best_model.pt"
                    base_ckpt_path = checkpoint_dir / f"{cfg['ckpt_base']}.pt"

                    if mask_ckpt_path.exists():
                        ckpt_path = mask_ckpt_path
                        run_dir = target_dir
                    elif base_ckpt_path.exists():
                        ckpt_path = base_ckpt_path
                        run_dir = checkpoint_dir
                    else:
                        logger.warning(
                            "Checkpoint for %s [%s] not found at %s or %s. Skipping...",
                            model_name,
                            mask_name,
                            mask_ckpt_path,
                            base_ckpt_path,
                        )
                        continue

                    logger.info("Loading Checkpoint for %s [%s]: %s", model_name, mask_name, ckpt_path)

                    model = cfg["model_cls"](**cfg["kwargs"]).to(device)
                    state_dict, raw_ckpt = load_clean_state_dict(ckpt_path, device)
                    model.load_state_dict(state_dict)
                    model.eval()

                    # Read companion JSON metrics if available
                    json_metrics = load_checkpoint_metrics_json(run_dir, "best_model_metrics.json")

                    # Fall back to checkpoint dictionary if JSON isn't present
                    ckpt_comp = raw_ckpt.get("computational_metrics", {}) if isinstance(raw_ckpt, dict) else {}

                    feature_names = test_ds.get_feature_names()
                    metadata_df = test_ds.get_metadata()

                    evaluator_kwargs = {
                        "model": model,
                        "model_type": cfg["type"],
                        "device": device.type,
                        "feature_names": feature_names,
                        "metadata_df": metadata_df,
                    }

                    if cfg["type"] == "patchtst":
                        evaluator_kwargs["patch_len"] = patch_len
                        evaluator_kwargs["stride"] = stride
                    elif cfg["type"] == "ts2vec_knn":
                        evaluator_kwargs["train_loader"] = train_loader

                    evaluator = AnomalyFourClassEvaluator(**evaluator_kwargs)

                    res = evaluate_unsupervised_model(
                        model_name=model_name,
                        evaluator=evaluator,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        percentile=percentile_tau,
                        metadata_df=metadata_df,
                        top_n_reports=20,
                    )
                    mask_results[model_name] = res

                    # Combine ML test evaluation results with stored computational metrics
                    all_mask_summary_rows.append({
                        "Feature Mask": mask_name,
                        "Model": model_name,
                        "AUROC": res["AUROC"],
                        "PR-AUC": res["PR-AUC"],
                        "Accuracy": res["Accuracy"],
                        "Binary F1": res["Binary F1"],
                        "Macro F1": res["Macro F1"],
                        "Binary F2": res["Binary F2"],
                        "Precision": res["Precision"],
                        "Recall": res["Recall"],
                        "FPR": res["FPR"],
                        "FNR": res["FNR"],                       
                        "Calibrated Tau": res["Calibrated Tau"],
                        # Computational / Hardware Benchmarks (from JSON or raw checkpoint)
                        "Params (M)": json_metrics.get("params_m", ckpt_comp.get("params_m", np.nan)),
                        "FLOPs (G)": json_metrics.get("forward_gflops", ckpt_comp.get("forward_gflops", np.nan)),
                        "Latency (ms)": json_metrics.get("inference_latency_ms", ckpt_comp.get("inference_latency_ms", np.nan)),
                        "Throughput (samp/s)": json_metrics.get("throughput_samples_per_sec", ckpt_comp.get("throughput_samples_per_sec", np.nan)),
                        "Peak VRAM (GB)": json_metrics.get("peak_vram_gb", ckpt_comp.get("peak_vram_gb", np.nan)),
                    })

                    # visualize per model and run_identifier
                    visualize_and_save_top_anomalies(
                        model_name=model_name,
                        res=res,
                        test_ds=test_ds,
                        metadata_df=metadata_df,
                        feature_names=feature_names,
                        output_dir=str(run_dir),
                        run_identifier=run_identifier,
                        top_k=3,
                        top_n=5,
                    )
                # display for all models within this run_identifier
                display_and_save_confusion_matrices(mask_results, output_dir=plots_dir, mask_name=run_identifier)



# ---------------------------------------------------------------------
    # CONSOLIDATED SUMMARY ACROSS ALL MASKS
    # ---------------------------------------------------------------------
    if all_mask_summary_rows:
        df_overall = pd.DataFrame(all_mask_summary_rows)

        # -----------------------------------------------------------------
        # Compute % Δ for Binary F1, AUROC, and PR-AUC relative to 'enabled_baseline'
        # -----------------------------------------------------------------
        baseline_df = df_overall[
            df_overall["Feature Mask"] == "enabled_baseline"
        ]

        if not baseline_df.empty:
            base_f1_map = baseline_df.set_index("Model")["Binary F1"].to_dict()
            base_auroc_map = baseline_df.set_index("Model")["AUROC"].to_dict()
            base_prauc_map = baseline_df.set_index("Model")["PR-AUC"].to_dict()

            def calc_pct_diff(row, col_name, base_map):
                model = row["Model"]
                curr_val = row[col_name]
                base_val = base_map.get(model)
                if base_val is None or base_val == 0:
                    return 0.00
                return round(((curr_val - base_val) / base_val) * 100.0, 2)

            df_overall["% Δ Binary F1"] = df_overall.apply(
                lambda r: calc_pct_diff(r, "Binary F1", base_f1_map), axis=1
            )
            df_overall["% Δ AUROC"] = df_overall.apply(
                lambda r: calc_pct_diff(r, "AUROC", base_auroc_map), axis=1
            )
            df_overall["% Δ PR-AUC"] = df_overall.apply(
                lambda r: calc_pct_diff(r, "PR-AUC", base_prauc_map), axis=1
            )
        else:
            df_overall["% Δ Binary F1"] = 0.00
            df_overall["% Δ AUROC"] = 0.00
            df_overall["% Δ PR-AUC"] = 0.00

        # Sort cleanly by Feature Mask and PR-AUC
        df_overall = df_overall.sort_values(
            by=["Feature Mask", "PR-AUC"], ascending=[True, False]
        ).reset_index(drop=True)

        logger.info(
            "\n================ FINAL MULTI-MASK BENCHMARK SUMMARY"
            " ================\n%s",
            df_overall.round(4).to_string(index=False),
        )

        csv_save_path = os.path.join(
            plots_dir, "benchmark_summary_all_masks.csv"
        )
        df_overall.to_csv(csv_save_path, index=False)
        logger.info("Saved consolidated summary CSV to: %s", csv_save_path)
    else:
        logger.warning("No checkpoints were successfully evaluated.")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('▶ %(message)s'))
    logger.addHandler(console_handler)

    env = LoadEnvironment()
    if getattr(env, "ENVIRONMENT", "local_dev") == "local_dev":
        logger.info("Running in Google Colab environment. Logs will be saved to /mnt/fast_storage/projects/aviation_insight/logs/diagnostics")
        output_dir = Path("/mnt/fast_storage/projects/aviation_insight/logs/diagnostics")
    else:
        logger.info(f"Running in {env.ENVIRONMENT} environment. Logs will be saved to {env.OUTPUT_DIR}/diagnostics")
        output_dir = env.OUTPUT_DIR / "diagnostics"

    file_handler = DiagnosticFileHandler(issue_token=ISSUE_TOKEN, output_dir=output_dir if 'output_dir' in locals() else Path("/mnt/fast_storage/projects/aviation_insight/logs/diagnostics"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    main()