import sys
import os
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from .palette_manager import theme
matplotlib.use("Agg")  # Non-interactive backend to avoid GUI pauses & fix blank saves

# Project root injection to ensure tools package imports resolve cleanly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.diagnostic_logger import DiagnosticFileHandler

logger = logging.getLogger("AviationInsight")

def compute_nominal_baseline(dataset) -> dict[str, np.ndarray]:
    """
    Computes average nominal sensor profiles across all nominal samples in dataset (label == 0).
    Returns a dict mapping feature_name -> 1D array of shape [T].
    """
    feature_names = dataset.get_feature_names()
    nominal_windows = []

    # Sample nominal windows (up to 200 samples for fast computation)
    for idx in range(len(dataset)):
        label = dataset.labels[idx]
        if label == 0:
            window, _, _ = dataset[idx]
            if hasattr(window, "detach"):
                window = window.detach().cpu().numpy()
            nominal_windows.append(window)
            if len(nominal_windows) >= 200:
                break

    if len(nominal_windows) == 0:
        return {}

    # Stack: [N, T, C] -> Mean across N -> [T, C]
    nominal_stack = np.stack(nominal_windows, axis=0)
    mean_nominal = np.mean(nominal_stack, axis=0)  # [T, C]

    baseline_dict = {}
    for ch_idx, col_name in enumerate(feature_names):
        baseline_dict[col_name] = mean_nominal[:, ch_idx]

    return baseline_dict


def build_flight_results_from_dataset(
    top_attributed_results: list[dict],
    dataset,  # AviationFourClassDataset
    channel_names: list[str] | dict,
    top_k_per_method: int = 3,
    nominal_baseline: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """
    Extracts time-series windows from AviationFourClassDataset and appends
    nominal reference baselines for direct visual debugging.
    """
    if isinstance(channel_names, dict):
        idx_to_name = channel_names
    else:
        idx_to_name = {i: name for i, name in enumerate(channel_names)}

    flight_results = []

    for item in top_attributed_results:
        global_idx = item["global_idx"]

        # Extract record from dataset: (normalized_window, label, metadata)
        window_tensor, class_label, meta = dataset[global_idx]

        if hasattr(window_tensor, "detach"):
            X = window_tensor.detach().cpu().numpy()
        elif hasattr(window_tensor, "cpu"):
            X = window_tensor.cpu().numpy()
        else:
            X = np.asarray(window_tensor)

        num_channels = len(idx_to_name)
        if X.ndim == 2 and X.shape[0] == num_channels and X.shape[1] != num_channels:
            X = X.T

        num_timesteps, C = X.shape

        # Build feature DataFrame
        data_dict = {}
        for ch_idx in range(C):
            col_name = idx_to_name.get(ch_idx, f"channel_{ch_idx}")
            data_dict[col_name] = X[:, ch_idx]

            # Append nominal baseline column if available
            if nominal_baseline and col_name in nominal_baseline:
                data_dict[f"{col_name}_nominal_ref"] = nominal_baseline[col_name]

        df_flight = pd.DataFrame(data_dict)

        # Ground truth anomaly mask (0: Nominal, 1: Anomaly)
        gt_binary = 1 if class_label != 0 else 0
        df_flight["gt_anomaly"] = np.full(num_timesteps, gt_binary, dtype=int)

        # 1. Store the constant Window Score across all rows (explicitly named)
        win_score = float(item.get("window_score", item.get("score", 0.0)))
        raw_tau = item.get("tau_threshold", np.nan)
        tau_val = float(raw_tau) if raw_tau is not None else np.nan
        df_flight["window_score"] = win_score
        df_flight["score"] = win_score  # Retain 'score' for backward compatibility
        df_flight["tau_threshold"] = tau_val

        # Attach 1D per-timestep error array (160 timesteps)
        if "timestep_error" in item and item["timestep_error"] is not None:
            err_seq = np.asarray(item["timestep_error"])
            
            # Handle both [N, T] and 1D [T] formats safely
            if err_seq.ndim == 2:
                err_seq = err_seq[global_idx]  # If the full 2D array was passed
                
            df_flight["timestep_error"] = err_seq  # Shape: (160,)
        else:
            df_flight["timestep_error"] = item["score"]

        grad_sensors = [
            s.strip()
            for s in item.get("latent_gradient_top_sensors", "").split(",")
            if s.strip()
        ][:top_k_per_method]

        nn_sensors = [
            s.strip()
            for s in item.get("nearest_nominal_top_sensors", "").split(",")
            if s.strip()
        ][:top_k_per_method]

        flight_results.append({
            "flight_id": item.get("flight_record", f"flight_{global_idx}"),
            "global_idx": global_idx,
            "class_label": class_label,
            "class_name": meta.get("class_name", "Unknown") if isinstance(meta, dict) else "Unknown",
            "data": df_flight,
            "grad_sensors": grad_sensors,
            "nn_sensors": nn_sensors,
        })

    return flight_results

def plot_top_flights_dual_attribution(
    flight_results: list[dict],
    channel_names_dict: dict | None = None,
    title_suffix: str = "Top Anomalies",
) -> plt.Figure:
    """
    Renders multi-panel time-series attribution plots off-screen.
    Plots selected sensor signals vs. the mean nominal baseline (dashed grey).
    """
    plt.ioff()  # Turn off interactive plotting
    num_flights = len(flight_results)

    if num_flights == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No Flights Selected", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(num_flights, 1, figsize=(14, 3.8 * num_flights), sharex=False)
    if num_flights == 1:
        axes = [axes]

    for ax, f_info in zip(axes, flight_results):
        df = f_info["data"]
        flight_id = f_info.get("flight_id", "Unknown Flight")
        class_name = f_info.get("class_name", "Unknown")
        gt_anom = (df["gt_anomaly"].sum() > 0) if "gt_anomaly" in df.columns else False

        grad_sensors = f_info.get("grad_sensors", [])
        nn_sensors = f_info.get("nn_sensors", [])

        # Reconstruct attribution source mapping
        sources = {}
        using_nn = False  # not using nearest nominal from memory bank (only implemented for ts2vec_knn model)
        if nn_sensors and using_nn:
            for s in nn_sensors:
                sources[s] = "Nearest Nominal"

        # 2. Process Latent Gradient sensors (overwrites/combines cleanly)
        for s in grad_sensors:
            if s in sources:
                sources[s] = "NN + LG"  # Or keep as "Latent Gradient" if preferred
            else:
                sources[s] = "Latent Gradient"


        # Unique active channels
        attributed_sensors = list(sources.keys())
        if len(attributed_sensors) == 0:
            # Fallback to first 3 columns if no attributions exist (e.g. for nominals)
            attributed_sensors = [c for c in df.columns if not c.endswith("_nominal_ref") and c not in ["gt_anomaly", "score"]][:3]

        for idx, sensor in enumerate(attributed_sensors):
            sensor_color = theme.get_sensor_color(idx)
            if sensor in df.columns:
                src_label = sources.get(sensor, "Selected Sensor")
                
                # Plot Actual Flight Sensor Window
                ax.plot(
                    df.index,
                    df[sensor],
                    label=f"{sensor} ({src_label})",
                    color=sensor_color,
                    linewidth=2.0,
                )

                # Plot Nominal Reference Baseline (Dashed)
                ref_col = f"{sensor}_nominal_ref"
                if ref_col in df.columns:
                    ax.plot(
                        df.index,
                        df[ref_col],
                        linestyle="--",
                        color=sensor_color,
                        alpha=0.7,
                        linewidth=1.2,
                        label=f"{sensor} (Nominal Mean)",
                    )

        # Plot Anomaly Score on Secondary Axis
        status_tag = f"Ground Truth: {class_name}" if gt_anom else "Ground Truth: Nominal"
        win_score = df["window_score"].iloc[0] if "window_score" in df.columns else df["score"].iloc[0]
        peak_e_t = df["timestep_error"].max() if "timestep_error" in df.columns else win_score
        tau_val = df["tau_threshold"].iloc[0] if "tau_threshold" in df.columns else np.nan

        tau_str = f"{tau_val:.4f}" if not np.isnan(tau_val) else "N/A"
        ax.set_title(
            f"Flight: {flight_id} | {status_tag} | Window Score: {win_score:.4f} (Tau: {tau_str}) | Peak E(t): {peak_e_t:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        # Plot Per-Timestep Error Profile E(t) on Secondary Axis
        if "timestep_error" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(
                df.index,
                df["timestep_error"],
                color=theme.CRIMSON,
                linestyle=":",
                linewidth=1.8,
                alpha=0.85,
                label="Reconstruction Error E(t)",
            )
            # Shading under the error curve highlights exact temporal spikes
            ax2.fill_between(
                df.index,
                0,
                df["timestep_error"],
                color=theme.CRIMSON,
                alpha=0.1,
            )
            ax2.set_ylabel("Timestep Error E(t)", color=theme.CRIMSON)
            ax2.tick_params(axis="y", labelcolor=theme.CRIMSON)

        ax.set_xlabel("Time Step / Window Frame (160 steps)")
        ax.set_ylabel("Z-Score Normalized")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"Diagnostic Diagnostic Plot - {title_suffix}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig

# def format_results_for_visualization(res, metadata_df, feature_names, top_k=3, top_n=5):
#     """
#     Converts the output dictionary from evaluate_unsupervised_model into 
#     the top_attributed_results format needed for full sequence construction.
#     """
#     top_indices = res["top_indices"][:top_n]
#     y_scores = res["y_scores"]
#     y_true = res["y_true"]
#     sensor_attributions = res["sensor_attributions"]
    
#     top_attributed_results = []

#     for rank_idx, idx in enumerate(top_indices):
#         score = float(y_scores[idx])
#         gt_anom = int(y_true[idx])
#         grad_vector = sensor_attributions[idx]

#         # Extract flight record identifier
#         if metadata_df is not None and idx < len(metadata_df):
#             row = metadata_df.iloc[idx]
#             flight_rec = str(row.get("flight_record", f"flight_{idx}"))
#         else:
#             flight_rec = f"flight_{idx}"

#         # Get top feature names based on attribution scores
#         if feature_names is not None:
#             top_attr_indices = np.argsort(grad_vector)[::-1][:top_k]
#             if isinstance(feature_names, dict):
#                 top_sensors = [feature_names[i] for i in top_attr_indices if i in feature_names]
#             else:
#                 top_sensors = [feature_names[i] for i in top_attr_indices if i < len(feature_names)]
#             top_sensors_str = ", ".join(top_sensors)
#         else:
#             top_attr_indices = np.argsort(grad_vector)[::-1][:top_k]
#             top_sensors_str = ", ".join([f"feature_{i}" for i in top_attr_indices])

#         top_attributed_results.append({
#             "global_idx": int(idx),
#             "flight_record": flight_rec,
#             "score": score,
#             "gt_anomaly": gt_anom,
#             "latent_gradient_top_sensors": top_sensors_str,
#             "nearest_nominal_top_sensors": top_sensors_str
#         })

#     return top_attributed_results


def print_top_anomalies_summary(model_name, flight_results):
    """Prints an executive tabular summary of top anomalies for a model."""
    table_data = []
    
    for rank, f_info in enumerate(flight_results, start=1):
        df = f_info['data']
        has_gt = (df['gt_anomaly'].sum() > 0) if 'gt_anomaly' in df.columns else False
        status = "True Positive (GT Active)" if has_gt else "Potential False Positive"
        
        table_data.append({
            "Rank": rank,
            "Flight ID": f_info['flight_id'],
            "Classification": status,
            "Max Score": f"{df['score'].max():.4f}" if 'score' in df.columns else "N/A",
            "Top Latent Sensors": ", ".join(f_info['grad_sensors']),
        })
        
    summary_df = pd.DataFrame(table_data)
    logger.info(f"\n--- TOP ANOMALY REPORT [{model_name}] ---\n{summary_df.to_string(index=False)}\n")
    return summary_df


def visualize_and_save_top_anomalies(
    model_name: str,
    res: dict,
    test_ds,  # AviationFourClassDataset
    metadata_df: pd.DataFrame,
    feature_names: list[str] | dict,
    output_dir: str,
    run_identifier: str,
    top_k: int = 3,
    top_n: int = 5,
    num_nominals_to_plot: int = 3,
):
    """
    Extracts top anomaly samples and nominal control samples, builds full sequence DataFrames 
    with reference baselines, logs summary tables, and saves high-resolution diagnostic PNGs.
    """
    try:
        y_scores = np.asarray(res.get("y_scores", []))
        y_true = np.asarray(res.get("y_true", []))
        top_indices = res.get("top_indices", [])
        sensor_attributions = res.get("sensor_attributions", [])
        y_errors = res.get("y_errors", None)
        tau_threshold = res.get("Calibrated Tau", res.get("Calibrated_Tau", np.nan))

        if len(top_indices) == 0 or len(sensor_attributions) == 0:
            logger.warning("No valid top indices or sensor attributions found for %s. Skipping plot.", model_name)
            return

        # Compute Mean Nominal Baseline across test set
        nominal_baseline = compute_nominal_baseline(test_ds)

        # -------------------------------------------------------------
        # 1. PROCESS TOP ANOMALIES (HIGHEST SCORES)
        # -------------------------------------------------------------
        top_anom_indices = top_indices[:top_n]
        top_anom_attributed = []

        for idx in top_anom_indices:
            window_score = float(y_scores[idx])
            gt_anom = int(y_true[idx])
            grad_vector = sensor_attributions[idx]

            # Extract the 160-timestep error vector for sample `idx`
            timestep_err = y_errors[idx] if y_errors is not None and idx < len(y_errors) else None

            flight_rec = str(metadata_df.iloc[idx].get("flight_record", f"flight_{idx}")) if metadata_df is not None and idx < len(metadata_df) else f"flight_{idx}"

            # Extract top-k sensor names
            top_attr_indices = np.argsort(grad_vector)[::-1][:top_k]
            if isinstance(feature_names, dict):
                top_sensors = [feature_names[i] for i in top_attr_indices if i in feature_names]
            elif isinstance(feature_names, (list, tuple, np.ndarray)):
                top_sensors = [feature_names[i] for i in top_attr_indices if i < len(feature_names)]
            else:
                top_sensors = [f"feature_{i}" for i in top_attr_indices]

            sensors_str = ", ".join(top_sensors)
            top_anom_attributed.append({
                "global_idx": int(idx),
                "flight_record": flight_rec,
                "score": window_score,
                "window_score": window_score,
                "tau_threshold": tau_threshold,
                "gt_anomaly": gt_anom,
                "timestep_error": timestep_err,  # anomaly errors(t) for this window
                "latent_gradient_top_sensors": sensors_str,
                "nearest_nominal_top_sensors": sensors_str,
            })

        flight_anom_results = build_flight_results_from_dataset(
            top_attributed_results=top_anom_attributed,
            dataset=test_ds,
            channel_names=feature_names,
            top_k_per_method=top_k,
            nominal_baseline=nominal_baseline,
        )

        # -------------------------------------------------------------
        # 2. PROCESS NOMINAL CONTROL SAMPLES (LOWEST SCORES / GT NOMINAL)
        # -------------------------------------------------------------
        nominal_indices = np.where(y_true == 0)[0]
        if len(nominal_indices) > 0:
            # Sort nominal indices by ascending score (truest nominals)
            lowest_score_nominals = nominal_indices[np.argsort(y_scores[nominal_indices])[:num_nominals_to_plot]]
        else:
            lowest_score_nominals = []

        top_nom_attributed = []
        for idx in lowest_score_nominals:
            window_score = float(y_scores[idx])
            
            grad_vector = sensor_attributions[idx] if idx < len(sensor_attributions) else np.zeros(len(feature_names))

            # Extract the 160-timestep error vector for sample `idx`
            timestep_err = y_errors[idx] if y_errors is not None and idx < len(y_errors) else None

            flight_rec = str(metadata_df.iloc[idx].get("flight_record", f"flight_{idx}")) if metadata_df is not None and idx < len(metadata_df) else f"flight_{idx}"
            top_attr_indices = np.argsort(grad_vector)[::-1][:top_k]
            top_sensors = [feature_names[i] for i in top_attr_indices if i < len(feature_names)]
            sensors_str = ", ".join(top_sensors)

            top_nom_attributed.append({
                "global_idx": int(idx),
                "flight_record": flight_rec,
                "score": window_score,
                "window_score": window_score,                
                "tau_threshold": tau_threshold,
                "gt_anomaly": 0,
                "timestep_error": timestep_err,  # anomaly errors(t) for this window
                "latent_gradient_top_sensors": sensors_str,
                "nearest_nominal_top_sensors": sensors_str,
            })

        flight_nom_results = build_flight_results_from_dataset(
            top_attributed_results=top_nom_attributed,
            dataset=test_ds,
            channel_names=feature_names,
            top_k_per_method=top_k,
            nominal_baseline=nominal_baseline,
        )

        # -------------------------------------------------------------
        # 3. PRINT DIAGNOSTIC SUMMARY TABLES
        # -------------------------------------------------------------
        anom_rows = []
        for rank, f_info in enumerate(flight_anom_results, start=1):
            df = f_info["data"]
            has_gt = f_info["class_label"] != 0
            status = f"True Positive ({f_info['class_name']})" if has_gt else "Potential False Positive"

            anom_rows.append({
                "Rank": rank,
                "Flight ID": f_info["flight_id"],
                "Classification": status,
                "Max Score": f"{df['score'].max():.4f}",
                "Top Attributed Sensors": ", ".join(f_info["grad_sensors"]),
            })

        summary_anom_df = pd.DataFrame(anom_rows)
        logger.info(
            "\n--- TOP ANOMALY REPORT [%s | %s] ---\n%s\n",
            model_name,
            run_identifier,
            summary_anom_df.to_string(index=False),
        )

        # -------------------------------------------------------------
        # 4. RENDER & SAVE DISK PLOTS
        # -------------------------------------------------------------
        clean_model_tag = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

        # Plot 1: Top Anomalies
        fig_anom = plot_top_flights_dual_attribution(
            flight_results=flight_anom_results,
            channel_names_dict=feature_names if isinstance(feature_names, dict) else None,
            title_suffix=f"{model_name} - Top Anomalies",
        )
        save_path_anom = os.path.join(output_dir, f"top_anomalies_{clean_model_tag}_{run_identifier}.png")
        fig_anom.savefig(save_path_anom, dpi=300, bbox_inches="tight")
        plt.close(fig_anom)
        logger.info("Saved top anomalies plot for [%s] to: %s", model_name, save_path_anom)

        # Plot 2: Nominal Baseline Comparison Controls
        if len(flight_nom_results) > 0:
            fig_nom = plot_top_flights_dual_attribution(
                flight_results=flight_nom_results,
                channel_names_dict=feature_names if isinstance(feature_names, dict) else None,
                title_suffix=f"{model_name} - Nominal Controls",
            )
            save_path_nom = os.path.join(output_dir, f"nominal_controls_{clean_model_tag}_{run_identifier}.png")
            fig_nom.savefig(save_path_nom, dpi=300, bbox_inches="tight")
            plt.close(fig_nom)
            logger.info("Saved nominal controls plot for [%s] to: %s", model_name, save_path_nom)

    except Exception as e:
        logger.warning("Failed to generate anomaly attribution plots for %s: %s", model_name, e, exc_info=True)