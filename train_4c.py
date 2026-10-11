import sys
import argparse
import json
import numpy as np
import logging
import time
import gc
import yaml
import random
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, parameter_count
from typing import Dict, Any, Optional
from pathlib import Path


from data_loader.aviation_four_class_dataset import AviationFourClassDataset
from evaluation.anomaly_four_class_evaluator import AnomalyFourClassEvaluator
from src.utils.load_environment import LoadEnvironment
from src.utils.early_stopping import EarlyStopping

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import Omni model and loss module
from models.omni_anomaly import OmniAnomaly

# Import TS2Vec model and loss module
from models.ts2vec import TS2Vec
from models.ts2v_models.losses import hierarchical_contrastive_loss

# Import PatchTST model and loss module
from models.patch_tst import PatchTST

# Import Anomaly Transformer model
from models.anomaly_transformer import AnomalyTransformer

# Import LSTM-VAE model and loss function
from models.lstm_vae import LSTMVAE, lstm_vae_loss

# Import conventional ConvAutoencoder model for comparison
from models.conventional_autoencoder import ConvAutoencoder

from tools.diagnostic_logger import DiagnosticFileHandler

logger = logging.getLogger("AviationInsight")
ISSUE_TOKEN = "TRAIN_MODELS"

import gc
import torch

def cleanup_pipeline_resources(*targets, **named_targets):
    """
    Cleans up PyTorch models, data loaders, evaluators, and intermediate variables
    from GPU/CPU memory across pipeline runs.
    
    Usage:
        cleanup_pipeline_resources(model, optimizer, train_loader, val_loader)
        # Or pass named references:
        cleanup_pipeline_resources(at_model=at_model, evaluator=evaluator)
    """
    # Clear explicit positional args
    for obj in targets:
        del obj

    # Clear named kwargs
    for key in list(named_targets.keys()):
        del named_targets[key]

    # Force Python garbage collection
    gc.collect()

    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def seed_everything(seed: int = 42):
    """Enforces strict determinism across Python, NumPy, PyTorch CPU & CUDA operations."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force CUDA deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 1.8+ / 2.0+ deterministic flag (warns if a non-deterministic op is called)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

def seed_worker(worker_id):
    """Seeds individual PyTorch DataLoader worker processes deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def measure_model_complexity(model: torch.nn.Module, sample_input: torch.Tensor) -> dict:
    """Computes total trainable parameters (Millions) and FLOPs (GFLOPs) without mutating model state dict."""
    # 1. Deepcopy model to avoid in-place parameter mutations from thop
    model_eval = copy.deepcopy(model).eval()
    device = next(model.parameters()).device

    # 2. Re-flatten RNN parameters to avoid PyTorch cuDNN memory layout warning
    for m in model_eval.modules():
        if isinstance(m, (nn.RNNBase, nn.RNN, nn.LSTM, nn.GRU)):
            m.flatten_parameters()

    if isinstance(sample_input, (list, tuple)):
        sample_input = tuple(x.to(device) for x in sample_input)
    else:
        sample_input = sample_input.to(device)

    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = params_count / 1e6
    gflops = 0.0

    try:
        from thop import profile
        with torch.no_grad():
            inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)
            flops, _ = profile(model_eval, inputs=inputs, verbose=False)
            gflops = flops / 1e9
    except Exception:
        gflops = 0.0

    # Explicitly clear deepcopied model references
    del model_eval

    return {
        "params_m": round(float(params_m), 4),
        "gflops": round(float(gflops), 6)
    }

def measure_inference_latency(
    model: torch.nn.Module, 
    sample_input: torch.Tensor, 
    num_warmup: int = 10, 
    num_runs: int = 50
) -> float:
    """
    Measures clean inference latency per flight window (in milliseconds).
    """
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    with torch.no_grad():
        # Warmup passes for GPU clock/kernel stabilization
        for _ in range(num_warmup):
            _ = model(sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed execution
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    return round(float(avg_latency_ms), 3)

# Handle np.float32 and np.float64 serialization in JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
        return super(NpEncoder, self).default(obj)


def save_checkpoint_and_metrics(
    save_dir: Path,
    filename: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    best_val_pr_auc: float,
    train_loss: float,
    val_metrics: Dict[str, Any],
    complexity_info: Dict[str, float],
    inference_latency_ms: float,
    epoch_duration_sec: float,
    throughput: float,
    peak_vram_gb: float,
    run_identifier: str,
    mask_name: str,
    logo_mask: Optional[str] = None,
    encoder_cls: type = NpEncoder,
) -> Path:
    """
    Saves a comprehensive PyTorch checkpoint binary (.pt) and an accompanying 
    lightweight JSON metrics summary (.json) for evaluation and run resuming.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / filename

    # 1. Build PyTorch State & Metadata Payload
    checkpoint = {
        # Training State
        "epoch": epoch,
        "run_identifier": str(run_identifier),
        "ckpt_path": str(checkpoint_path.resolve()),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_pr_auc": float(best_val_pr_auc),
        
        # RNG States for Deterministic Resume
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },

        # Run Configuration
        "config": {
            "model_type": str(getattr(model, "model_type", model.__class__.__name__)),
            "feature_mask": str(mask_name),
            "logo_mask": str(logo_mask or "none"),
            "run_identifier": str(run_identifier),
            "ckpt_path": str(checkpoint_path.resolve()),
        },

        # ML Quality & Separation Metrics
        "metrics": {
            "train_loss": train_loss,
            "val_pr_auc": val_metrics["PR-AUC"],
            "val_auroc": val_metrics["AUROC"],
            "val_f1_binary": val_metrics["F1_Binary"],
            "val_f1_macro": val_metrics["F1_Macro"],
            "calibrated_tau": val_metrics.get("Calibrated_Tau", 0.0),
            "precision": val_metrics.get("Prec", 0.0),
            "recall": val_metrics.get("Rec", 0.0),
            "fpr": val_metrics.get("FPR", 0.0),
            "fnr": val_metrics.get("FNR", 0.0),
            "tp_count": val_metrics.get("TP", 0),
            "fp_count": val_metrics.get("FP", 0),
            "tn_count": val_metrics.get("TN", 0),
            "fn_count": val_metrics.get("FN", 0),
        },

        # Computational Efficiency Metrics
        "computational_metrics": {
            "params_m": complexity_info["params_m"],
            "forward_gflops": complexity_info["gflops"],
            "inference_latency_ms": inference_latency_ms,
            "epoch_duration_sec": epoch_duration_sec,
            "throughput_samples_per_sec": throughput,
            "peak_vram_gb": peak_vram_gb,
        }
    }

    # 2. Save PyTorch Checkpoint Binary
    torch.save(checkpoint, checkpoint_path)

    # 3. Save Lightweight Summary JSON for evaluate.py / Polars aggregation
    json_path = save_dir / f"{checkpoint_path.stem}_metrics.json"
    summary_data = {
        **checkpoint["config"],
        "best_epoch": epoch,
        **checkpoint["metrics"],
        **checkpoint["computational_metrics"]
    }

    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2, cls=encoder_cls)

    return checkpoint_path


def load_yaml_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Computes symmetric KL divergence helper for Association Discrepancy."""
    p_norm = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-5)
    q_norm = q / (torch.sum(q, dim=-1, keepdim=True) + 1e-5)
    return torch.sum(p_norm * (torch.log(p_norm + 1e-5) - torch.log(q_norm + 1e-5)), dim=-1).mean()

def train_conv_ae_epoch(model, dataloader, optimizer, criterion, scaler, device) -> float:
    """Executes single training epoch for ConvAutoencoder using PyTorch AMP."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        data = batch[0].to(device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                recon = model(data)
                loss = criterion(recon, data)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def run_conv_ae_pipeline(exp_config_path: str):
    env = LoadEnvironment()
    exp_cfg = load_yaml_config(exp_config_path)
    data_fractions = exp_cfg.get("data_fractions", [1.0]) # Default to full dataset if omitted

    if not isinstance(data_fractions, list):
        data_fractions = [data_fractions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info(f"⚡ Starting ConvAutoencoder Training Run on Device: {device}")

    # 1. Extract Seed and Enforce Determinism Immediately
    seed = exp_cfg.get("defaults", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Global seed set to: {seed} (Deterministic mode enabled)")

    # 2. Setup PyTorch DataLoader Generator
    g = torch.Generator()
    g.manual_seed(seed)

    checkpoint_base_dir = env.CHECKPOINT_DIR
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = exp_cfg["models"]["conv_ae"]
    epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 30))
    lr = float(model_cfg.get("learning_rate", exp_cfg["defaults"].get("learning_rate", 1e-3)))
    # 1. Initialize Early Stopping before epoch loop
    patience = exp_cfg.get("defaults", {}).get("patience", 10)
    patience_delta = exp_cfg.get("defaults", {}).get("patience_delta", 0.0005)
    early_stopper = EarlyStopping(patience=patience, delta=patience_delta, mode="max")

    # -------------------------------------------------------------------------
    # FEATURE MASK & TRAIN-TIME LOGO SWEEP LOOP
    # -------------------------------------------------------------------------
    feature_masks = exp_cfg.get("feature_masks", ["enabled_baseline"])
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)

    # Normalize single string or None into an iterable list
    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]

    for mask_name in feature_masks:
        for logo_mask in train_logo_masks:
            for fraction in data_fractions:
                # Build descriptive directory name: e.g., omni_enabled_baseline_logo_acc
                logo_tag = f"_logo_{logo_mask}" if logo_mask else ""
                frac_tag = f"_frac_{int(fraction * 100)}" if fraction < 1.0 else ""
                run_identifier = f"{mask_name}{logo_tag}{frac_tag}"

                logger.info(
                    f"\n================ MASK: {mask_name} | TRAIN LOGO: {logo_mask or 'NONE'} ================"
                )

                train_loader = val_loader = model = optimizer = evaluator = None  # Initialize to None for cleanup in finally block
                try:
                    # 1. Datasets & Loaders
                    train_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="train",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )
                    val_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="val",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )



                    nominal_train_subset = train_ds.get_nominal_subset()
                    train_loader = DataLoader(
                        nominal_train_subset,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=True,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=False,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    input_shape = (train_ds.window_size, train_ds.num_channels)

                    logger.info(
                        f"🚀 Training ConvAutoencoder on [{mask_name}] | Input Shape: {input_shape} | "
                        f"Epochs: {epochs} | LR: {lr}"
                    )

                    # 2. Instantiate ConvAutoencoder Model
                    model = ConvAutoencoder(input_shape=input_shape).to(device)

                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=lr,
                        weight_decay=float(exp_cfg["defaults"].get("weight_decay", 1e-5)),
                    )
                    criterion = nn.MSELoss()
                    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

                    evaluator = AnomalyFourClassEvaluator(
                        model=model,
                        model_type="conv_ae",
                        device=device.type,
                    )

                    # 2. Checkpoint Save Directory with LOGO Tag
                    experiment_tag = exp_cfg.get("experiment_tag", "standard_training")
                    save_dir = checkpoint_base_dir / experiment_tag / f"{evaluator.model_type}_{run_identifier}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Saving checkpoints to: {save_dir}")

                    # =====================================================================
                    # Measure Static Model Complexity (Before Epoch Loop)
                    # =====================================================================
                    # Take a single sample window from train_loader for shape probing
                    sample_batch = next(iter(train_loader))
                    sample_x = sample_batch[0][:1]  # Shape: (1, seq_len, num_features)

                    complexity_info = measure_model_complexity(model, sample_x)
                    inference_latency_ms = measure_inference_latency(model, sample_x)

                    # Reset peak GPU memory stats before training begins
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    best_pr_auc = 0.0
                    # Metrics history container
                    metrics_history = []

                    for epoch in range(1, epochs + 1):
                        # Synchronize and start timer for epoch throughput calculation
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        epoch_start_time = time.perf_counter()
                        start_time = time.time()
                        avg_train_loss = train_conv_ae_epoch(model, train_loader, optimizer, criterion, scaler, device)
                        elapsed = time.time() - start_time

                        # Fast evaluation during training loop (no autograd, no cuDNN errors)
                        val_metrics = evaluator.get_metrics(val_loader, compute_gradients=False)
                        current_pr_auc = val_metrics["PR-AUC"]

                        if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        epoch_duration_sec = time.perf_counter() - epoch_start_time

                        # Calculate throughput (samples / sec)
                        total_samples = len(train_loader.dataset)
                        throughput = total_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0

                        # Capture Peak VRAM usage
                        peak_vram_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3) 
                            if torch.cuda.is_available() else 0.0
                        )

                        logger.info(
                            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6f} | "
                            f"Val AUROC: {val_metrics['AUROC']:.4f} | Val PR-AUC: {current_pr_auc:.4f} | "
                            f"Val F1_Macro: {val_metrics['F1_Macro']:.4f} | Val F1_Binary: {val_metrics['F1_Binary']:.4f} | "
                            f"Time: {epoch_duration_sec:.2f}s | Throughput: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_gb:.2f} GB"
                        )

                        # 2. Update Early Stopper and Check Best Score
                        is_early_stop = early_stopper(current_pr_auc)
                            
                        if early_stopper.counter == 0:
                            
                            save_checkpoint_and_metrics(
                                    save_dir=save_dir,
                                    filename="best_model.pt",
                                    epoch=epoch,
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=None,  # Pass None if no scheduler
                                    best_val_pr_auc=early_stopper.best_metric_val,
                                    train_loss=avg_train_loss,
                                    val_metrics=val_metrics,
                                    complexity_info=complexity_info,
                                    inference_latency_ms=inference_latency_ms,
                                    epoch_duration_sec=epoch_duration_sec,
                                    throughput=throughput,
                                    peak_vram_gb=peak_vram_gb,
                                    run_identifier=run_identifier,
                                    mask_name=mask_name,
                                    logo_mask=logo_mask,
                            )
                            logger.info(f"  ⭐️ New best ConvAutoencoder PR-AUC saved: {early_stopper.best_metric_val:.4f}")

                        metrics_history.append({
                                "epoch": epoch,
                                "model_type": evaluator.model_type,
                                "feature_mask": mask_name,
                                "logo_mask": logo_mask or "none",
                                "run_identifier": run_identifier,
                                # Core Summary
                                "train_loss": avg_train_loss,
                                "val_pr_auc": val_metrics.get("PR-AUC", 0.0),
                                "val_auroc": val_metrics.get("AUROC", 0.0),
                                "val_f1_binary": val_metrics.get("F1_Binary", 0.0),
                                "val_f1_macro": val_metrics.get("F1_Macro", 0.0),
                                # Operational & Convergence Metrics
                                "calibrated_tau": val_metrics.get("Calibrated_Tau", 0.0),
                                "precision": val_metrics.get("Prec", 0.0),
                                "recall": val_metrics.get("Rec", 0.0),
                                "fpr": val_metrics.get("FPR", 0.0),
                                "fnr": val_metrics.get("FNR", 0.0),
                                # Count Metrics for Convergence Tracking
                                "tp_count": val_metrics.get("TP", 0),
                                "fp_count": val_metrics.get("FP", 0),
                                "tn_count": val_metrics.get("TN", 0),
                                "fn_count": val_metrics.get("FN", 0),
                                "total_detected_anomalies": val_metrics.get("TP", 0) + val_metrics.get("FP", 0),
                                # Computational Metrics
                                "params_m": complexity_info["params_m"],
                                "forward_gflops": complexity_info["gflops"],
                                "inference_latency_ms": inference_latency_ms,
                                "epoch_duration_sec": epoch_duration_sec,
                                "throughput_samples_per_sec": throughput,
                                "peak_vram_gb": peak_vram_gb,
                            })
                        # 3. Check Early Stopping Condition to Break Loop
                        if is_early_stop:
                            logger.info(f"\n✋ Early stopping triggered at Epoch {epoch:02d}. Advancing to next model...")
                            break

                    # Save structured history alongside best model checkpoint
                    history_path = save_dir / "training_history.json"
                    with open(history_path, "w") as f:
                        json.dump(metrics_history, f, indent=2, cls=NpEncoder)

                except Exception as e:
                    logger.error(f"[ERROR] Failed at Mask: {mask_name}, Group: {run_identifier} -> {e}")
                    # Decide whether to raise or continue to the next fold
                    continue

                finally:
                    # Pass whichever variables exist in the scope of the current pipeline
                    cleanup_pipeline_resources(
                        model, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        evaluator
                    )
                    # Re-assign variables to None if they remain in caller scope
                    model = optimizer = train_loader = val_loader = evaluator = None


def train_lstm_vae_epoch(model, dataloader, optimizer, beta, scaler, device) -> float:
    """Executes single training epoch for LSTM-VAE using PyTorch AMP."""
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        data = batch[0].to(device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                recon, mu, logvar = model(data)
                loss = lstm_vae_loss(recon, data, mu, logvar, beta=beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, mu, logvar = model(data)
            loss = lstm_vae_loss(recon, data, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def run_lstm_vae_pipeline(exp_config_path: str):
    env = LoadEnvironment()
    exp_cfg = load_yaml_config(exp_config_path)
    data_fractions = exp_cfg.get("data_fractions", [1.0]) # Default to full dataset if omitted

    if not isinstance(data_fractions, list):
        data_fractions = [data_fractions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info(f"⚡ Starting LSTM-VAE Training Run on Device: {device}")

    # 1. Extract Seed and Enforce Determinism Immediately
    seed = exp_cfg.get("defaults", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Global seed set to: {seed} (Deterministic mode enabled)")

    # 2. Setup PyTorch DataLoader Generator
    g = torch.Generator()
    g.manual_seed(seed)

    checkpoint_base_dir = env.CHECKPOINT_DIR
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = exp_cfg["models"]["lstm_vae"]
    beta = float(model_cfg.get("beta", 1.0))

    # 1. Initialize Early Stopping before epoch loop
    patience = exp_cfg.get("defaults", {}).get("patience", 10)
    patience_delta = exp_cfg.get("defaults", {}).get("patience_delta", 0.0005)
    early_stopper = EarlyStopping(patience=patience, delta=patience_delta, mode="max")
    epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 30))

    # -------------------------------------------------------------------------
    # FEATURE MASK & TRAIN-TIME LOGO SWEEP LOOP
    # -------------------------------------------------------------------------
    feature_masks = exp_cfg.get("feature_masks", ["enabled_baseline"])
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)

    # Normalize single string or None into an iterable list
    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]

    for mask_name in feature_masks:
        for logo_mask in train_logo_masks:
            for fraction in data_fractions:
                # Build descriptive directory name: e.g., omni_enabled_baseline_logo_acc
                logo_tag = f"_logo_{logo_mask}" if logo_mask else ""
                frac_tag = f"_frac_{int(fraction * 100)}" if fraction < 1.0 else ""
                run_identifier = f"{mask_name}{logo_tag}{frac_tag}"


                logger.info(
                    f"\n================ MASK: {mask_name} | TRAIN LOGO: {logo_mask or 'NONE'} ================"
                )

                model = optimizer = train_loader = val_loader = evaluator = None
                try: 
                    # 1. Datasets & Loaders
                    train_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="train",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )
                    val_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="val",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )

                    nominal_train_subset = train_ds.get_nominal_subset()
                    train_loader = DataLoader(
                        nominal_train_subset,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=True,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=False,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    logger.info(
                        f"🚀 Training LSTM-VAE on [{mask_name}] | Window: {train_ds.window_size} | "
                        f"Channels: {train_ds.num_channels} | Epochs: {epochs}"
                    )

                    # 2. Instantiate LSTM-VAE Model
                    model = LSTMVAE(
                        input_dim=train_ds.num_channels,
                        hidden_dim=model_cfg.get("hidden_dim", 128),
                        latent_dim=model_cfg.get("latent_dim", 32),
                        num_layers=model_cfg.get("num_layers", 2),
                        dropout=model_cfg.get("dropout", 0.1),
                    ).to(device)

                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(exp_cfg["defaults"].get("learning_rate", 1e-4)),
                        weight_decay=float(exp_cfg["defaults"].get("weight_decay", 1e-5)),
                    )
                    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

                    evaluator = AnomalyFourClassEvaluator(
                        model=model,
                        model_type="lstm_vae",
                        device=device.type,
                    )

                    # 2. Checkpoint Save Directory with LOGO Tag
                    experiment_tag = exp_cfg.get("experiment_tag", "standard_training")
                    save_dir = checkpoint_base_dir / experiment_tag / f"{evaluator.model_type}_{run_identifier}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Saving checkpoints to: {save_dir}")

                    # =====================================================================
                    # Measure Static Model Complexity (Before Epoch Loop)
                    # =====================================================================
                    # Take a single sample window from train_loader for shape probing
                    sample_batch = next(iter(train_loader))
                    sample_x = sample_batch[0][:1]  # Shape: (1, seq_len, num_features)

                    complexity_info = measure_model_complexity(model, sample_x)
                    inference_latency_ms = measure_inference_latency(model, sample_x)

                    # Reset peak GPU memory stats before training begins
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Metrics history container
                    metrics_history = []

                    for epoch in range(1, epochs + 1):
                        # Synchronize and start timer for epoch throughput calculation
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        epoch_start_time = time.perf_counter()

                        start_time = time.time()
                        avg_train_loss = train_lstm_vae_epoch(model, train_loader, optimizer, beta, scaler, device)
                        elapsed = time.time() - start_time

                        # Fast evaluation during training loop (no autograd, no cuDNN errors)
                        val_metrics = evaluator.get_metrics(val_loader, compute_gradients=False)
                        current_pr_auc = val_metrics["PR-AUC"]

                        if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        epoch_duration_sec = time.perf_counter() - epoch_start_time

                        # Calculate throughput (samples / sec)
                        total_samples = len(train_loader.dataset)
                        throughput = total_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0

                        # Capture Peak VRAM usage
                        peak_vram_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3) 
                            if torch.cuda.is_available() else 0.0
                        )

                        logger.info(
                            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6f} | "
                            f"Val AUROC: {val_metrics['AUROC']:.4f} | Val PR-AUC: {current_pr_auc:.4f} | "
                            f"Val F1_Macro: {val_metrics['F1_Macro']:.4f} | Val F1_Binary: {val_metrics['F1_Binary']:.4f} | "
                            f"Time: {epoch_duration_sec:.2f}s | Throughput: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_gb:.2f} GB"
                        )

                        # 2. Update Early Stopper and Check Best Score
                        is_early_stop = early_stopper(current_pr_auc)
                        
                        if early_stopper.counter == 0:  # New best score found (counter resets to 0)
                            save_checkpoint_and_metrics(
                                    save_dir=save_dir,
                                    filename="best_model.pt",
                                    epoch=epoch,
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=None,  # Pass None if no scheduler
                                    best_val_pr_auc=early_stopper.best_metric_val,
                                    train_loss=avg_train_loss,
                                    val_metrics=val_metrics,
                                    complexity_info=complexity_info,
                                    inference_latency_ms=inference_latency_ms,
                                    epoch_duration_sec=epoch_duration_sec,
                                    throughput=throughput,
                                    peak_vram_gb=peak_vram_gb,
                                    run_identifier=run_identifier,
                                    mask_name=mask_name,
                                    logo_mask=logo_mask,
                            )
                            logger.info(f"  ⭐️ New best LSTM-VAE PR-AUC saved: {early_stopper.best_metric_val:.4f}")

                        metrics_history.append({
                            "epoch": epoch,
                            "model_type": evaluator.model_type,
                            "feature_mask": mask_name,
                            "logo_mask": logo_mask or "none",
                            "run_identifier": run_identifier,
                            # Core Summary
                            "train_loss": avg_train_loss,
                            "val_pr_auc": val_metrics.get("PR-AUC", 0.0),
                            "val_auroc": val_metrics.get("AUROC", 0.0),
                            "val_f1_binary": val_metrics.get("F1_Binary", 0.0),
                            "val_f1_macro": val_metrics.get("F1_Macro", 0.0),
                            # Operational & Convergence Metrics
                            "calibrated_tau": val_metrics.get("Calibrated_Tau", 0.0),
                            "precision": val_metrics.get("Prec", 0.0),
                            "recall": val_metrics.get("Rec", 0.0),
                            "fpr": val_metrics.get("FPR", 0.0),
                            "fnr": val_metrics.get("FNR", 0.0),
                            # Count Metrics for Convergence Tracking
                            "tp_count": val_metrics.get("TP", 0),
                            "fp_count": val_metrics.get("FP", 0),
                            "tn_count": val_metrics.get("TN", 0),
                            "fn_count": val_metrics.get("FN", 0),
                            "total_detected_anomalies": val_metrics.get("TP", 0) + val_metrics.get("FP", 0),
                            # Computational Metrics
                            "params_m": complexity_info["params_m"],
                            "forward_gflops": complexity_info["gflops"],
                            "inference_latency_ms": inference_latency_ms,
                            "epoch_duration_sec": epoch_duration_sec,
                            "throughput_samples_per_sec": throughput,
                            "peak_vram_gb": peak_vram_gb,
                        })


                        # 3. Check Early Stopping Condition to Break Loop
                        if is_early_stop:
                            logger.info(f"\n✋ Early stopping triggered at Epoch {epoch:02d}. Advancing to next model...")
                            break

                    # Save structured history alongside best model checkpoint
                    history_path = save_dir / "training_history.json"
                    with open(history_path, "w") as f:
                        json.dump(metrics_history, f, indent=2, cls=NpEncoder)

                except Exception as e:
                        logger.error(f"[ERROR] Pipeline run failed: {e}")
                        raise e  # Re-raise or handle as needed for sweep tracking

                finally:
                    # Executes unconditionally whether the run succeeds or crashes
                    cleanup_pipeline_resources(
                        model, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        evaluator
                    )
                    model = optimizer = train_loader = val_loader = evaluator = None


def train_at_epoch(model, dataloader, optimizer, criterion, k_lambda, device) -> float:
    model.train()
    total_rec_loss = 0.0

    for batch in dataloader:
        data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        # ----------------------------------------------------
        # Phase 1: Maximize Association Discrepancy
        # ----------------------------------------------------
        optimizer.zero_grad()
        enc_out, series, prior, _ = model(data)
        rec_loss = criterion(enc_out, data)

        p_loss = 0.0
        for s, p in zip(series, prior):
            p_loss += kl_loss(p, s.detach())
        p_loss = p_loss / len(series)

        loss1 = rec_loss - k_lambda * p_loss
        loss1.backward(retain_graph=True)
        optimizer.step()

        # FREE PHASE 1 GRAPH REFERENCES
        del enc_out, series, prior, rec_loss, p_loss, loss1

        # ----------------------------------------------------
        # Phase 2: Minimize Association Discrepancy & Recon Error
        # ----------------------------------------------------
        optimizer.zero_grad()
        enc_out, series, prior, _ = model(data)
        rec_loss = criterion(enc_out, data)

        s_loss = 0.0
        for s, p in zip(series, prior):
            s_loss += kl_loss(s, p.detach())
        s_loss = s_loss / len(series)

        loss2 = rec_loss + k_lambda * s_loss
        loss2.backward()
        optimizer.step()

        total_rec_loss += rec_loss.item()

        # FREE PHASE 2 GRAPH REFERENCES
        del enc_out, series, prior, rec_loss, s_loss, loss2

    return total_rec_loss / len(dataloader)


def run_at_pipeline(exp_config_path: str):
    env = LoadEnvironment()
    exp_cfg = load_yaml_config(exp_config_path)
    data_fractions = exp_cfg.get("data_fractions", [1.0]) # Default to full dataset if omitted

    if not isinstance(data_fractions, list):
        data_fractions = [data_fractions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info(f"⚡ Starting Anomaly Transformer Training Run on Device: {device}")

    # 1. Extract Seed and Enforce Determinism Immediately
    seed = exp_cfg.get("defaults", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Global seed set to: {seed} (Deterministic mode enabled)")

    # 2. Setup PyTorch DataLoader Generator
    g = torch.Generator()
    g.manual_seed(seed)

    checkpoint_base_dir = env.CHECKPOINT_DIR
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = exp_cfg["models"]["anomaly_transformer"]
    k_lambda = float(model_cfg.get("k_lambda", 3.0))

    # -------------------------------------------------------------------------
    # FEATURE MASK & TRAIN-TIME LOGO SWEEP LOOP
    # -------------------------------------------------------------------------
    feature_masks = exp_cfg.get("feature_masks", ["enabled_baseline"])
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)

    # Normalize single string or None into an iterable list
    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]

    for mask_name in feature_masks:
        for logo_mask in train_logo_masks:
            for fraction in data_fractions:
                # Build descriptive directory name: e.g., omni_enabled_baseline_logo_acc
                logo_tag = f"_logo_{logo_mask}" if logo_mask else ""
                frac_tag = f"_frac_{int(fraction * 100)}" if fraction < 1.0 else ""
                run_identifier = f"{mask_name}{logo_tag}{frac_tag}"

                logger.info(
                    f"\n================ MASK: {mask_name} | TRAIN LOGO: {logo_mask or 'NONE'} ================"
                )

                at_model = optimizer = train_loader = val_loader = evaluator = None
                try:

                    # 1. Datasets & Loaders
                    train_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="train",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )
                    val_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="val",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )

                    nominal_train_subset = train_ds.get_nominal_subset()
                    train_loader = DataLoader(
                        nominal_train_subset,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=True,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=False,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    # 2. Instantiate Anomaly Transformer Model
                    at_model = AnomalyTransformer(
                        win_size=train_ds.window_size,
                        enc_in=train_ds.num_channels,
                        c_out=train_ds.num_channels,
                        d_model=model_cfg.get("d_model", 128),
                        n_heads=model_cfg.get("n_heads", 8),
                        e_layers=model_cfg.get("e_layers", 3),
                        d_ff=model_cfg.get("d_ff", 256),
                        dropout=model_cfg.get("dropout", 0.1),
                        output_attention=True,
                    ).to(device)

                    optimizer = torch.optim.AdamW(
                        at_model.parameters(),
                        lr=float(exp_cfg["defaults"].get("learning_rate", 1e-4)),
                        weight_decay=float(exp_cfg["defaults"].get("weight_decay", 1e-5)),
                    )
                    criterion = nn.MSELoss()

                    # Instantiate AnomalyFourClassEvaluator for validation monitoring
                    evaluator = AnomalyFourClassEvaluator(
                        model=at_model,
                        model_type="anomaly_transformer",
                        device=device.type,
                    )

                    # 2. Checkpoint Save Directory with LOGO Tag
                    experiment_tag = exp_cfg.get("experiment_tag", "standard_training")
                    save_dir = checkpoint_base_dir / experiment_tag / f"{evaluator.model_type}_{run_identifier}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Saving checkpoints to: {save_dir}")

                    # 1. Initialize Early Stopping before epoch loop
                    patience = exp_cfg.get("defaults", {}).get("patience", 10)
                    patience_delta = exp_cfg.get("defaults", {}).get("patience_delta", 0.0005)
                    early_stopper = EarlyStopping(patience=patience, delta=patience_delta, mode="max")
                    # Fallback hierarchy: model-specific epoch -> global default epoch -> hardcoded fallback
                    epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 30))

                    # =====================================================================
                    # Measure Static Model Complexity (Before Epoch Loop)
                    # =====================================================================
                    # Take a single sample window from train_loader for shape probing
                    sample_batch = next(iter(train_loader))
                    sample_x = sample_batch[0][:1]  # Shape: (1, seq_len, num_features)

                    complexity_info = measure_model_complexity(at_model, sample_x)
                    inference_latency_ms = measure_inference_latency(at_model, sample_x)

                    # Reset peak GPU memory stats before training begins
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Metrics history container
                    metrics_history = []

                    for epoch in range(1, epochs + 1):
                        # Synchronize and start timer for epoch throughput calculation
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        epoch_start_time = time.perf_counter()

                        start_time = time.time()
                        avg_train_loss = train_at_epoch(at_model, train_loader, optimizer, criterion, k_lambda, device)
                        elapsed = time.time() - start_time

                        # Validation evaluation
                        val_metrics = evaluator.get_metrics(val_loader, compute_gradients=False)
                        current_pr_auc = val_metrics["PR-AUC"]

                        if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        epoch_duration_sec = time.perf_counter() - epoch_start_time

                        # Calculate throughput (samples / sec)
                        total_samples = len(train_loader.dataset)
                        throughput = total_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0

                        # Capture Peak VRAM usage
                        peak_vram_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3) 
                            if torch.cuda.is_available() else 0.0
                        )

                        logger.info(
                            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6f} | "
                            f"Val AUROC: {val_metrics['AUROC']:.4f} | Val PR-AUC: {current_pr_auc:.4f} | "
                            f"Val F1_Macro: {val_metrics['F1_Macro']:.4f} | Val F1_Binary: {val_metrics['F1_Binary']:.4f} | "
                            f"Time: {epoch_duration_sec:.2f}s | Throughput: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_gb:.2f} GB"
                        )

                        # 2. Update Early Stopper and Check Best Score
                        is_early_stop = early_stopper(current_pr_auc)
                        
                        if early_stopper.counter == 0:
                            best_pr_auc = current_pr_auc
                            save_checkpoint_and_metrics(
                                    save_dir=save_dir,
                                    filename="best_model.pt",
                                    epoch=epoch,
                                    model=at_model,
                                    optimizer=optimizer,
                                    scheduler=None,  # Pass None if no scheduler
                                    best_val_pr_auc=best_pr_auc,
                                    train_loss=avg_train_loss,
                                    val_metrics=val_metrics,
                                    complexity_info=complexity_info,
                                    inference_latency_ms=inference_latency_ms,
                                    epoch_duration_sec=epoch_duration_sec,
                                    throughput=throughput,
                                    peak_vram_gb=peak_vram_gb,
                                    run_identifier=run_identifier,
                                    mask_name=mask_name,
                                    logo_mask=logo_mask,
                            )
                            logger.info(f"  ⭐️ New best Anomaly Transformer PR-AUC saved: {best_pr_auc:.4f}")

                        metrics_history.append({
                                "epoch": epoch,
                                "model_type": evaluator.model_type,
                                "feature_mask": mask_name,
                                "logo_mask": logo_mask or "none",
                                "run_identifier": run_identifier,
                                # Core Summary
                                "train_loss": avg_train_loss,
                                "val_pr_auc": val_metrics["PR-AUC"],
                                "val_auroc": val_metrics["AUROC"],
                                "val_f1_binary": val_metrics["F1_Binary"],
                                "val_f1_macro": val_metrics["F1_Macro"],
                                # Operational & Convergence Metrics
                                "calibrated_tau": val_metrics.get("Calibrated_Tau", None),
                                "precision": val_metrics["Prec"],
                                "recall": val_metrics["Rec"],
                                "fpr": val_metrics["FPR"],
                                "fnr": val_metrics.get("FNR", 0.0),
                                # Count Metrics for Convergence Tracking
                                "tp_count": val_metrics["TP"],
                                "fp_count": val_metrics["FP"],
                                "total_detected_anomalies": val_metrics["TP"] + val_metrics["FP"],
                            })
                        # 3. Check Early Stopping Condition to Break Loop
                        if is_early_stop:
                            logger.info(f"\n✋ Early stopping triggered at Epoch {epoch:02d}. Advancing to next model...")
                            break

                    # Save structured history alongside best model checkpoint
                    history_path = save_dir / "training_history.json"
                    with open(history_path, "w") as f:
                        json.dump(metrics_history, f, indent=2, cls=NpEncoder)

                except Exception as e:
                        logger.error(f"[ERROR] Pipeline run failed: {e}")
                        raise e  # Re-raise or handle as needed for sweep tracking

                finally:
                
                    # Executes unconditionally whether the run succeeds or crashes
                    cleanup_pipeline_resources(
                        at_model, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        evaluator
                    )
                    at_model = optimizer = train_loader = val_loader = evaluator = None

def train_patchtst_epoch(model, dataloader, optimizer, criterion, scaler, patch_len, stride, device) -> float:
    """Executes single training epoch with unfold patching and PyTorch AMP."""
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        optimizer.zero_grad()

        # Permute & unfold into [bs, num_patch, num_features, patch_len]
        x_raw = data.permute(0, 2, 1)  # [bs, num_features, window_size]
        x_patched = x_raw.unfold(dimension=-1, size=patch_len, step=stride)
        x_in = x_patched.permute(0, 2, 1, 3).contiguous()  # [bs, num_patch, num_features, patch_len]

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                recon = model(x_in)
                loss = criterion(recon, x_in)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon = model(x_in)
            loss = criterion(recon, x_in)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def run_patchtst_pipeline(exp_config_path: str):
    env = LoadEnvironment()
    exp_cfg = load_yaml_config(exp_config_path)
    data_fractions = exp_cfg.get("data_fractions", [1.0]) # Default to full dataset if omitted

    if not isinstance(data_fractions, list):
        data_fractions = [data_fractions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info(f"⚡ Starting PatchTST Training Run on Device: {device}")

    # 1. Extract Seed and Enforce Determinism Immediately
    seed = exp_cfg.get("defaults", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Global seed set to: {seed} (Deterministic mode enabled)")

    # 2. Setup PyTorch DataLoader Generator
    g = torch.Generator()
    g.manual_seed(seed)                       

    checkpoint_base_dir = env.CHECKPOINT_DIR
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = exp_cfg["models"]["patchtst"]
    patch_len = model_cfg.get("patch_len", 16)
    stride = model_cfg.get("stride", 8)

    # -------------------------------------------------------------------------
    # FEATURE MASK & TRAIN-TIME LOGO SWEEP LOOP
    # -------------------------------------------------------------------------
    feature_masks = exp_cfg.get("feature_masks", ["enabled_baseline"])
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)

    # Normalize single string or None into an iterable list
    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]

    for mask_name in feature_masks:
        for logo_mask in train_logo_masks:
            for fraction in data_fractions:
                # Build descriptive directory name: e.g., omni_enabled_baseline_logo_acc
                logo_tag = f"_logo_{logo_mask}" if logo_mask else ""
                frac_tag = f"_frac_{int(fraction * 100)}" if fraction < 1.0 else ""
                run_identifier = f"{mask_name}{logo_tag}{frac_tag}"

                logger.info(
                    f"\n================ MASK: {mask_name} | TRAIN LOGO: {logo_mask or 'NONE'} ================"
                )

                patch_model = optimizer = train_loader = val_loader = evaluator = None
                try: 

                    # 1. Datasets & Loaders
                    train_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="train",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )
                    val_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="val",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )

                    nominal_train_subset = train_ds.get_nominal_subset()
                    train_loader = DataLoader(
                        nominal_train_subset,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=True,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=False,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    # 2. Dynamic Patch Count Calculation
                    window_size = train_ds.window_size
                    num_patch = (window_size - patch_len) // stride + 1

                    logger.info(
                        f"🚀 Training PatchTST on [{mask_name}] | Window: {window_size} | "
                        f"Channels: {train_ds.num_channels} | Patches: {num_patch}"
                    )

                    # 3. Instantiate PatchTST
                    patch_model = PatchTST(
                        c_in=train_ds.num_channels,
                        target_dim=window_size,
                        patch_len=patch_len,
                        stride=stride,
                        num_patch=num_patch,
                        head_type="pretrain",
                        d_model=model_cfg.get("d_model", 128),
                        n_layers=model_cfg.get("n_layers", 3),
                        n_heads=model_cfg.get("n_heads", 16),
                        dropout=model_cfg.get("dropout", 0.1),
                        attn_dropout=model_cfg.get("attn_dropout", 0.1),
                        head_dropout=model_cfg.get("head_dropout", 0.2),
                    ).to(device)

                    optimizer = torch.optim.AdamW(
                        patch_model.parameters(),
                        lr=float(exp_cfg["defaults"].get("learning_rate", 1e-4)),
                        weight_decay=float(exp_cfg["defaults"].get("weight_decay", 1e-5)),
                    )
                    criterion = nn.MSELoss()
                    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

                    # Instantiate AnomalyFourClassEvaluator for validation monitoring
                    evaluator = AnomalyFourClassEvaluator(
                        model=patch_model,
                        model_type="patchtst",
                        device=device.type,
                        patch_len=patch_len,
                        stride=stride,
                    )

                    # 2. Checkpoint Save Directory with LOGO Tag
                    experiment_tag = exp_cfg.get("experiment_tag", "standard_training")
                    save_dir = checkpoint_base_dir / experiment_tag / f"{evaluator.model_type}_{run_identifier}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Saving checkpoints to: {save_dir}")

                    # 1. Initialize Early Stopping before epoch loop
                    patience = exp_cfg.get("defaults", {}).get("patience", 10)
                    patience_delta = exp_cfg.get("defaults", {}).get("patience_delta", 0.0005)
                    early_stopper = EarlyStopping(patience=patience, delta=patience_delta, mode="max")
                    epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 100))

                    # =====================================================================
                    # Measure Static Model Complexity (Before Epoch Loop)
                    # =====================================================================
                    # Take a single sample window from train_loader for shape probing
                    sample_batch = next(iter(train_loader))
                    sample_x = sample_batch[0][:1]  # Shape: (1, seq_len, num_features)

                    # Unfold 3D sample_x to 4D if running PatchTST [bs x num_patch x n_vars x patch_len] for pretrain
                    if evaluator.model_type == "patchtst":
                        x_raw = sample_x.permute(0, 2, 1)  # [bs, num_features, window_size]
                        sample_input = x_raw.unfold(dimension=-1, size=patch_len, step=stride).permute(0, 2, 1, 3).contiguous()
                    else:
                        sample_input = sample_x

                    complexity_info = measure_model_complexity(patch_model, sample_input)
                    inference_latency_ms = measure_inference_latency(patch_model, sample_input)

                    # Reset peak GPU memory stats before training begins
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Metrics history container
                    metrics_history = []
                    for epoch in range(1, epochs + 1):
                        # Synchronize and start timer for epoch throughput calculation
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        epoch_start_time = time.perf_counter()

                        start_time = time.time()
                        avg_train_loss = train_patchtst_epoch(
                            patch_model, train_loader, optimizer, criterion, scaler, patch_len, stride, device
                        )
                        elapsed = time.time() - start_time

                        # Validation metrics computed via evaluator
                        val_metrics = evaluator.get_metrics(val_loader, compute_gradients=False)
                        current_pr_auc = val_metrics["PR-AUC"]

                        if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        epoch_duration_sec = time.perf_counter() - epoch_start_time

                        # Calculate throughput (samples / sec)
                        total_samples = len(train_loader.dataset)
                        throughput = total_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0

                        # Capture Peak VRAM usage
                        peak_vram_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3) 
                            if torch.cuda.is_available() else 0.0
                        )

                        logger.info(
                            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6f} | "
                            f"Val AUROC: {val_metrics['AUROC']:.4f} | Val PR-AUC: {current_pr_auc:.4f} | "
                            f"Val F1_Macro: {val_metrics['F1_Macro']:.4f} | Val F1_Binary: {val_metrics['F1_Binary']:.4f} | "
                            f"Time: {epoch_duration_sec:.2f}s | Throughput: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_gb:.2f} GB"
                        )

                        # 2. Update Early Stopper and Check Best Score
                        is_early_stop = early_stopper(current_pr_auc)

                        # Checkpoint on best Val PR-AUC
                        if early_stopper.counter == 0:
                            save_checkpoint_and_metrics(
                                    save_dir=save_dir,
                                    filename="best_model.pt",
                                    epoch=epoch,
                                    model=patch_model,
                                    optimizer=optimizer,
                                    scheduler=None,  # Pass None if no scheduler
                                    best_val_pr_auc=early_stopper.best_metric_val,
                                    train_loss=avg_train_loss,
                                    val_metrics=val_metrics,
                                    complexity_info=complexity_info,
                                    inference_latency_ms=inference_latency_ms,
                                    epoch_duration_sec=epoch_duration_sec,
                                    throughput=throughput,
                                    peak_vram_gb=peak_vram_gb,
                                    run_identifier=run_identifier,
                                    mask_name=mask_name,
                                    logo_mask=logo_mask,
                            )
                            logger.info(f"  ⭐️ New best PatchTST PR-AUC saved: {early_stopper.best_metric_val:.4f}")

                        metrics_history.append({
                            "epoch": epoch,
                            "model_type": evaluator.model_type,
                            "feature_mask": mask_name,
                            "logo_mask": logo_mask or "none",
                            "run_identifier": run_identifier,
                            # Core Summary
                            "train_loss": avg_train_loss,
                            "val_pr_auc": val_metrics.get("PR-AUC", 0.0),
                            "val_auroc": val_metrics.get("AUROC", 0.0),
                            "val_f1_binary": val_metrics.get("F1_Binary", 0.0),
                            "val_f1_macro": val_metrics.get("F1_Macro", 0.0),
                            # Operational & Convergence Metrics
                            "calibrated_tau": val_metrics.get("Calibrated_Tau", 0.0),
                            "precision": val_metrics.get("Prec", 0.0),
                            "recall": val_metrics.get("Rec", 0.0),
                            "fpr": val_metrics.get("FPR", 0.0),
                            "fnr": val_metrics.get("FNR", 0.0),
                            # Count Metrics for Convergence Tracking
                            "tp_count": val_metrics.get("TP", 0),
                            "fp_count": val_metrics.get("FP", 0),
                            "tn_count": val_metrics.get("TN", 0),
                            "fn_count": val_metrics.get("FN", 0),
                            "total_detected_anomalies": val_metrics.get("TP", 0) + val_metrics.get("FP", 0),
                            # Computational Metrics
                            "params_m": complexity_info["params_m"],
                            "forward_gflops": complexity_info["gflops"],
                            "inference_latency_ms": inference_latency_ms,
                            "epoch_duration_sec": epoch_duration_sec,
                            "throughput_samples_per_sec": throughput,
                            "peak_vram_gb": peak_vram_gb,
                        })

                        # 3. Check Early Stopping Condition to Break Loop
                        if is_early_stop:
                            logger.info(f"\n✋ Early stopping triggered at Epoch {epoch:02d}. Advancing to next model...")
                            break

                    # Save structured history alongside best model checkpoint
                    history_path = save_dir / "training_history.json"
                    with open(history_path, "w") as f:
                        json.dump(metrics_history, f, indent=2, cls=NpEncoder)

                except Exception as e:
                        logger.error(f"[ERROR] Pipeline run failed: {e}")
                        raise e  # Re-raise or handle as needed for sweep tracking

                finally:
                    # Executes unconditionally whether the run succeeds or crashes
                    cleanup_pipeline_resources(
                        patch_model, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        evaluator
                    )

def train_ts2vec_epoch(model_net, dataloader, optimizer, scaler, device) -> float:
    """Executes a single TS2Vec hierarchical contrastive training epoch."""
    model_net.train()
    running_loss = 0.0

    for batch in dataloader:
        x_batch = batch[0].to(device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # Sample random temporal/instance masks
            mask1 = torch.bernoulli(torch.full((x_batch.size(0), x_batch.size(1)), 0.5, device=device)).bool()
            mask2 = torch.bernoulli(torch.full((x_batch.size(0), x_batch.size(1)), 0.5, device=device)).bool()

            out1 = model_net(x_batch, mask=mask1)
            out2 = model_net(x_batch, mask=mask2)

            loss = hierarchical_contrastive_loss(out1, out2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def run_ts2vec_pipeline(exp_config_path: str):
    env = LoadEnvironment()
    exp_cfg = load_yaml_config(exp_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info(f"⚡ Starting TS2Vec Training Run on Device: {device}")

    # 1. Extract Seed and Enforce Determinism Immediately
    seed = exp_cfg.get("defaults", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Global seed set to: {seed} (Deterministic mode enabled)")

    # 2. Setup PyTorch DataLoader Generator
    g = torch.Generator()
    g.manual_seed(seed)


    checkpoint_base_dir = env.CHECKPOINT_DIR
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = exp_cfg["models"]["ts2vec"]

    # -------------------------------------------------------------------------
    # FEATURE MASK & TRAIN-TIME LOGO SWEEP LOOP
    # -------------------------------------------------------------------------
    feature_masks = exp_cfg.get("feature_masks", ["enabled_baseline"])
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)
    data_fractions = exp_cfg.get("data_fractions", [1.0]) # Default to full dataset if omitted

    # Normalize single string or None into an iterable list
    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]
    if not isinstance(data_fractions, list):
        data_fractions = [data_fractions]


    for mask_name in feature_masks:
        for logo_mask in train_logo_masks:
            for fraction in data_fractions:
                # Build descriptive directory name: e.g., omni_enabled_baseline_logo_acc
                logo_tag = f"_logo_{logo_mask}" if logo_mask else ""
                frac_tag = f"_frac_{int(fraction * 100)}" if fraction < 1.0 else ""
                run_identifier = f"{mask_name}{logo_tag}{frac_tag}"

                logger.info(
                    f"\n================ MASK: {mask_name} | TRAIN LOGO: {logo_mask or 'NONE'} ================"
                )

                ts2vec_wrapper = optimizer = train_loader = val_loader = ts2vec_knn_evaluator = None

                try:
                    # 1. Datasets & Loaders
                    train_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="train",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )
                    val_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="val",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )

                    # 2. Checkpoint Save Directory with LOGO Tag
                    experiment_tag = exp_cfg.get("experiment_tag", "standard_training")
                    save_dir = checkpoint_base_dir / experiment_tag / f"ts2vec_{run_identifier}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Saving checkpoints to: {save_dir}")

                    nominal_train_subset = train_ds.get_nominal_subset()
                    train_loader = DataLoader(
                        nominal_train_subset,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=True,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=False,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    logger.info(f"🚀 Training TS2Vec on [{mask_name}] (Input Dim: {train_ds.num_channels})")

                    # 2. Instantiate TS2Vec Network
                    ts2vec_wrapper = TS2Vec(
                        input_dims=train_ds.num_channels,
                        output_dims=model_cfg.get("output_dims", 320),
                        hidden_dims=model_cfg.get("hidden_dims", 64),
                        depth=model_cfg.get("depth", 10),
                        device=device,
                    )
                    
                    # Extract underlying PyTorch network and compile
                    ts2vec_net = ts2vec_wrapper.net
                    if hasattr(torch, "compile"):
                        ts2vec_net = torch.compile(ts2vec_net)
                    ts2vec_net.to(device)

                    optimizer = torch.optim.AdamW(
                        ts2vec_net.parameters(),
                        lr=float(exp_cfg["defaults"].get("learning_rate", 1e-4)),
                        weight_decay=float(exp_cfg["defaults"].get("weight_decay", 1e-5)),
                    )
                    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

                    epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 100))

                    # =====================================================================
                    # Measure Static Model Complexity (Before Epoch Loop)
                    # =====================================================================
                    # Take a single sample window from train_loader for shape probing
                    sample_batch = next(iter(train_loader))
                    sample_x = sample_batch[0][:1]  # Shape: (1, seq_len, num_features)

                    complexity_info = measure_model_complexity(ts2vec_net, sample_x)
                    inference_latency_ms = measure_inference_latency(ts2vec_net, sample_x)

                    # Reset peak GPU memory stats before training begins
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                        # 1. Initialize Early Stopping before epoch loop
                        patience = exp_cfg.get("defaults", {}).get("patience", 10)
                        patience_delta = exp_cfg.get("defaults", {}).get("patience_delta", 0.0005)
                        early_stopper = EarlyStopping(patience=patience, delta=patience_delta, mode="max")

                        epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 100))
                        metrics_history = []

                    for epoch in range(1, epochs + 1):
                        # Synchronize and start timer for epoch throughput calculation
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        epoch_start_time = time.perf_counter()

                        start_time = time.time()
                        avg_train_loss = train_ts2vec_epoch(ts2vec_net, train_loader, optimizer, scaler, device)
                        elapsed = time.time() - start_time

                        # 3. Fit memory bank and evaluate on validation set using AnomalyFourClassEvaluator
                        ts2vec_knn_evaluator = AnomalyFourClassEvaluator(
                            model=ts2vec_net,
                            model_type="ts2vec_knn",
                            device=device.type,
                            train_loader=train_loader,  # Extracts & caches nominal memory bank automatically
                            k=model_cfg.get("k", 5),
                            normalize_embeddings=model_cfg.get("normalize_embeddings", False),
                        )

                        val_metrics = ts2vec_knn_evaluator.get_metrics(val_loader, auto_percentile=95.0, compute_gradients=False)
                        current_pr_auc = val_metrics["PR-AUC"]

                        if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        epoch_duration_sec = time.perf_counter() - epoch_start_time

                        # Calculate throughput (samples / sec)
                        total_samples = len(train_loader.dataset)
                        throughput = total_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0

                        # Capture Peak VRAM usage
                        peak_vram_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3) 
                            if torch.cuda.is_available() else 0.0
                        )

                        logger.info(
                            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6f} | "
                            f"Val AUROC: {val_metrics['AUROC']:.4f} | Val PR-AUC: {current_pr_auc:.4f} | "
                            f"Val F1_Macro: {val_metrics['F1_Macro']:.4f} | Val F1_Binary: {val_metrics['F1_Binary']:.4f} | "
                            f"Time: {epoch_duration_sec:.2f}s | Throughput: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_gb:.2f} GB"
                        )

                        # 2. Update Early Stopper and Check Best Score
                        is_early_stop = early_stopper(current_pr_auc)

                        # Save best checkpoint based on Val PR-AUC
                        if early_stopper.counter == 0:

                            save_checkpoint_and_metrics(
                                    save_dir=save_dir,
                                    filename="best_model.pt",
                                    epoch=epoch,
                                    model=ts2vec_net,
                                    optimizer=optimizer,
                                    scheduler=None,  # Pass None if no scheduler
                                    best_val_pr_auc=early_stopper.best_metric_val,
                                    train_loss=avg_train_loss,
                                    val_metrics=val_metrics,
                                    complexity_info=complexity_info,
                                    inference_latency_ms=inference_latency_ms,
                                    epoch_duration_sec=epoch_duration_sec,
                                    throughput=throughput,
                                    peak_vram_gb=peak_vram_gb,
                                    run_identifier=run_identifier,
                                    mask_name=mask_name,
                                    logo_mask=logo_mask,
                            )
                            logger.info(f"  ⭐️ New best TS2Vec k-NN PR-AUC saved: {early_stopper.best_metric_val:.4f}")

                        # store metrics history for analysis and visualization
                        metrics_history.append({
                            "epoch": epoch,
                            "model_type": ts2vec_knn_evaluator.model_type,
                            "feature_mask": mask_name,
                            "logo_mask": logo_mask or "none",
                            "run_identifier": run_identifier,
                            # Core Summary
                            "train_loss": avg_train_loss,
                            "val_pr_auc": val_metrics.get("PR-AUC", 0.0),
                            "val_auroc": val_metrics.get("AUROC", 0.0),
                            "val_f1_binary": val_metrics.get("F1_Binary", 0.0),
                            "val_f1_macro": val_metrics.get("F1_Macro", 0.0),
                            # Operational & Convergence Metrics
                            "calibrated_tau": val_metrics.get("Calibrated_Tau", 0.0),
                            "precision": val_metrics.get("Prec", 0.0),
                            "recall": val_metrics.get("Rec", 0.0),
                            "fpr": val_metrics.get("FPR", 0.0),
                            "fnr": val_metrics.get("FNR", 0.0),
                            # Count Metrics for Convergence Tracking
                            "tp_count": val_metrics.get("TP", 0),
                            "fp_count": val_metrics.get("FP", 0),
                            "tn_count": val_metrics.get("TN", 0),
                            "fn_count": val_metrics.get("FN", 0),
                            "total_detected_anomalies": val_metrics.get("TP", 0) + val_metrics.get("FP", 0),
                            # Computational Metrics
                            "params_m": complexity_info["params_m"],
                            "forward_gflops": complexity_info["gflops"],
                            "inference_latency_ms": inference_latency_ms,
                            "epoch_duration_sec": epoch_duration_sec,
                            "throughput_samples_per_sec": throughput,
                            "peak_vram_gb": peak_vram_gb,
                        })

                        # 3. Check Early Stopping Condition to Break Loop
                        if is_early_stop:
                            logger.info(f"\n✋ Early stopping triggered at Epoch {epoch:02d}. Advancing to next model...")
                            break
            
                    # Save structured history alongside best model checkpoint
                    history_path = save_dir / "training_history.json"
                    with open(history_path, "w") as f:
                        json.dump(metrics_history, f, indent=2, cls=NpEncoder)

                except Exception as e:
                    logger.error(f"[ERROR] Failed at Mask: {mask_name}, Group: {run_identifier} -> {e}")
                    # Decide whether to raise or continue to the next fold
                    continue
                finally:
                    # 7. Reclaim GPU/CPU memory BEFORE the next sweep fold starts
                    cleanup_pipeline_resources(
                        ts2vec_wrapper, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        ts2vec_knn_evaluator
                    )
                    ts2vec_wrapper = optimizer = train_loader = val_loader = ts2vec_knn_evaluator = None

def build_omni_model(input_dim: int, seq_len: int, model_cfg: dict) -> nn.Module:
    """Instantiates OmniAnomaly matching notebook parameters."""
    from models.omni_anomaly import OmniAnomaly
    return OmniAnomaly(
        n_features=input_dim,
        window_size=seq_len,
        latent_dim=model_cfg.get("latent_dim", 8),
        rnn_hidden=model_cfg.get("rnn_hidden", 64),
    )


def train_omni_epoch(model, dataloader, optimizer, scaler, device) -> float:
    """Executes single training epoch with PyTorch AMP mixed precision."""
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        x_batch = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                recon_x, mu, logvar = model(x_batch)
                recons_loss = F.mse_loss(recon_x, x_batch)
                kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recons_loss + 0.01 * kld_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon_x, mu, logvar = model(x_batch)
            recons_loss = F.mse_loss(recon_x, x_batch)
            kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recons_loss + 0.01 * kld_loss
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def run_omni_pipeline(exp_config_path: str):
    env = LoadEnvironment()
    exp_cfg = load_yaml_config(exp_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    
    logger.info(f"⚡ Starting Training Run on Device: {device}")

    # 1. Extract Seed and Enforce Determinism Immediately
    seed = exp_cfg.get("defaults", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Global seed set to: {seed} (Deterministic mode enabled)")

    # 2. Setup PyTorch DataLoader Generator
    g = torch.Generator()
    g.manual_seed(seed)

    checkpoint_base_dir = env.CHECKPOINT_DIR
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # FEATURE MASK & TRAIN-TIME LOGO SWEEP LOOP
    # -------------------------------------------------------------------------
    feature_masks = exp_cfg.get("feature_masks", ["enabled_baseline"])
    train_logo_masks = exp_cfg.get("logo_masks", {}).get("train_time", None)
    data_fractions = exp_cfg.get("data_fractions", [1.0]) # Default to full dataset if omitted

    # Normalize single string or None into an iterable list
    if not isinstance(train_logo_masks, list):
        train_logo_masks = [train_logo_masks]
    if not isinstance(data_fractions, list):
        data_fractions = [data_fractions]

    for mask_name in feature_masks:
        for logo_mask in train_logo_masks:
            for fraction in data_fractions:
                # Build descriptive directory name: e.g., omni_enabled_baseline_logo_acc
                logo_tag = f"_logo_{logo_mask}" if logo_mask else ""
                frac_tag = f"_frac_{int(fraction * 100)}" if fraction < 1.0 else ""
                run_identifier = f"{mask_name}{logo_tag}{frac_tag}"

                logger.info(
                    f"\n================ MASK: {mask_name} | TRAIN LOGO: {logo_mask or 'NONE'} | DATA FRAC: {fraction} ================"
                )

                model = optimizer = train_loader = val_loader = evaluator = None

                try:
                    # 1. Datasets & Loaders
                    train_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="train",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )
                    val_ds = AviationFourClassDataset(
                        manifest_dir=env.DATABASE_DIR / "manifests",
                        database_dir=env.DATABASE_DIR,
                        stats_path=env.DATABASE_DIR / "metadata" / "train_split_stats.json",
                        split="val",
                        feature_set=mask_name,
                        logo_mask_col=logo_mask,
                    )

                    nominal_train_subset = train_ds.get_nominal_subset(fraction=fraction, seed=seed)
                    train_loader = DataLoader(
                        nominal_train_subset,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=True,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=exp_cfg["defaults"]["batch_size"],
                        shuffle=False,
                        num_workers=exp_cfg["defaults"].get("num_workers", 4),
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                    )

                    # 2. Extract OmniAnomaly Configuration
                    model_cfg = exp_cfg["models"]["omni"]
                    logger.info(f"\n🚀 Training OmniAnomaly on [{mask_name}] (Input Dim: {train_ds.num_channels})")

                    model = build_omni_model(
                        input_dim=train_ds.num_channels,
                        seq_len=train_ds.window_size,
                        model_cfg=model_cfg,
                    ).to(device)

                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(exp_cfg["defaults"].get("learning_rate", 1e-4)),
                        weight_decay=float(exp_cfg["defaults"].get("weight_decay", 1e-5)),
                    )
                    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

                    # Instantiate AnomalyFourClassEvaluator for validation tracking
                    evaluator = AnomalyFourClassEvaluator(model, model_type="omni", device=device.type)

                    # 2. Checkpoint Save Directory with LOGO Tag
                    experiment_tag = exp_cfg.get("experiment_tag", "standard_training")
                    save_dir = checkpoint_base_dir / experiment_tag / f"{evaluator.model_type}_{run_identifier}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Saving checkpoints to: {save_dir}")

                    best_pr_auc = 0.0
                    epochs = model_cfg.get("epochs", exp_cfg["defaults"].get("epochs", 100))
                    # Initialize Early Stopping before epoch loop (e.g., patience=8 for PR-AUC)
                    patience = exp_cfg.get("defaults", {}).get("patience", 10)
                    patience_delta = exp_cfg.get("defaults", {}).get("patience_delta", 0.0005)
                    early_stopper = EarlyStopping(patience=patience, delta=patience_delta, mode="max")

                    # =====================================================================
                    # Measure Static Model Complexity (Before Epoch Loop)
                    # =====================================================================
                    # Take a single sample window from train_loader for shape probing
                    sample_batch = next(iter(train_loader))
                    sample_x = sample_batch[0][:1]  # Shape: (1, seq_len, num_features)

                    complexity_info = measure_model_complexity(model, sample_x)
                    inference_latency_ms = measure_inference_latency(model, sample_x)

                    # Reset peak GPU memory stats before training begins
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Metrics history container
                    metrics_history = []

                    for epoch in range(1, epochs + 1):
                        # Synchronize and start timer for epoch throughput calculation
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        epoch_start_time = time.perf_counter()

                        start_time = time.time()
                        avg_train_loss = train_omni_epoch(model, train_loader, optimizer, scaler, device)
                        elapsed = time.time() - start_time

                        # Compute Validation Metrics using AnomalyFourClassEvaluator
                        val_metrics = evaluator.get_metrics(val_loader, compute_gradients=False)
                        current_pr_auc = val_metrics["PR-AUC"]

                        if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        epoch_duration_sec = time.perf_counter() - epoch_start_time

                        # Calculate throughput (samples / sec)
                        total_samples = len(train_loader.dataset)
                        throughput = total_samples / epoch_duration_sec if epoch_duration_sec > 0 else 0.0

                        # Capture Peak VRAM usage
                        peak_vram_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3) 
                            if torch.cuda.is_available() else 0.0
                        )

                        logger.info(
                            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.6f} | "
                            f"Val AUROC: {val_metrics['AUROC']:.4f} | Val PR-AUC: {current_pr_auc:.4f} | "
                            f"Val F1_Macro: {val_metrics['F1_Macro']:.4f} | Val F1_Binary: {val_metrics['F1_Binary']:.4f} | "
                            f"Time: {epoch_duration_sec:.2f}s | Throughput: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_gb:.2f} GB"
                        )

                        # Save best checkpoint based on Val PR-AUC
                        is_early_stop = early_stopper(current_pr_auc)  # update current Val PR-AUC
                        if early_stopper.counter == 0:   # best score

                            save_checkpoint_and_metrics(
                                    save_dir=save_dir,
                                    filename="best_model.pt",
                                    epoch=epoch,
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=None,  # Pass None if no scheduler
                                    best_val_pr_auc=early_stopper.best_metric_val,
                                    train_loss=avg_train_loss,
                                    val_metrics=val_metrics,
                                    complexity_info=complexity_info,
                                    inference_latency_ms=inference_latency_ms,
                                    epoch_duration_sec=epoch_duration_sec,
                                    throughput=throughput,
                                    peak_vram_gb=peak_vram_gb,
                                    run_identifier=run_identifier,
                                    mask_name=mask_name,
                                    logo_mask=logo_mask,
                            )
                            logger.info(f"  ⭐️ New best PR-AUC saved: {early_stopper.best_metric_val:.4f}")

                        metrics_history.append({
                            "epoch": epoch,
                            "model_type": evaluator.model_type,
                            "feature_mask": mask_name,
                            "logo_mask": logo_mask or "none",
                            "run_identifier": run_identifier,
                            # Core Summary
                            "train_loss": avg_train_loss,
                            "val_pr_auc": val_metrics.get("PR-AUC", 0.0),
                            "val_auroc": val_metrics.get("AUROC", 0.0),
                            "val_f1_binary": val_metrics.get("F1_Binary", 0.0),
                            "val_f1_macro": val_metrics.get("F1_Macro", 0.0),
                            # Operational & Convergence Metrics
                            "calibrated_tau": val_metrics.get("Calibrated_Tau", 0.0),
                            "precision": val_metrics.get("Prec", 0.0),
                            "recall": val_metrics.get("Rec", 0.0),
                            "fpr": val_metrics.get("FPR", 0.0),
                            "fnr": val_metrics.get("FNR", 0.0),
                            # Count Metrics for Convergence Tracking
                            "tp_count": val_metrics.get("TP", 0),
                            "fp_count": val_metrics.get("FP", 0),
                            "tn_count": val_metrics.get("TN", 0),
                            "fn_count": val_metrics.get("FN", 0),
                            "total_detected_anomalies": val_metrics.get("TP", 0) + val_metrics.get("FP", 0),
                            # Computational Metrics
                            "params_m": complexity_info["params_m"],
                            "forward_gflops": complexity_info["gflops"],
                            "inference_latency_ms": inference_latency_ms,
                            "epoch_duration_sec": epoch_duration_sec,
                            "throughput_samples_per_sec": throughput,
                            "peak_vram_gb": peak_vram_gb,
                        })

                        # Check early stopping condition
                        if is_early_stop:
                            logger.info(f"\n✋ Early stopping triggered at Epoch {epoch:02d}. Advancing to next model...")
                            break

                    # Save structured history alongside best model checkpoint
                    history_path = save_dir / "training_history.json"
                    with open(history_path, "w") as f:
                        json.dump(metrics_history, f, indent=2, cls=NpEncoder)

                except Exception as e:
                    logger.error(f"[ERROR] Failed at Mask: {mask_name}, Group: {run_identifier} -> {e}")
                    # Decide whether to raise or continue to the next fold
                    continue

                finally:
                    # 7. Reclaim GPU/CPU memory BEFORE the next sweep fold starts
                    cleanup_pipeline_resources(
                        model, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        evaluator
                    )
                    model = optimizer = train_loader = val_loader = evaluator = None


def main():

    # 1. Pipeline Registry Map
    MODEL_PIPELINES = {
        "omni": run_omni_pipeline,
        "patchtst": run_patchtst_pipeline,
        "ts2vec": run_ts2vec_pipeline,
        "at": run_at_pipeline,
        "lstm_vae": run_lstm_vae_pipeline,
        "conv_ae": run_conv_ae_pipeline,
    }


    parser = argparse.ArgumentParser(
        description="Aviation Flight Data Anomaly Detection Pipeline Execution Engine"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["omni"],
        help="List of model pipelines to run sequentially (e.g., --models omni at lstm_vae conv_ae)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments_features.yaml",
        help="Path to experiment configuration YAML file",
    )

    args = parser.parse_args()

    # Normalize model strings
    requested_models = [m.lower().strip() for m in args.models]

    logger.info(f"🚀 Execution Queue: {requested_models}")
    logger.info(f"📄 Configuration File: {args.config}")

    for model_name in requested_models:
        if model_name not in MODEL_PIPELINES:
            logger.error(
                f"❌ Model '{model_name}' not recognized. Available models: {list(MODEL_PIPELINES.keys())}"
            )
            continue

        logger.info(f"\n==================================================")
        logger.info(f"  STARTING PIPELINE: {model_name.upper()}")
        logger.info(f"==================================================")

        try:
            # Execute model training runner
            pipeline_fn = MODEL_PIPELINES[model_name]
            pipeline_fn(exp_config_path=args.config)
            logger.info(f"✅ Successfully finished training: {model_name}\n")
        except Exception as e:
            logger.error(f"💥 Pipeline failed for model '{model_name}': {e}", exc_info=True)
        finally:
            # Clean up residual model weights, optimizer states, and CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


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

    # Pass list directly via shell magic for colab
    # !python train_4c.py --models omni lstm_vae conv_ae --config config/experiments.yaml