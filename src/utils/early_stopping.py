import numpy as np
import torch
import logging

logger = logging.getLogger("AviationInsight")

class EarlyStopping:
    """
    Early stopping handler to terminate training when a monitored metric 
    stops improving after a given patience window.
    """
    def __init__(
        self, 
        patience: int = 7, 
        delta: float = 1e-4, 
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Args:
            patience: How many epochs to wait after last time monitored metric improved.
            delta: Minimum change in monitored metric to qualify as an improvement.
            mode: One of 'min' (for loss) or 'max' (for AUROC / PR-AUC / F1).
            verbose: If True, prints a message for each improvement.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    @property
    def best_metric_val(self) -> float:
        """Returns the raw un-transformed best metric value."""
        if self.best_score is None:
            return 0.0 if self.mode == "max" else float("inf")
        return self.best_score if self.mode == "max" else -self.best_score

    def __call__(self, val_metric: float) -> bool:
        score = val_metric if self.mode == "max" else -val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"  ⏳ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop