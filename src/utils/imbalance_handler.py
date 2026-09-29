import numpy as np
import torch

class ImbalanceHandler:
    """
    Dedicated class for managing class imbalance recalibration, 
    threshold optimization, and robust score transformations.
    """
    def __init__(self, f_beta: float = 2.0, num_steps: int = 200):
        self.f_beta = f_beta
        self.num_steps = num_steps

    def calibrate_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        mode: str = "f1",
        auto_percentile: float = 95.0
    ) -> float:
        """
        Calculates anomaly threshold (tau).
        
        Args:
            y_true: Ground truth binary labels [N]
            y_scores: Continuous anomaly scores [N]
            mode: "f2" (prioritizes recall), "f1", or "percentile"
            auto_percentile: Percentile threshold applied when mode="percentile"
        """
        # Fallback to standard nominal percentile thresholding
        if mode == "percentile":
            nominal_mask = (y_true == 0)
            ref_scores = y_scores[nominal_mask] if np.any(nominal_mask) else y_scores
            return float(np.percentile(ref_scores, auto_percentile))

        # Dynamic F_beta optimization grid search
        beta_val = 2.0 if mode == "f2" else (1.0 if mode == "f1" else self.f_beta)
        thresholds = np.linspace(np.min(y_scores), np.max(y_scores), self.num_steps)
        best_fbeta = -1.0
        best_tau = float(thresholds[0])

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            prec = tp / max(tp + fp, 1e-8)
            rec = tp / max(tp + fn, 1e-8)

            fbeta = ((1 + beta_val**2) * prec * rec) / max(((beta_val**2) * prec) + rec, 1e-8)

            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_tau = float(thresh)

        # Enforce positive threshold floor here rather than distorting raw scores
        return max(0.0, best_tau)

    @staticmethod
    def distance_weighted_knn(
        topk_dists: torch.Tensor, 
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Applies inverse-distance weighting for imbalanced k-NN spaces."""
        weights = 1.0 / (topk_dists + eps)
        return torch.sum(topk_dists * weights, dim=-1) / torch.sum(weights, dim=-1)

    @staticmethod
    def robust_iqr_scale(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Standard continuous IQR scaling preserving relative score margins.
        Does NOT hard-clip scores to zero, avoiding artificial zero-bucket collapse.
        """
        q25, q75 = np.percentile(scores, [25, 75])
        iqr = q75 - q25
        if iqr == 0:
            return scores - np.median(scores)
        
        return (scores - q25) / (iqr + eps)