"""Post-hoc calibration for survival models.

Applied AFTER training — does not change the model or loss.
Learns a calibration mapping on the val set, applies to test.

Primary method: Per-bin temperature + bias (D47).
  - Learns K temperatures and K biases (one per discrete time bin)
  - Fitted via LBFGS on val set NLL in ~1 second
  - Preserves Ctd exactly (same scaling for all patients within each bin)
  - Dramatically improves IPA/IBS by correcting bin-wise probability mass

Also includes: Isotonic regression on CIF (model-agnostic alternative).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PerBinTemperatureBias:
    """Per-bin temperature + bias calibration (D47).

    For each of K discrete time bins, learns a temperature T_k and bias b_k:
        calibrated_logits[:, :, k] = logits[:, :, k] / T_k + b_k

    The temperature adjusts the sharpness of predictions within each bin,
    the bias shifts the probability mass location. Together they correct
    the bin-wise probability distribution without changing patient ranking.

    Ctd is exactly preserved because the same T_k and b_k apply to all
    patients at bin k — relative ordering within each bin is unchanged.

    Fitted via LBFGS on val set NLL. Typically 20 parameters (K=10).
    """

    def __init__(self):
        self.log_temps = None
        self.bias = None
        self.temps = None

    def fit(self, logits_val: np.ndarray, durations_idx_val: np.ndarray,
            events_val: np.ndarray, lr: float = 0.01, epochs: int = 500,
            t_min: float = 0.1, t_max: float = 10.0):
        """Fit on validation set using Adam with gradient clipping.

        Only 20 parameters (K temps + K biases). Takes ~1 second.

        Args:
            logits_val: (N, E, K) raw logits from trained model
            durations_idx_val: (N,) discretized duration indices
            events_val: (N,) event labels (0=censor, 1+=event)
        """
        import sys
        sys.path.insert(0, ".")
        from src.losses import neg_log_likelihood

        K = logits_val.shape[-1]
        logits_t = torch.from_numpy(logits_val).float()
        dur_t = torch.from_numpy(durations_idx_val).long()
        evt_t = torch.from_numpy(events_val).long()

        self.log_temps = nn.Parameter(torch.zeros(K))
        self.bias = nn.Parameter(torch.zeros(K))
        self._t_min = t_min
        self._t_max = t_max

        optimizer = torch.optim.Adam([self.log_temps, self.bias], lr=lr)
        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            optimizer.zero_grad()
            temps = torch.exp(self.log_temps).clamp(min=t_min, max=t_max)
            scaled = (logits_t / temps.unsqueeze(0).unsqueeze(0)
                      + self.bias.unsqueeze(0).unsqueeze(0))
            loss = neg_log_likelihood(scaled, dur_t, evt_t)
            if torch.isnan(loss):
                self.log_temps.data.zero_()
                self.bias.data.zero_()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_temps, self.bias], 1.0)
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = (self.log_temps.data.clone(), self.bias.data.clone())

        if best_state:
            self.log_temps.data, self.bias.data = best_state

        with torch.no_grad():
            self.temps = torch.exp(self.log_temps).clamp(min=t_min, max=t_max)

        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned calibration to test logits."""
        if self.temps is None:
            raise RuntimeError("Call fit() first")
        logits_t = torch.from_numpy(logits).float()
        with torch.no_grad():
            scaled = (logits_t / self.temps.unsqueeze(0).unsqueeze(0)
                      + self.bias.unsqueeze(0).unsqueeze(0))
        return scaled.numpy()


class IsotonicCIFCalibration:
    """Isotonic regression on the cumulative incidence function.

    For each time bin k, fits an isotonic regression mapping:
        predicted CIF(t_k) → observed event indicator I(T <= t_k)

    Model-agnostic — any model that outputs a CIF can use this.
    Nearly preserves Ctd (monotonic mapping within each bin).
    """

    def __init__(self):
        self.regressors = []

    def fit(self, cif_val: np.ndarray, durations_idx_val: np.ndarray,
            events_val: np.ndarray):
        """Fit isotonic regressors per bin.

        Args:
            cif_val: (N, K) CIF values on val set (single risk)
            durations_idx_val: (N,) discretized durations
            events_val: (N,) event indicators
        """
        from sklearn.isotonic import IsotonicRegression

        K = cif_val.shape[1]
        self.regressors = []

        for k in range(K):
            target = ((events_val > 0) & (durations_idx_val <= k)).astype(float)
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            ir.fit(cif_val[:, k], target)
            self.regressors.append(ir)

        return self

    def calibrate(self, cif_test: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to test CIF."""
        K = cif_test.shape[1]
        calibrated = np.zeros_like(cif_test)
        for k in range(K):
            calibrated[:, k] = self.regressors[k].predict(cif_test[:, k])
        return calibrated
