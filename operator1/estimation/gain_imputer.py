"""GAIN (Generative Adversarial Imputation Networks) for MNAR data.

A lightweight adversarial imputer specifically designed for data that
is Missing Not At Random.  The generator learns to produce realistic
imputations while a discriminator tries to distinguish real from
imputed values.  This adversarial training is robust to MNAR because
the discriminator forces plausible imputations even when the missingness
mechanism is informative.

Reference: Yoon, Jordon, van der Schaar (2018) "GAIN: Missing Data
Imputation using Generative Adversarial Nets" (ICML 2018).

Requires ``torch >= 2.0``.  Falls back gracefully if unavailable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_EPOCHS = 100
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_HIDDEN_DIM = 32
_DEFAULT_LR = 1e-3
_DEFAULT_ALPHA = 10.0  # reconstruction weight
_DEFAULT_HINT_RATE = 0.9  # fraction of mask hints revealed to discriminator
_MIN_TRAIN_ROWS = 30


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class GAINResult:
    """Result container for GAIN imputation."""

    imputed_values: dict[str, pd.Series] = field(default_factory=dict)
    confidence_scores: dict[str, pd.Series] = field(default_factory=dict)
    d_loss_final: float = 0.0
    g_loss_final: float = 0.0
    n_epochs_trained: int = 0
    fallback_used: bool = False


def train_and_impute_gain(
    df: pd.DataFrame,
    target_vars: list[str],
    feature_cols: list[str],
    mnar_masks: dict[str, pd.Series] | None = None,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    epochs: int = _DEFAULT_EPOCHS,
    lr: float = _DEFAULT_LR,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    alpha: float = _DEFAULT_ALPHA,
    hint_rate: float = _DEFAULT_HINT_RATE,
) -> GAINResult:
    """Train GAIN and impute MNAR-classified missing values.

    Parameters
    ----------
    df:
        Full feature table.
    target_vars:
        Variables to impute.
    feature_cols:
        Predictor columns.
    mnar_masks:
        Per-variable masks indicating which rows are MNAR.
    hidden_dim:
        Hidden layer width for G and D.
    epochs:
        Training epochs.
    lr:
        Learning rate.
    batch_size:
        Mini-batch size.
    alpha:
        Reconstruction weight (forces G to match observed values).
    hint_rate:
        Fraction of mask hints given to discriminator.

    Returns
    -------
    GAINResult
    """
    result = GAINResult()

    if not _check_torch():
        logger.warning("torch not available -- GAIN imputer cannot run")
        result.fallback_used = True
        return result

    import torch
    import torch.nn as nn

    # Build joint matrix
    all_cols = list(set(feature_cols + target_vars))
    all_cols = [c for c in all_cols if c in df.columns]
    dim = len(all_cols)

    X_raw = df[all_cols].copy()
    for col in feature_cols:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].ffill().bfill()

    # Normalize
    means = X_raw.mean()
    stds = X_raw.std()
    stds[stds < 1e-10] = 1.0
    X_norm = ((X_raw - means) / stds).fillna(0)

    # Mask matrix: 1 where observed, 0 where missing
    mask_matrix = X_raw.notna().astype(float)

    X_np = X_norm.values.astype(np.float32)
    M_np = mask_matrix.values.astype(np.float32)

    n_rows = X_np.shape[0]
    if n_rows < _MIN_TRAIN_ROWS:
        logger.warning("GAIN: only %d rows (need %d) -- skipping", n_rows, _MIN_TRAIN_ROWS)
        result.fallback_used = True
        return result

    # --- Define Generator and Discriminator ---

    class Generator(nn.Module):
        def __init__(self, d: int, h: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d * 2, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, d),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            inp = torch.cat([x, m], dim=1)
            return self.net(inp)

    class Discriminator(nn.Module):
        def __init__(self, d: int, h: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d * 2, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, d),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
            inp = torch.cat([x, hint], dim=1)
            return self.net(inp)

    G = Generator(dim, hidden_dim)
    D = Discriminator(dim, hidden_dim)
    G_opt = torch.optim.Adam(G.parameters(), lr=lr)
    D_opt = torch.optim.Adam(D.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X_np)
    M_tensor = torch.FloatTensor(M_np)

    # --- Training loop ---
    d_loss_val = 0.0
    g_loss_val = 0.0

    for epoch in range(epochs):
        # Shuffle indices
        perm = torch.randperm(n_rows)

        for i in range(0, n_rows, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X_tensor[idx]
            m_batch = M_tensor[idx]

            # Random noise for missing entries
            z = torch.rand_like(x_batch)
            # Combine: observed values + noise for missing
            x_input = m_batch * x_batch + (1 - m_batch) * z

            # Generate
            g_out = G(x_input, m_batch)
            x_hat = m_batch * x_batch + (1 - m_batch) * g_out

            # Hint vector: reveal some of the mask to D
            hint_random = torch.rand_like(m_batch)
            hint = m_batch * 1.0  # reveal all observed
            hint[hint_random > hint_rate] = 0.5  # ambiguate some

            # --- Train Discriminator ---
            D_opt.zero_grad()
            d_pred = D(x_hat.detach(), hint)
            d_loss = -torch.mean(
                m_batch * torch.log(d_pred + 1e-8)
                + (1 - m_batch) * torch.log(1 - d_pred + 1e-8)
            )
            d_loss.backward()
            D_opt.step()

            # --- Train Generator ---
            G_opt.zero_grad()
            g_out2 = G(x_input, m_batch)
            x_hat2 = m_batch * x_batch + (1 - m_batch) * g_out2
            d_pred2 = D(x_hat2, hint)

            # Adversarial loss: fool D on missing entries
            g_adv_loss = -torch.mean((1 - m_batch) * torch.log(d_pred2 + 1e-8))
            # Reconstruction loss on observed entries
            g_recon_loss = torch.mean((m_batch * (x_batch - g_out2)) ** 2)
            g_loss = g_adv_loss + alpha * g_recon_loss

            g_loss.backward()
            G_opt.step()

            d_loss_val = float(d_loss.item())
            g_loss_val = float(g_loss.item())

    result.d_loss_final = d_loss_val
    result.g_loss_final = g_loss_val
    result.n_epochs_trained = epochs

    # --- Impute missing values ---
    with torch.no_grad():
        z_full = torch.rand_like(X_tensor)
        x_input_full = M_tensor * X_tensor + (1 - M_tensor) * z_full
        g_imputed = G(x_input_full, M_tensor)
        x_complete = M_tensor * X_tensor + (1 - M_tensor) * g_imputed

    X_imputed = x_complete.numpy()
    # Denormalize
    X_imputed = X_imputed * stds.values.astype(np.float32) + means.values.astype(np.float32)
    X_imputed_df = pd.DataFrame(X_imputed, index=df.index, columns=all_cols)

    # --- Extract results for target variables ---
    for var in target_vars:
        if var not in all_cols:
            continue

        mnar_mask = mnar_masks.get(var, df[var].isna()) if mnar_masks else df[var].isna()
        imputed = pd.Series(np.nan, index=df.index, dtype=float)
        confidence = pd.Series(np.nan, index=df.index, dtype=float)

        imputed[mnar_mask] = X_imputed_df.loc[mnar_mask, var]

        # Confidence from discriminator: how well can D tell these are fake?
        with torch.no_grad():
            hint_eval = M_tensor * 1.0
            d_scores = D(x_complete, hint_eval).numpy()

        var_idx = all_cols.index(var)
        d_var = d_scores[:, var_idx]
        # Higher D score on missing entries = G fooled D = better imputation
        conf_raw = pd.Series(d_var, index=df.index)
        conf_raw = conf_raw.clip(0.1, 0.85)  # cap confidence for MNAR
        confidence[mnar_mask] = conf_raw[mnar_mask]

        result.imputed_values[var] = imputed
        result.confidence_scores[var] = confidence

    return result
