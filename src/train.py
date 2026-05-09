"""Generic training loop for MM-GraphSurv (dataset-agnostic)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .data import build_dataloaders, modality_slices_for
from .losses import MMGraphSurvLoss
from .models import MMGraphSurv
from .eval.metrics import evaluate_split, eval_summary
from .utils import seed_everything, get_device, resolve_path, ensure_dir


def build_model(cfg: dict) -> MMGraphSurv:
    # Merge raw_modality_dims (pre-pool dims for rad/icd) into modality_dims
    # so the model knows the input dimension for PoolableModalityEncoder
    all_dims = dict(cfg["model"]["modality_dims"])
    all_dims.update(cfg["model"].get("raw_modality_dims", {}))
    return MMGraphSurv(
        modality_dims=all_dims,
        modality_keys=cfg["dataset"]["modalities"],
        pool_dims=cfg["model"].get("pool_dims", {}),
        num_windows=cfg["data"].get("n_blocks", cfg["data"].get("max_visits", 1)),
        num_risks=cfg["dataset"]["num_risks"],
        num_durations=cfg["model"]["num_durations"],
        k_neighbors=cfg["model"]["k_neighbors"],
        alpha_ema=cfg["model"]["alpha_ema"],
        gin_dims=tuple(cfg["model"]["gin_dims"]),
        dropout=cfg["model"]["dropout"],
        use_temporal_edges=cfg["model"].get("use_temporal_edges", True),
        cause_specific_proj=cfg["model"].get("cause_specific_proj", False),
        cause_specific_gin=cfg["model"].get("cause_specific_gin", False),
        per_cause_hazard=cfg["model"].get("per_cause_hazard", False),
        modality_types=cfg["dataset"].get("modality_types"),
        fusion_mode=cfg["model"].get("fusion_mode", "graph"),
        adjacency_mode=cfg["model"].get("adjacency_mode", "learned"),
    )


# ---------------------------------------------------------------------------
# Monitoring helpers
# ---------------------------------------------------------------------------

def _param_stats(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Per-module weight norm and max abs value."""
    stats = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Group by top-level module (e.g. "encoders.dynamic", "gin", "heads")
        prefix = name.split(".")[0]
        if prefix not in stats:
            stats[prefix] = {"w_norm": 0.0, "w_max": 0.0, "g_norm": 0.0, "g_max": 0.0, "n": 0}
        s = stats[prefix]
        s["w_norm"] += p.data.norm().item() ** 2
        s["w_max"] = max(s["w_max"], p.data.abs().max().item())
        if p.grad is not None:
            s["g_norm"] += p.grad.norm().item() ** 2
            s["g_max"] = max(s["g_max"], p.grad.abs().max().item())
        s["n"] += p.numel()
    # Sqrt the accumulated squared norms
    for s in stats.values():
        s["w_norm"] = s["w_norm"] ** 0.5
        s["g_norm"] = s["g_norm"] ** 0.5
    return stats


def _fmt_monitor(stats: dict, loss_parts: dict, lr: float) -> str:
    """Format one-line monitoring summary."""
    parts_str = "  ".join(f"{k}={v:.4f}" for k, v in loss_parts.items())
    lines = [f"    loss: {parts_str}  lr={lr:.1e}"]
    for mod, s in sorted(stats.items()):
        lines.append(
            f"    {mod:12s}  W|{s['w_norm']:7.2f}|  Wmax={s['w_max']:.3f}"
            f"  G|{s['g_norm']:7.4f}|  Gmax={s['g_max']:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_dataset(cfg: dict, run_dir: str | Path | None = None) -> dict:
    seed_everything(cfg["data"]["seed"])
    device = get_device()

    loaders = build_dataloaders(cfg)
    slices = modality_slices_for(cfg)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} ({n_params/1e6:.2f}M params)")

    # Auto class weights: inverse frequency per event label, capped.
    cw = cfg["loss"].get("class_weights")
    if cw is None or cw == "auto":
        train_events = loaders["train"].dataset.events.numpy()
        num_risks = cfg["dataset"]["num_risks"]
        counts = np.bincount(train_events, minlength=num_risks + 1).astype(float)
        counts = np.maximum(counts, 1.0)
        total = counts.sum()
        weights = total / (len(counts) * counts)
        weights = np.clip(weights, 0.5, 10.0)
        cw = weights.tolist()
        print(f"  Auto class weights: {[f'{w:.2f}' for w in cw]}")

    # Precompute global per-bin rank weights if bin-stratified ranking is on.
    bin_rank_weights = None
    if cfg["loss"].get("bin_stratified_rank", False):
        from .losses import compute_bin_rank_weights
        num_bins_ = cfg["model"].get("num_durations", cfg["data"].get("num_durations"))
        tds = loaders["train"].dataset
        bin_rank_weights = compute_bin_rank_weights(
            tds.events, tds.durations_idx,
            num_bins=num_bins_,
            smoothing=cfg["loss"].get("bin_rank_smoothing", "sqrt"),
            max_ratio=cfg["loss"].get("bin_rank_max_ratio", 3.0),
        )
        print(f"  Bin-stratified rank weights: {[f'{v:.2f}' for v in bin_rank_weights.tolist()]}")

    loss_fn = MMGraphSurvLoss(
        alpha_nll=cfg["loss"]["alpha_nll"],
        beta_rank=cfg["loss"]["beta_rank"],
        lambda_reg=cfg["loss"]["lambda_reg"],
        rank_sigma=cfg["loss"]["rank_sigma"],
        class_weights=cw,
        cif_smooth_weight=cfg["loss"].get("cif_smooth_weight", 0.0),
        per_cause_hazard=cfg["model"].get("per_cause_hazard", False),
        bin_stratified_rank=cfg["loss"].get("bin_stratified_rank", False),
        num_bins=cfg["model"].get("num_durations", cfg["data"].get("num_durations")),
        bin_rank_weights=bin_rank_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg["training"]["lr_patience"],
    )

    _name_map = {"eicu": "eICU", "mimic": "MIMIC", "mcmed": "MCMED",
                  "pbc2": "PBC2", "support": "SUPPORT",
                  "gbsg": "GBSG", "metabric": "METABRIC",
                  "hirid": "HIRID", "hirid_circ": "HIRID"}
    cuts_path = resolve_path(cfg["data"]["processed_dir"]) / f"cuts_{_name_map[cfg['dataset']['name']]}.npy"
    cuts = np.load(cuts_path)

    if run_dir is None:
        run_dir = resolve_path(f"checkpoints/{cfg['dataset']['name']}_{int(time.time())}")
    run_dir = ensure_dir(run_dir)
    print(f"Run dir: {run_dir}")

    best_val = float("inf")
    bad_epochs = 0
    history = []

    # Opt-in augmentations (HiRID-specific; default disabled everywhere else).
    mixup_alpha     = float(cfg["training"].get("mixup_alpha", 0.0))
    feat_dropout_p  = float(cfg["training"].get("feature_dropout_dynamic", 0.0))
    feat_dropout_range = cfg["training"].get("feature_dropout_range", None)  # (lo, hi)

    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train()
        train_losses = []
        epoch_parts = {"nll": 0.0, "rank": 0.0, "smooth": 0.0}
        n_batches = 0

        for batch in loaders["train"]:
            x = batch["x"].to(device)
            durations_idx = batch["durations_idx"].to(device)
            events = batch["events"].to(device)
            modality_mask = batch.get("modality_mask")
            if modality_mask is not None:
                modality_mask = modality_mask.to(device)

            # ---- Augmentation (a): feature dropout on dynamic block only ----
            if feat_dropout_p > 0 and feat_dropout_range is not None:
                lo, hi = feat_dropout_range
                N, S, F_total = x.shape
                # Per-patient mask: each feature zeroed independently with prob p
                # Same mask across all 6 windows (so a feature is either "off" for
                # the whole observation or "on").
                dyn_mask = (torch.rand(N, 1, hi - lo, device=x.device) > feat_dropout_p).float()
                full_mask = torch.ones(N, 1, F_total, device=x.device)
                full_mask[:, :, lo:hi] = dyn_mask
                x = x * full_mask   # broadcasts over S

            # ---- Augmentation (b): input mixup ----
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[perm]
                durations_idx_b = durations_idx[perm]
                events_b = events[perm]
                mixup_lam = lam
            else:
                durations_idx_b = None
                mixup_lam = 1.0

            optimizer.zero_grad()
            out = model(x, modality_slices=slices, modality_mask=modality_mask)

            if mixup_alpha > 0:
                # Convex-combine the two loss halves: same x, two label sets
                loss_a, parts_a = loss_fn(out, durations_idx, events)
                loss_b, parts_b = loss_fn(out, durations_idx_b, events_b)
                loss = mixup_lam * loss_a + (1 - mixup_lam) * loss_b
                parts = {k: mixup_lam * parts_a[k] + (1 - mixup_lam) * parts_b[k]
                         for k in parts_a}
            else:
                loss, parts = loss_fn(out, durations_idx, events)

            if torch.isnan(loss):
                print(f"  *** NaN loss at epoch {epoch} batch {n_batches}! ***")
                print(f"      parts: nll={parts['nll'].item():.4f} rank={parts['rank'].item():.4f} smooth={parts['smooth'].item():.4f}")
                print(f"      logits: min={out['logits'].min().item():.4f} max={out['logits'].max().item():.4f}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["clip_norm"])
            optimizer.step()
            train_losses.append(loss.item())
            for k in epoch_parts:
                epoch_parts[k] += parts[k].item()
            n_batches += 1

        if n_batches == 0:
            print("  No valid batches — stopping.")
            break

        for k in epoch_parts:
            epoch_parts[k] /= n_batches

        # Collect weight/grad stats after last training batch
        pstats = _param_stats(model)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loaders["val"]:
                x = batch["x"].to(device)
                durations_idx = batch["durations_idx"].to(device)
                events = batch["events"].to(device)
                modality_mask = batch.get("modality_mask")
                if modality_mask is not None:
                    modality_mask = modality_mask.to(device)
                out = model(x, modality_slices=slices, modality_mask=modality_mask)
                loss, _ = loss_fn(out, durations_idx, events)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        **{f"loss_{k}": v for k, v in epoch_parts.items()}})

        print(f"  E{epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}")
        print(_fmt_monitor(pstats, epoch_parts, lr_now))

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["training"]["patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

    # Evaluate best
    ckpt = run_dir / "best_model.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    print("\n=== Test set evaluation (best checkpoint) ===")
    metrics = evaluate_split(
        model, loaders["test"], cuts,
        num_risks=cfg["dataset"]["num_risks"], device=device,
        modality_slices=slices,
    )
    print(eval_summary(metrics))

    import json
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "test_metrics": metrics}, f, indent=2)

    return metrics
