"""Clinical-Longformer embedding cache.

Runs Clinical-Longformer (yikuan8/Clinical-Longformer) on a list of texts
once and caches the resulting (N, 768) embeddings to disk. Subsequent
preprocessing runs reuse the cache instead of re-running the LM.

Usage:
    from src.preprocessing.longformer_cache import get_or_compute_embeddings
    emb = get_or_compute_embeddings(
        texts=df['note'].tolist(),
        cache_path='data/mcmed/cache/rad_embeddings.npy',
        model_name='yikuan8/Clinical-Longformer',
        batch_size=16,
        max_length=1024,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np


def get_or_compute_embeddings(
    texts: List[str],
    cache_path: str | Path,
    model_name: str = "yikuan8/Clinical-Longformer",
    batch_size: int = 16,
    max_length: int = 1024,
    overwrite: bool = False,
) -> np.ndarray:
    """Return (N, 768) Longformer embeddings, computing and caching if needed."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            print(f"  Loaded cached embeddings from {cache_path}: {emb.shape}")
            return emb
        print(f"  Cache size mismatch ({emb.shape[0]} vs {len(texts)}); recomputing")

    print(f"  Computing Longformer embeddings for {len(texts)} texts ...")
    emb = _compute_embeddings(texts, model_name, batch_size, max_length)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    print(f"  Saved cache: {cache_path} ({emb.shape}, {emb.nbytes / 1e6:.0f} MB)")
    return emb


def _compute_embeddings(
    texts: List[str], model_name: str, batch_size: int, max_length: int,
) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from torch.cuda.amp import autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=max_length,
            ).to(device)
            with autocast(enabled=device.type == "cuda"):
                z = model(**inputs).last_hidden_state.mean(dim=1)
            out.append(z.float().cpu().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"    {i + len(batch)}/{len(texts)} done")

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float32)
