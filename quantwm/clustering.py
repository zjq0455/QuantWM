from __future__ import annotations
from typing import Any, Callable
import torch
@torch.no_grad()
def range_aware_historical_prq_quantize_tensor(tensor: torch.Tensor, *, channel_weights: torch.Tensor, num_stages: int, codebook_size: int, kmeans_max_iters: int, quantize_fn: Callable[[torch.Tensor], torch.Tensor], quant_block_size: int=64, rerank_candidates: int=4, assignment_chunk_size: int=2048, stats: Any | None=None, cluster_mean_compensation: bool=False, centroid_format: str='native') -> torch.Tensor:
    """Historical-Q with analytical residual-range-aware local reranking.

    For each token, coarse Historical-Q supplies a small candidate set that
    always includes its original assignment. Candidate centroids are scored
    by sum_j W_j * Range(residual_group_j)^2, the uniform-INT2 analytical
    proxy from the Range-HQ derivation. No fake quantization is executed while
    selecting a centroid. Centroids are updated with raw-K arithmetic means,
    and the final residual is quantized once by the configured A4 path.
    """
    import os
    centroid_format = os.getenv('KV_META_CENTROID_FORMAT', centroid_format)
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    if tensor.ndim != 4:
        raise ValueError(f'expected BHSD tensor, got {tuple(tensor.shape)}')
    (bsz, heads, seq, dim) = tensor.shape
    if channel_weights.shape != (heads, dim):
        raise ValueError(f'weights must be [H,D]={(heads, dim)}, got {tuple(channel_weights.shape)}')
    block_size = int(quant_block_size)
    if dim % block_size:
        raise ValueError(f'channel dimension {dim} must be divisible by block size {block_size}')
    batches = bsz * heads
    clusters = max(1, min(int(codebook_size), seq))
    candidate_count = max(1, min(int(rerank_candidates), clusters))
    weight = channel_weights.float().clamp_min(1e-08)
    weight = weight / weight.mean(dim=-1, keepdim=True).clamp_min(1e-08)
    weight_flat = weight.unsqueeze(0).expand(bsz, -1, -1).reshape(batches, dim)
    sqrt_weight = weight_flat.sqrt().unsqueeze(1)
    group_weight = weight_flat.reshape(batches, dim // block_size, block_size).sum(dim=-1)
    residual = tensor.reshape(batches, seq, dim).contiguous()
    labels_list: list[torch.Tensor] = []
    centers_list: list[torch.Tensor] = []
    for _ in range(int(num_stages)):
        scaled = (residual.float() * sqrt_weight).to(tensor.dtype).contiguous()
        (coarse_labels, coarse_centers_scaled, _, _) = batch_kmeans_Euclid(scaled, n_clusters=clusters, max_iters=int(kmeans_max_iters))
        coarse_labels = coarse_labels.long()
        offsets = torch.arange(batches, device=tensor.device, dtype=coarse_labels.dtype).mul(clusters).unsqueeze(1)
        flat_labels = (coarse_labels + offsets).reshape(-1)
        sums = torch.zeros(batches * clusters, dim, device=tensor.device, dtype=torch.float32)
        sums.index_add_(0, flat_labels, residual.float().reshape(-1, dim))
        counts = torch.bincount(flat_labels, minlength=batches * clusters).reshape(batches, clusters)
        occupied_centers = sums.reshape(batches, clusters, dim) / counts.clamp_min(1).unsqueeze(-1)
        fallback_centers = coarse_centers_scaled.float() / sqrt_weight
        centers = torch.where(counts.eq(0).unsqueeze(-1), fallback_centers, occupied_centers).to(tensor.dtype)
        scaled_centers = centers.float() * sqrt_weight
        center_norm = scaled_centers.square().sum(dim=-1).unsqueeze(1)
        reranked_chunks = []
        for start in range(0, seq, int(assignment_chunk_size)):
            stop = min(seq, start + int(assignment_chunk_size))
            scaled_tokens = scaled[:, start:stop].float()
            token_norm = scaled_tokens.square().sum(dim=-1, keepdim=True)
            distance = token_norm + center_norm - 2.0 * torch.bmm(scaled_tokens, scaled_centers.transpose(1, 2))
            nearest = distance.topk(k=candidate_count, dim=-1, largest=False, sorted=True).indices
            coarse = coarse_labels[:, start:stop]
            coarse_present = (nearest == coarse.unsqueeze(-1)).any(dim=-1)
            nearest = nearest.clone()
            nearest[:, :, -1] = torch.where(coarse_present, nearest[:, :, -1], coarse)
            candidate_centers = torch.gather(centers.unsqueeze(1).expand(-1, stop - start, -1, -1), 2, nearest.unsqueeze(-1).expand(-1, -1, -1, dim)).float()
            candidate_residual = residual[:, start:stop].float().unsqueeze(2) - candidate_centers
            groups = candidate_residual.reshape(batches, stop - start, candidate_count, dim // block_size, block_size)
            group_range = groups.amax(dim=-1) - groups.amin(dim=-1)
            range_score = (group_range.square() * group_weight[:, None, None, :]).sum(dim=-1)
            winner = range_score.argmin(dim=-1, keepdim=True)
            selected_labels = torch.gather(nearest, 2, winner).squeeze(-1)
            reranked_chunks.append(selected_labels)
        labels = torch.cat(reranked_chunks, dim=1)
        rerank_flat_labels = (labels + offsets).reshape(-1)
        rerank_sums = torch.zeros(batches * clusters, dim, device=tensor.device, dtype=torch.float32)
        rerank_sums.index_add_(0, rerank_flat_labels, residual.float().reshape(-1, dim))
        rerank_counts = torch.bincount(rerank_flat_labels, minlength=batches * clusters).reshape(batches, clusters)
        updated_centers = (rerank_sums.reshape(batches, clusters, dim) / rerank_counts.clamp_min(1).unsqueeze(-1)).to(tensor.dtype)
        centers = torch.where(rerank_counts.eq(0).unsqueeze(-1), centers, updated_centers)
        from .metadata import fake_quantize_centroids
        stored_centers = fake_quantize_centroids(centers, storage_format=centroid_format)
        centers = stored_centers
        gathered = torch.gather(centers, 1, labels.unsqueeze(-1).expand(-1, -1, dim))
        residual = residual - gathered
        labels_list.append(labels)
        centers_list.append(centers)
    reconstruction = quantize_fn(residual.reshape(bsz, heads, seq, dim)).reshape(batches, seq, dim)
    for (labels, centers) in zip(labels_list, centers_list):
        reconstruction = reconstruction + torch.gather(centers, 1, labels.unsqueeze(-1).expand(-1, -1, dim))
    return reconstruction.reshape_as(tensor).to(tensor.dtype)
