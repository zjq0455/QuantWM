"""Shared KV2 reconstruction; no video or attention-quality evaluation."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import torch

from .clustering import range_aware_historical_prq_quantize_tensor
from .correction import query_lowrank_error_sidecar
from .history import HistoricalQueryStatsTracker
from .rounding import full_int2_affine_fake_quantize, qaware_adaptive_full_int2_fake_quantize


class Runtime:
    def __init__(self, layers: int):
        self.method = os.environ.get('QUANTWM_METHOD', 'bf16')
        if self.method not in {'bf16', 'qvg', 'quantwm'}:
            raise ValueError(self.method)
        self.tracker = HistoricalQueryStatsTracker(layer_count=layers, query_samples=16)
        self.quantized_layer_spans = 0
        self.quantization_events = 0
        atexit.register(self.save)

    def save(self):
        filename = os.environ.get('QUANTWM_RUNTIME_REPORT')
        if filename and int(os.environ.get('RANK', '0')) == 0:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                'method': self.method, 'execution': 'fake KV2, dequantized tensors',
                'quantized_layer_spans': self.quantized_layer_spans,
                'quantization_events': self.quantization_events,
                'group_size': 64, 'centroids': 256, 'kmeans_iterations': 2,
                'qlr_rank': 8 if self.method == 'quantwm' else 0,
                'query_samples': 16 if self.method == 'quantwm' else 0,
                'qlr_coefficient_format': 'int8' if self.method == 'quantwm' else None,
                'query_subspace': 'Q16/eigh' if self.method == 'quantwm' else None,
                'scale_format': 'fp8', 'centroid_format': 'bf16',
            }, indent=2) + '\n')

    @torch.no_grad()
    def pair(self, k, v, *, config, layer, span_start, tokens_per_frame, queries=None):
        from quant_videogen.compress import get_quantize_fn
        from quant_videogen.functions import prq_quantize_tensor
        if self.method == 'bf16':
            return k, v
        standard = get_quantize_fn(config.quant_type, config)
        weight = self.tracker.get_weight(layer, heads=k.shape[1], dim=k.shape[-1], device=k.device)
        queries = self.tracker.get_queries(layer) if queries is None else queries
        if tokens_per_frame <= 0 or k.shape[2] % tokens_per_frame:
            raise ValueError(f'KV span {k.shape[2]} does not contain complete frames of {tokens_per_frame} tokens')
        devices = [k.device.index] if k.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1234 + int(span_start) + 1000 * int(layer))
            if self.method == 'qvg':
                kq = prq_quantize_tensor(k, num_stages=config.num_prq_stages,
                    codebook_size=min(config.cache_num_k_centroids,k.shape[2]),
                    kmeans_max_iters=config.kmeans_max_iters,quantize_fn=standard)
            else:
                def a4(value):
                    return qaware_adaptive_full_int2_fake_quantize(value,block_size=64,
                        tokens_per_frame=tokens_per_frame,queries=queries)
                kq = range_aware_historical_prq_quantize_tensor(k,channel_weights=weight,
                    num_stages=config.num_prq_stages,
                    codebook_size=min(config.cache_num_k_centroids,k.shape[2]),
                    kmeans_max_iters=config.kmeans_max_iters,quantize_fn=a4,
                    quant_block_size=64,rerank_candidates=4)
                kq = query_lowrank_error_sidecar(k,kq,queries=queries,rank=8,
                    coefficient_format='int8',decomposition='eigh')
            torch.manual_seed(5678 + int(span_start) + 1000 * int(layer))
            def v4(value):
                return full_int2_affine_fake_quantize(value,block_size=64)
            vq = prq_quantize_tensor(v,num_stages=config.num_prq_stages,
                codebook_size=min(config.cache_num_v_centroids,v.shape[2]),
                kmeans_max_iters=config.kmeans_max_iters,
                quantize_fn=v4 if self.method == 'quantwm' else standard)
        self.quantized_layer_spans += 1
        return kq.to(k.dtype), vq.to(v.dtype)

