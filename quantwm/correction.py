from __future__ import annotations
import os
from typing import Any
import torch
def _fp8_e4m3(value: torch.Tensor) -> torch.Tensor:
    """Round a compact correction coefficient through QVG's E4M3 path."""
    from quant_videogen.sim.quant.lowbit_quantize import floatExMy_quantize_triton
    return floatExMy_quantize_triton(value.contiguous(), e_bit=4, m_bit=3, use_all_one_exponent=True, use_all_one_mantissa=False).float()

def _fp8_e5m2(value: torch.Tensor) -> torch.Tensor:
    """Round coefficients through QVG E5M2 fake path."""
    from quant_videogen.sim.quant.lowbit_quantize import floatExMy_quantize_triton
    return floatExMy_quantize_triton(value.contiguous(), e_bit=5, m_bit=2, use_all_one_exponent=False, use_all_one_mantissa=True).float()

def _scaled_int8_per_token(value: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """Symmetric INT8 fake quantization with one scale per token rank vector."""
    scale = value.abs().amax(dim=-1, keepdim=True).div(127.0)
    safe_scale = scale.clamp_min(eps)
    reconstructed = torch.round(value / safe_scale).clamp(-127, 127) * safe_scale
    return torch.where(scale > eps, reconstructed, torch.zeros_like(reconstructed))

def _quantize_qlr_coefficients(value, coefficient_format, eps):
    """Select QLR coefficient fake format; legacy fp8 remains E4M3."""
    selected = os.environ.get('KV_TRICK_QLR_COEFF_FORMAT', coefficient_format).lower()
    selected = {'fp8': 'fp8_e4m3', 'e4m3': 'fp8_e4m3', 'e5m2': 'fp8_e5m2'}.get(selected, selected)
    if selected == 'fp8_e4m3':
        return (_fp8_e4m3(value), 8.0, selected)
    if selected == 'fp8_e5m2':
        return (_fp8_e5m2(value), 8.0, selected)
    if selected == 'int8':
        return (_scaled_int8_per_token(value, eps=eps), 8.0, selected)
    if selected == 'fp4':
        raise ValueError('FP4 coefficients are not part of this release')
    raise ValueError(f'unsupported QLR coefficient format: {selected}')

@torch.no_grad()
def query_lowrank_error_sidecar(clean: torch.Tensor, quantized: torch.Tensor, *, queries: torch.Tensor | None, rank: int=4, coefficient_format: str='fp8', decomposition: str='svd', stats: Any | None=None, eps: float=1e-10) -> torch.Tensor:
    """Restore Query-visible K error with a compact rank-R sidecar."""
    valid = clean.ndim == 4 and quantized.shape == clean.shape and (queries is not None) and (queries.ndim == 4) and (queries.shape[:2] == clean.shape[:2]) and (queries.shape[-1] == clean.shape[-1]) and (queries.shape[2] > 0)
    if not valid:
        return quantized
    target = clean.float()
    base = quantized.float()
    query = queries.to(device=clean.device, dtype=torch.float32)
    use_rank = max(1, min(int(rank), query.shape[2], query.shape[3]))
    if decomposition == 'svd':
        (_, singular_values, vh) = torch.linalg.svd(query, full_matrices=False)
        basis = vh[..., :use_rank, :]
        reliable = singular_values[..., use_rank - 1] > eps
    elif decomposition == 'eigh':
        with torch.autocast(device_type=query.device.type, enabled=False):
            gram = torch.matmul(query.float(), query.float().transpose(-1, -2))
            gram = 0.5 * (gram + gram.transpose(-1, -2))
            (eigenvalues, eigenvectors) = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues[..., -use_rank:].flip(-1).clamp_min(0.0)
        eigenvectors = eigenvectors[..., -use_rank:].flip(-1)
        basis = torch.einsum('bhmd,bhmr->bhrd', query, eigenvectors)
        basis = basis / eigenvalues.sqrt().clamp_min(eps).unsqueeze(-1)
        reliable = eigenvalues[..., use_rank - 1] > eps * eps
    else:
        raise ValueError(f'unsupported QLR decomposition: {decomposition}')
    error = target - base
    coefficients = torch.einsum('bhsd,bhrd->bhsr', error, basis)
    (coefficients_q, coefficient_bits, selected_format) = _quantize_qlr_coefficients(coefficients, coefficient_format, eps)
    correction = torch.einsum('bhsr,bhrd->bhsd', coefficients_q, basis)
    candidate = base + correction
    base_logit_error = torch.einsum('bhqd,bhsd->bhqs', query, error)
    candidate_error = target - candidate
    candidate_logit_error = torch.einsum('bhqd,bhsd->bhqs', query, candidate_error)
    base_sse = base_logit_error.square().sum(dim=2).unsqueeze(-1)
    candidate_sse = candidate_logit_error.square().sum(dim=2).unsqueeze(-1)
    accept = reliable.unsqueeze(-1).unsqueeze(-1) & torch.isfinite(candidate_sse) & (candidate_sse < base_sse)
    corrected = torch.where(accept, candidate, base)
    return corrected.to(quantized.dtype)
