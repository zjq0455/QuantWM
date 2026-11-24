from __future__ import annotations
import torch
def _frame_view(tensor: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f'expected BHSD tensor, got {tuple(tensor.shape)}')
    (bsz, heads, seq, dim) = tensor.shape
    if seq % int(tokens_per_frame):
        raise ValueError(f'sequence length {seq} is not divisible by tokens_per_frame={tokens_per_frame}')
    return tensor.reshape(bsz, heads, seq // int(tokens_per_frame), int(tokens_per_frame), dim)

@torch.no_grad()
def qaware_adaptive_full_int2_fake_quantize(tensor: torch.Tensor, *, block_size: int, tokens_per_frame: int, queries: torch.Tensor | None, threshold_scope: str='frame', metadata_format: str='fp8', stats: QuantizerStats | None=None, eps: float=1e-08) -> torch.Tensor:
    """Four-level signed affine INT2 {-2,-1,0,+1} with anti-collapse."""
    import os
    metadata_format = os.getenv('KV_META_SCALE_ZERO_FORMAT', metadata_format)
    from quant_videogen.sim.quant.lowbit_quantize import floatExMy_quantize_triton
    if tensor.shape[-1] % int(block_size):
        raise ValueError(f'channel dimension {tensor.shape[-1]} must be divisible by block_size={block_size}')
    (qmin, qmax) = (-2, 1)
    blocks = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // int(block_size), int(block_size))
    block_min = blocks.amin(dim=-1, keepdim=True)
    block_max = blocks.amax(dim=-1, keepdim=True)
    scales = ((block_max - block_min) / float(qmax - qmin)).clamp_min(1e-10)
    scales = floatExMy_quantize_triton(scales.contiguous(), e_bit=4, m_bit=3, use_all_one_exponent=True, use_all_one_mantissa=False).float().clamp(min=1.0 / 512.0, max=torch.finfo(torch.float8_e4m3fn).max)
    zero_point = (qmin - torch.round(block_min / scales)).clamp(qmin, qmax)
    from .metadata import fake_quantize_scale_zero
    (scales, zero_point) = fake_quantize_scale_zero(scales, zero_point, storage_format=metadata_format)
    normalized = blocks / scales
    rtn_codes = torch.round(normalized + zero_point).clamp(qmin, qmax)
    normalized_flat = normalized.reshape_as(tensor.float())
    if queries is not None and queries.ndim == 4 and (queries.shape[0] == tensor.shape[0]) and (queries.shape[1] == tensor.shape[1]) and (queries.shape[-1] == tensor.shape[-1]):
        query_weight = queries.float().square().mean(dim=2, keepdim=True)
    else:
        query_weight = None
    if threshold_scope == 'frame':
        grouped = _frame_view(normalized_flat, tokens_per_frame)
        energy = grouped.square()
        if query_weight is not None:
            energy = energy * query_weight.unsqueeze(2)
        zero_region = grouped.abs() < 0.5
        collapse = (energy * zero_region).sum(dim=(3, 4), keepdim=True)
        collapse = collapse / energy.sum(dim=(3, 4), keepdim=True).clamp_min(eps)
    elif threshold_scope == 'quant_group':
        energy = normalized.square()
        if query_weight is not None:
            energy = energy * query_weight.reshape(tensor.shape[0], tensor.shape[1], 1, tensor.shape[-1] // int(block_size), int(block_size))
        zero_region = normalized.abs() < 0.5
        collapse = (energy * zero_region).sum(dim=-1, keepdim=True)
        collapse = collapse / energy.sum(dim=-1, keepdim=True).clamp_min(eps)
    else:
        raise ValueError(f'unsupported threshold_scope={threshold_scope!r}')
    collapse = collapse.clamp(0.0, 1.0 - 1e-06)
    root = torch.sqrt(1.0 - collapse)
    tau = root / (1.0 + root)
    zero_codes = zero_point.expand_as(rtn_codes)
    zero_mask = rtn_codes == zero_codes
    if threshold_scope == 'frame':
        tau_flat = tau.expand(tensor.shape[0], tensor.shape[1], tensor.shape[2] // tokens_per_frame, tokens_per_frame, tensor.shape[-1]).reshape_as(tensor)
        promote = zero_mask.reshape_as(tensor) & (normalized_flat.abs() >= tau_flat)
    else:
        promote = zero_mask & (normalized.abs() >= tau)
    positive_code = (zero_codes + 1).clamp(max=qmax)
    negative_code = (zero_codes - 1).clamp(min=qmin)
    direction_code = torch.where(blocks >= 0, positive_code, negative_code)
    promoted_codes = torch.where(promote.reshape_as(blocks) & (direction_code != zero_codes), direction_code, rtn_codes).clamp(qmin, qmax)
    output = ((promoted_codes - zero_point) * scales).reshape_as(tensor.float())
    rtn_output = ((rtn_codes - zero_point) * scales).reshape_as(tensor.float())
    return output.to(tensor.dtype)

@torch.no_grad()
def full_int2_affine_fake_quantize(tensor: torch.Tensor, *, block_size: int, metadata_format: str='fp8', stats: QuantizerStats | None=None) -> torch.Tensor:
    """Plain four-level affine INT2 RTN for the V residual path."""
    import os
    metadata_format = os.getenv('KV_META_SCALE_ZERO_FORMAT', metadata_format)
    from quant_videogen.sim.quant.lowbit_quantize import floatExMy_quantize_triton
    if tensor.shape[-1] % int(block_size):
        raise ValueError(f'channel dimension {tensor.shape[-1]} must be divisible by block_size={block_size}')
    (qmin, qmax) = (-2, 1)
    blocks = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // int(block_size), int(block_size))
    block_min = blocks.amin(dim=-1, keepdim=True)
    block_max = blocks.amax(dim=-1, keepdim=True)
    scales = ((block_max - block_min) / float(qmax - qmin)).clamp_min(1e-10)
    scales = floatExMy_quantize_triton(scales.contiguous(), e_bit=4, m_bit=3, use_all_one_exponent=True, use_all_one_mantissa=False).float().clamp(min=1.0 / 512.0, max=torch.finfo(torch.float8_e4m3fn).max)
    zero_point = (qmin - torch.round(block_min / scales)).clamp(qmin, qmax)
    from .metadata import fake_quantize_scale_zero
    (scales, zero_point) = fake_quantize_scale_zero(scales, zero_point, storage_format=metadata_format)
    codes = torch.round(blocks / scales + zero_point).clamp(qmin, qmax)
    output = ((codes - zero_point) * scales).reshape_as(tensor.float())
    return output.to(tensor.dtype)
