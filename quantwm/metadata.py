from __future__ import annotations
import torch
@torch.no_grad()
def fake_quantize_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Round through the same finite E4M3 value set used by QVG scales."""
    from quant_videogen.sim.quant.lowbit_quantize import floatExMy_quantize_triton
    return floatExMy_quantize_triton(tensor.contiguous(), e_bit=4, m_bit=3, use_all_one_exponent=True, use_all_one_mantissa=False).to(tensor.dtype)

@torch.no_grad()
def fake_quantize_int8_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric INT8 metadata with one FP16 secondary scale per tensor."""
    work = tensor.float()
    secondary_scale = work.abs().amax().div(127.0).clamp_min(1e-12)
    return (work.div(secondary_scale).round().clamp(-127, 127) * secondary_scale).to(tensor.dtype)

@torch.no_grad()
def fake_quantize_positive_int8_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Positive INT8 scale storage with one FP16 secondary scale.

    Quantizer scales must remain strictly positive: permitting code zero makes
    the subsequent normalization undefined. Since scales are non-negative,
    all 127 positive INT8 magnitudes are used and finite inputs stay finite.
    """
    work = tensor.float()
    secondary_scale = work.amax().div(127.0).clamp_min(1e-12)
    codes = work.div(secondary_scale).round().clamp(1, 127)
    return (codes * secondary_scale).to(tensor.dtype)

@torch.no_grad()
def fake_quantize_scale_zero(scale: torch.Tensor, zero_point: torch.Tensor, *, storage_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dequantized scale/zero values for the selected storage format."""
    if storage_format == 'fp8':
        return (fake_quantize_e4m3(scale), fake_quantize_e4m3(zero_point))
    if storage_format == 'int8':
        return (fake_quantize_positive_int8_per_tensor(scale), zero_point.to(torch.int8).to(zero_point.dtype))
    if storage_format in {'native', 'bf16'}:
        return (scale, zero_point)
    raise ValueError(f'unsupported scale/zero metadata format: {storage_format!r}')

@torch.no_grad()
def fake_quantize_centroids(centroids: torch.Tensor, *, storage_format: str) -> torch.Tensor:
    if storage_format in {'native', 'bf16'}:
        return centroids
    if storage_format == 'fp8':
        return fake_quantize_e4m3(centroids)
    raise ValueError(f'unsupported centroid metadata format: {storage_format!r}')
