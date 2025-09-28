from typing import Dict, Literal
import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, gemm_fp8
from flashinfer.fused_moe.core import convert_to_block_layout, _maybe_get_cached_w2_permute_indices
from tests.utils_fp8 import to_float8

_cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n", [2560, 5120])
@pytest.mark.parametrize("k", [8192, 16384, 32768])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16])
@pytest.mark.parametrize("backend", ["trtllm", "auto"])
@pytest.mark.parametrize("auto_tuning", [False]) # FIXME: auto-tuning not working for now.
def test_gemm_fp8(
    m: int,
    n: int,
    k: int,
    input_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    res_dtype: torch.dtype,
    backend: Literal["trtllm", "auto"],
    auto_tuning: bool,
):
    torch.manual_seed(123)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    res = torch.zeros([m, n], device="cuda", dtype=res_dtype)
    global_scale = input_inv_s * mat2_inv_s

    permute_indices = _maybe_get_cached_w2_permute_indices(
        _cache_permute_indices,
        mat2_fp8,
        128
    )
    shuffled_weights = mat2_fp8[permute_indices.to(device=mat2_fp8.device)].contiguous()
    block_layout_weights = convert_to_block_layout(shuffled_weights, 128)
    with autotune(auto_tuning):
        gemm_fp8(
            input_fp8,
            block_layout_weights,
            global_scale,
            res,
            backend=backend,
        )

    reference = torch.mm(input, mat2.transpose(-2, -1))
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
