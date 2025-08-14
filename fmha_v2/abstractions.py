from typing import Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


# same definition as fused_multihead_attention.h.
class AttentionMaskType(IntEnum):
    PADDING = 0
    CAUSAL = 1
    SLIDING_OR_CHUNKED_CAUSAL = 2
    CUSTOM_MASK = 3


class InputLayout(IntEnum):
    PACKED_QKV = 0
    CONTIGUOUS_Q_KV = 1
    Q_PAGED_KV = 2
    SEPARATE_Q_K_V = 3


@dataclass(frozen=True)
class FMHAKernelSpec:
    sm: int
    dtype: str
    seq_len: int
    head_size: int
    warps_m: int
    warps_n: int
    version: int
    interleaved: bool
    ldgsts_q: int
    ldgsts_k: int
    ldgsts_v: int
    share_smem_k_v: bool
    loop_step: int
    has_noloop: bool
    noloop_step: int
    unroll_threshold: int
    has_scale_max: bool
    ctas_per_head: int = 1
    sm_mma: int = 1
    head_interleaved: bool = True
    flash_attention: bool = False
    kv_loop_step: int = 64
    flash_attention_bh_upper_threshold: int = -1
    limit_qk_fragments: bool = False
    limit_v_fragments: bool = False
    tiled: int = 0
    warp_specialization: bool = False
    q_tile_buffers: int = 1
    kv_tile_buffers: int = 1
    scheduling_mode: int = 0
    input_layout: InputLayout = InputLayout.PACKED_QKV
    cross_mha: int = 0
    alibi: bool = True
    enable_attn_logit_softcapping: bool = False
    return_softmax_stats: bool = False
    disabled_mask_types: Optional[Tuple[int, int, int, int]] = None
    head_size_v: int = 0
    sage_block_sizes: Optional[Tuple[int, int, int]] = None
    output_dtype: Optional[str] = None
    is_mtp: bool = False
