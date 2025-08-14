from typing import Tuple

from fmha_v2.constants import sm2name, hopper_traits2shape
from fmha_v2.abstractions import FMHAKernelSpec, InputLayout

pythonBoolean2cpp = {True: "true", False: "false"}


def get_effective_sm_and_name(fmha_kernel_spec: FMHAKernelSpec) -> Tuple[int, str]:
    sm = fmha_kernel_spec.sm
    # Override the mma instruction with an older one.
    if fmha_kernel_spec.sm_mma in sm2name:
        assert fmha_kernel_spec.sm_mma <= fmha_kernel_spec.sm, (
            "Instruction version should be at most target arch"
        )
        sm = fmha_kernel_spec.sm_mma
    sm_name = sm2name[sm]
    return sm, sm_name


def get_GMMA_shape(instruction_traits, m, n, k, warps_n) -> Tuple[int, int, int]:
    gmma_k = hopper_traits2shape[instruction_traits][-1]

    # gmma shape is 64xgmma_nx16, gmma_n should be as big as possible, but not bigger than n
    # gmma_n should also be smaller than 256
    gmma_m = 64
    gmma_n = 0
    # find the largest supported n
    n_supported = [(i + 1) * 8 for i in range(32)][::-1]
    n_target = n // warps_n
    assert n_target * warps_n == n
    assert n_supported[0] == 256 and n_supported[-1] == 8
    for cand_n in n_supported:
        if n_target % cand_n == 0:
            gmma_n = cand_n
            break
    assert gmma_n > 0, "No supported GMMA_N found!"

    return gmma_m, gmma_n, gmma_k


def get_hopper_instruction_traits(
    instruction_traits, kernel_spec: FMHAKernelSpec
) -> Tuple[str, str]:
    gmma_shape_p = get_GMMA_shape(
        instruction_traits,
        kernel_spec.loop_step,
        kernel_spec.seq_len,
        kernel_spec.head_size,
        kernel_spec.warps_n,
    )

    instruction_traits_p = f"{instruction_traits}<{', '.join([str(x) for x in gmma_shape_p])}, false, false>"

    gmma_shape_o = get_GMMA_shape(
        instruction_traits,
        kernel_spec.loop_step,
        kernel_spec.head_size,
        kernel_spec.seq_len,
        1,
    )
    instruction_traits_o = f"{instruction_traits}<{', '.join([str(x) for x in gmma_shape_o])}, true, false>"

    return instruction_traits_p, instruction_traits_o


def selected_mask_types(kspec: FMHAKernelSpec) -> Tuple[str, str, str, str]:
    # by default, we generate all combinations.
    # '1' means true, '0' means false.
    padding_mask = "1"
    causal_mask = "1"
    sliding_or_chunked_causal_mask = "1"
    custom_mask = "1"
    # only generate certain needed combinations of input_layout and mask types for trt-llm.
    if "GENERATE_CUBIN" in os.environ:
        if kspec.sage_block_sizes:
            # SageAttention only needs padding mask now
            causal_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"
        elif (kspec.head_size, kspec.head_size_v) == (192, 128):
            # MLA context phase only needs causal mask now
            padding_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"
        elif (kspec.head_size, kspec.head_size_v) == (576, 512):
            # MLA generation phase only needs padding mask (MtpMask) now
            causal_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"
        # encoder models (head_size = 32 / 64 / 128) need packed_qkv input layout + padding mask.
        elif kspec.input_layout == InputLayout.PACKED_QKV:
            # NOTE: 72 is added for vision transformer
            if kspec.head_size not in [32, 64, 72, 128]:
                padding_mask = "0"
        # only cross attention (head_size = 32/64/128) needs contiguous_q_kv input layout + padding mask / custom_mask.
        elif kspec.input_layout == InputLayout.CONTIGUOUS_Q_KV:
            causal_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            if kspec.head_size not in [32, 64, 72, 128]:
                padding_mask = "0"
                custom_mask = "0"
        # paged kv cache is always needed in gpt variants.
        # cross-attention also needs paged kv cache.
        elif kspec.input_layout == InputLayout.Q_PAGED_KV:
            if kspec.head_size not in [32, 64, 128]:
                padding_mask = "0"

        # alibi specialized kernels only need causal mask.
        if kspec.alibi and kspec.warp_specialization:
            padding_mask = "0"
            sliding_or_chunked_causal_mask = "0"
            custom_mask = "0"

        # enable_attn_logit_softcapping kernels only need causal mask or sliding_or_chunked_causal_mask.
        if kspec.enable_attn_logit_softcapping:
            padding_mask = "0"
            custom_mask = "0"

    return padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask


def enable_mutex(kspec: FMHAKernelSpec) -> str:
    fp32_accu_dtype = kspec.dtype in ["fp16_fp32", "bf16"]
    enable_mutex = "false" if (fp32_accu_dtype or kspec.head_size <= 64) else "true"
    return enable_mutex


def get_reg_count(kspec):
    # if kspec.paged_kv_input and kspec.dtype in ['fp16', 'fp16_fp32', 'bf16']:
    #     dma_reg_count = 72
    #     compute_reg_count = 216
    if kspec.input_layout == InputLayout.Q_PAGED_KV:
        dma_reg_count = 56
        compute_reg_count = 224
    else:
        dma_reg_count = 40
        compute_reg_count = 232
    return dma_reg_count, compute_reg_count


def enable_tma_store(kspec):
    # TMA copies data in the 16B granularity.
    return (
        "true"
        if (kspec.dtype in ["e4m3", "e4m3_fp32"] and kspec.head_size % 16 == 0)
        else "false"
    )
