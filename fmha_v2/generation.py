from dataclasses import asdict
from typing import Optional
from fmha_v2.constants import (
    MAX_STGS_PER_LOOP,
    dtype2traits,
    hopper_dtype2traits,
    dtype2OutputType,
    dtype2bytes,
)
from fmha_v2.utils import (
    enable_mutex,
    enable_tma_store,
    get_effective_sm_and_name,
    get_hopper_instruction_traits,
    get_reg_count,
    selected_mask_types,
    pythonBoolean2cpp,
)
from fmha_v2.abstractions import FMHAKernelSpec, InputLayout

from jinja2 import Environment, PackageLoader

JinjaEnv = Environment(loader=PackageLoader("fmha_v2"))


def generate(fmha_kernel_spec: FMHAKernelSpec, lname: str, kname: str) -> Optional[str]:
    """
    In this function we build the templating parameters for the kernel templates.
    """

    min_cuda_version = 0  # no restriction

    # The architecture that determines the instruction.
    effective_sm, sm_name = get_effective_sm_and_name(fmha_kernel_spec)

    if effective_sm >= 80:
        min_cuda_version = 11000

    launcher_name = lname
    causal_kernel_name = kname.replace("__placeholder__", "_causal")
    custom_mask_kernel_name = kname.replace("__placeholder__", "_custom_mask")

    sliding_or_chunked_causal_kernel_name = kname.replace(
        "__placeholder__", "_sliding_or_chunked_causal"
    )
    kernel_name = kname.replace("__placeholder__", "")

    # FIXME: use separate parameters when generating cubins for trtllm.
    if not fmha_kernel_spec.cross_mha:
        params_type = "bert::Fused_multihead_attention_params_v{}".format(
            fmha_kernel_spec.version
        )
    else:
        params_type = "bert::Fused_multihead_attention_params_mhca"

    if effective_sm < 90:
        instruction_traits = (
            sm_name.capitalize() + "_" + dtype2traits[fmha_kernel_spec.dtype]
        )
    elif effective_sm == 90:
        instruction_traits = (
            sm_name.capitalize() + "_" + hopper_dtype2traits[fmha_kernel_spec.dtype]
        )
        # for hopper, we differentiate instruction_traits_o and instruction_traits_p
        instruction_traits_p, instruction_traits_o = get_hopper_instruction_traits(
            instruction_traits, fmha_kernel_spec
        )
    else:
        instruction_traits = ""  # TODO: verify if correct

    if effective_sm < 90:
        if fmha_kernel_spec.flash_attention:
            kernel_variant = "flash_attention"
        else:
            kernel_variant = "1xN" if fmha_kernel_spec.warps_m == 1 else "2x2"
    elif effective_sm == 90:
        if fmha_kernel_spec.warps_n > 1:
            # for hopper we slice the problem along the M dim.
            kernel_variant = "4xN" + "_hopper"
        else:
            kernel_variant = "4x1" + "_hopper"

    if effective_sm < 90:
        kernel_traits = "Kernel_traits_"
    elif effective_sm == 90:
        kernel_traits = "FMHA_kernel_traits_hopper_"
    else:
        kernel_traits = ""  # TODO: not sure this is correct.

    if fmha_kernel_spec.interleaved:
        kernel_traits += "interleaved_v2"
    elif fmha_kernel_spec.cross_mha:
        kernel_traits += "fmhca"
    else:
        kernel_traits += "v{}".format(fmha_kernel_spec.version)

    # decide whether to paged_kv kernel traits for ampere-style kernels.
    if effective_sm < 90:
        if fmha_kernel_spec.input_layout == InputLayout.Q_PAGED_KV:
            kernel_traits += "_paged_kv_cache"
        elif fmha_kernel_spec.input_layout == InputLayout.CONTIGUOUS_Q_KV:
            kernel_traits += "_contiguous_kv_cache"

    flags = 0
    if fmha_kernel_spec.ldgsts_q:
        flags |= 1
    if fmha_kernel_spec.ldgsts_k:
        flags |= 2
    if fmha_kernel_spec.ldgsts_v:
        flags |= 4
    if fmha_kernel_spec.share_smem_k_v and not fmha_kernel_spec.limit_qk_fragments:
        flags |= 8
    if fmha_kernel_spec.has_scale_max:
        flags |= 16
    if not fmha_kernel_spec.head_interleaved:
        flags |= 32
    if fmha_kernel_spec.limit_qk_fragments:
        flags |= 128
    if fmha_kernel_spec.limit_v_fragments:
        flags |= 256
    if fmha_kernel_spec.has_noloop:
        # NOTE do not use flags 512 = 0x200 as it is reserved; do not add to flags because it
        # will be selectively added to no-loop kernel trait upon generating .cu templates
        pass
    if fmha_kernel_spec.enable_attn_logit_softcapping:
        flags |= 2048
    if fmha_kernel_spec.tiled:
        flags |= 4096
    if fmha_kernel_spec.is_mtp:
        flags |= 8192

    # only generate certain needed combinations of input_layout and mask types for trt-llm.
    mask_types = selected_mask_types(fmha_kernel_spec)
    if all([mt == "0" for mt in mask_types]):
        # If no mask type is select, then we do not generate any kernel.
        return None

    padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask = mask_types
    kernel_flags = "0x{:02x}u".format(flags)

    heads_interleaved_flag = pythonBoolean2cpp[fmha_kernel_spec.head_interleaved]

    disable_fadd_trick = (
        1 if effective_sm >= 86 else 0
    )  # this will force generating F2IP

    enable_mutex_flag = enable_mutex(fmha_kernel_spec)

    has_alibi = pythonBoolean2cpp[fmha_kernel_spec.alibi]

    input_layout_flag = str(int(fmha_kernel_spec.input_layout))

    run_fct_name = (
        "run_packed_qkv"
        if fmha_kernel_spec.input_layout == InputLayout.PACKED_QKV
        else "run_separate_q_and_kv"
    )

    dma_reg_count, compute_reg_count = get_reg_count(fmha_kernel_spec)

    use_tma_store_flag = enable_tma_store(fmha_kernel_spec)

    enable_attn_logit_softcapping_flag = pythonBoolean2cpp[
        fmha_kernel_spec.enable_attn_logit_softcapping
    ]

    return_softmax_stats_flag = pythonBoolean2cpp[fmha_kernel_spec.return_softmax_stats]

    # needed by warpspec kernels.
    fp8_kernel = fmha_kernel_spec.dtype in ["e4m3", "e4m3_fp32"]
    kernel_traits_header = (
        "fmha::ws::Kernel_traits_Hopper_qgmma_e4m3_fp32<"
        if fp8_kernel
        else f"fmha::ws::Kernel_traits<fmha::{instruction_traits},"
    )

    # output type.
    output_dtype_ = f"fmha::{dtype2OutputType[fmha_kernel_spec.output_dtype if fmha_kernel_spec.output_dtype is not None else fmha_kernel_spec.dtype]}"

    # sage attention block sizes.
    sage_block_size_q = 0
    sage_block_size_k = 0
    sage_block_size_v = 0
    if fp8_kernel and fmha_kernel_spec.sage_block_sizes:
        assert fmha_kernel_spec.output_dtype is not None, (
            "output_dtype must be specified for fp8 sage attention kernels"
        )
        sage_block_size_q = fmha_kernel_spec.sage_block_sizes[0]
        sage_block_size_k = fmha_kernel_spec.sage_block_sizes[1]
        sage_block_size_v = fmha_kernel_spec.sage_block_sizes[2]

    TMA_config = r"""
    // TMA configuration
    // Note that this may only need to init once during inference (for different layers)
    // Reuse the same traits for initializing tma descriptors.
    fmha::ws::DMA<Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, stream);
    """
    params_str = "params"
    attn_mask_type_str = "using Attention_mask_type = fmha::Attention_mask_type;"
    bert_launch_params = (
        "using Launch_params = bert::Fused_multihead_attention_launch_params;"
    )
    include_str = ""
    num_compute_groups_str = "static constexpr int NUM_COMPUTE_GROUPS = 2;"
    fused_multihead_attention_params_v2_str = f"{params_type}"
    const_fused_multihead_attention_params_v2_str = f"const {params_type}"
    setmaxnreg_dma_str = r"""
        const int DMA_REG_COUNT = {dma_reg_count};
        asm volatile("{{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}}" ::"n"(DMA_REG_COUNT));""".format(
        dma_reg_count=dma_reg_count
    )
    setmaxnreg_compute_str = r"""
        const int COMPUTE_REG_COUNT = {compute_reg_count};
        asm volatile("{{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}}" ::"n"(COMPUTE_REG_COUNT));""".format(
        compute_reg_count=compute_reg_count
    )
    local_ns_open = ""
    local_ns_close = ""

    tmp = dict(locals(), **asdict(fmha_kernel_spec))

    if effective_sm < 90:
        if fmha_kernel_spec.flash_attention:
            template = JinjaEnv.get_template("kernel_fa.c.jinja")
            tmp["MAX_STGS_PER_LOOP"] = MAX_STGS_PER_LOOP
            tmp["use_multi_cta"] = False
            code = template.render(tmp)
        else:
            template = JinjaEnv.get_template("kernel.c.jinja")
            tmp["MAX_STGS_PER_LOOP"] = MAX_STGS_PER_LOOP
            use_multi_cta = 1 if fmha_kernel_spec.ctas_per_head > 1 else 0
            tmp["use_multi_cta"] = use_multi_cta
            code = template.render(tmp)
    elif effective_sm == 90:
        use_tma = 1
        if fmha_kernel_spec.ldgsts_q:
            use_tma = 0
        if fmha_kernel_spec.warp_specialization:
            template = JinjaEnv.get_template("kernel_hopper_ws.c.jinja")
            tmp["use_tma"] = use_tma
            tmp["bytes_per_elt"] = dtype2bytes[fmha_kernel_spec.dtype]
            code = template.render(tmp)
        else:
            template = JinjaEnv.get_template("kernel_hopper.c.jinja")
            tmp["use_tma"] = use_tma
            code = template.render(tmp)
    else:
        raise RuntimeError("No template found for this configuration.")
    return code
