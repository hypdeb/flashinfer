"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace
from typing import List

import functools

import torch

from flashinfer.artifacts import ArtifactPath, MetaInfoHash
from flashinfer.autotuner import (
    AutoTuner,
    TuningConfig,
    DynamicTensorSpec,
    ConstraintSpec,
    TunableRunner,
    OptimizationProfile,
)
from flashinfer.fused_moe.utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from flashinfer.gemm import DEFAULT_WORKSPACE_SIZE
from flashinfer.jit import setup_cubin_loader, JitSpec, gen_jit_spec, sm100a_nvcc_flags
from flashinfer.jit import env as jit_env
from flashinfer.jit.cubin_loader import get_cubin
from flashinfer.utils import _get_cache_buf


def gen_trtllm_gen_flavored_gemm_module() -> JitSpec:
    include_path = f"{ArtifactPath.TRTLLM_GEN_GEMM}/include"
    header_name = "flashinferMetaInfo"

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}",
        MetaInfoHash.TRTLLM_GEN_GEMM,
        ".h",
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"
    return gen_jit_spec(
        "trtllm_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_flavored_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_ENABLE_CUDA",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_GEMM}\\"',
        ]
        + sm100a_nvcc_flags,
        # link "include" sub-directory in cache
        extra_include_paths=[jit_env.FLASHINFER_CUBIN_DIR / include_path],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def get_trtllm_flavored_gemm_module():
    mod = gen_trtllm_gen_flavored_gemm_module()
    op = mod.build_and_load()
    setup_cubin_loader(str(mod.get_library_path()))

    class TrtllmFlavoredGemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a_tensor_index = 0
            b_tensor_index = 1

            # NOTE : expects  A=MxK, B=(K//B)xNxB, out=MxN
            a = profile.get_opt_shapes()[a_tensor_index]
            b = profile.get_opt_shapes()[b_tensor_index]
            m = a[0]
            n = b[1]
            k = a[1]
            (
                a,
                b,
                global_scale,
                out,
                workspace_buffer,
            ) = inputs
            type_e4m3 = 1
            type_bf16 = 2
            valid_tactics = list(
                op.trtllm_flavored_gemm_tactics(m, n, k, type_e4m3, type_bf16)
            )
            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            (
                a,
                b,
                global_scale,
                out,
                workspace_buffer,
            ) = inputs
            if tactic < 0:
                return out
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]
            workspace_size = op.get_workspace_size_in_bytes(m, n, k, tactic)
            workspace_buffer = _get_cache_buf(
                "trllm_flavored_gemm", workspace_size, a.device
            )
            op.trtllm_flavored_gemm(
                workspace_buffer,
                a,
                b,
                global_scale,
                out,
                tactic,
            )
            return out

    def gemm_runner():
        return TrtllmFlavoredGemmRunner()

    # Register the module
    return SimpleNamespace(
        gemm_runner=gemm_runner,
    )


def trtllm_flavored_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    global_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    r"""GEMM optimized for low M dimension. The optimizations require special pre-processing for the input tensor B.
    For this reason, this implementation is only recommended for scenarios where B can be pre-processed offline, once. Luckily, this
    is often the case in inference, where B is a weight matrix that can be massaged on-load.
    - First B has to be shuffled.
    - Then, its layout has to be adjusted as blockwise row-major.
      The blocksize is 128 for e4m3, and therefore, the K dimension must be divisible by 128.
    Utilities to perform these operations are implemented in Flashinfer, and their usage is demonstrated in the tests for this operation.

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (m, k), fp8 e4m3.

    B: torch.Tensor
        Mat2 tensor, shape (n, k), fp8 e4m3.

    global_scale: torch.Tensor
        Scale tensor for the output, float.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), bf16.

    backend: Literal["trtllm", "auto"]
        The backend to use for the operation. Defaults to ``"trtllm"``, as it is the only supported backend currently.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (m, n), bf16, will be a reference to the output parameter, as only inplace is supported.

    See tests/test_fp8_gemm.py for usage examples.
    """

    workspace_buffer = _get_cache_buf(
        "gemm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, A.device
    )

    tuner = AutoTuner.get()
    a_tensor_index = 0
    out_tensor_index = 3
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (a_tensor_index,),
                (-2,),
                get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2,
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                out_tensor_index, -2, lambda shapes: shapes[a_tensor_index][-2]
            ),
        ),
    )
    inputs = [A, B, global_scale, out, workspace_buffer]
    runners: List[TunableRunner] = []
    runners.append(get_trtllm_flavored_gemm_module().gemm_runner())
    runner, tactic = tuner.choose_one(
        "trtllm_flavored_gemm",
        runners,
        tuning_config,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)
    return out
