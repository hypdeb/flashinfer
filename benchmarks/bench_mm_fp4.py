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

import itertools
from typing import Dict
import numpy as np
import torch

from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

_cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}


def bench_mm_fp4(m: int, n: int, k: int, fp4_type: str, res_dtype: torch.dtype, use_128x4_sf_layout: bool, backend: str):
    torch.manual_seed(123)

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "trtllm":
        if res_dtype == torch.float16:
            print("Skipping test for trtllm fp4 with float16")
            return
        if compute_capability[0] in [11, 12]:
            print("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
            return
    if not use_128x4_sf_layout and backend != "trtllm":
        print("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
        return
    if backend == "cudnn":
        print("Skipping test for cudnn fp4 with auto_tuning=True")
        return
    if not use_nvfp4 and backend != "cudnn":
        print("mx_fp4 is only supported for cudnn backend")
        return

    if fp4_type == "nvfp4":
        use_nvfp4 = True
    else:
        use_nvfp4 = False

    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4
    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()
    do_shuffle_b = backend == "trtllm"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)


    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"
    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None
    res = torch.empty([m, n], device="cuda", dtype=res_dtype)
    block_size = 16 if use_nvfp4 else 32
    with autotune(True):
        mm_fp4(
            input_fp4,
            mat2_fp4.T,
            input_inv_s,
            mat2_inv_s.T,
            alpha,
            res_dtype,
            res,
            block_size=block_size,
            use_8x4_sf_layout=not use_128x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
        )

    measurements = bench_gpu_time(
        lambda: mm_fp4(
            input_fp4,
            mat2_fp4.T,
            input_inv_s,
            mat2_inv_s.T,
            alpha,
            res_dtype,
            res,
            block_size=block_size,
            use_8x4_sf_layout=not use_128x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
        ),
        dry_run_time_ms=500,
        repeat_time_ms=2500,
        use_cuda_graph=True,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms

    bandwidth = (
        (
            input_fp4.numel() * input_fp4.element_size()
            + mat2_fp4.numel() * mat2_fp4.element_size()
            + res.numel() * res.element_size()
        )
        / ms
        / 1e9
    )

    print(
        f"mm_fp4 m={m} n={n} k={k} fp4_type={fp4_type} res_dtype={res_dtype} use_128x4_sf_layout={use_128x4_sf_layout} backend={backend}: {tflops_per_second:.2f} TFLOPs/s over {ms:.6f} ms, {bandwidth:.2f} TB/s"
    )


if __name__ == "__main__":
    ms = [1, 2, 4, 8, 16, 32, 64]
    ns = [2560, 5120, 8192]
    ks = [16384, 32768]
    fp4_types = ["nvfp4", "mxfp4_alpha"]
    use_128x4_sf_layouts = [True, False]
    backends = ["trtllm", "cutlass"]
    for m, n, k, fp4_type, use_128x4_sf_layout, backend in itertools.product(ms, ns, ks, fp4_types, use_128x4_sf_layouts, backends):
        bench_mm_fp4(m, n, k, fp4_type, torch.bfloat16, use_128x4_sf_layout, backend)


