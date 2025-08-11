import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from flashinfer.jit.abstractions import JitSpec

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.FLASHINFER_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.FLASHINFER_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.FLASHINFER_WORKSPACE_DIR / "flashinfer_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("flashinfer.jit: " + msg)


logger = FlashInferJITLogger("flashinfer.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        compute_arch_match = re.search(r"compute_(\d+)", cuda_arch_flags)
        if compute_arch_match is None:
            raise RuntimeError(
                f"No compute arch flag found in torch CUDA arch flags: {cuda_arch_flags}."
            )
        try:
            arch = int(compute_arch_match.group(1))
        except ValueError as int_conversion_exception:
            raise RuntimeError(
                f"Invalid compute arch flag ({compute_arch_match.group(0)}) found in torch CUDA arch flags: {cuda_arch_flags}."
            ) from int_conversion_exception
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


def clear_cache_dir():
    if os.path.exists(jit_env.FLASHINFER_JIT_DIR):
        import shutil

        shutil.rmtree(jit_env.FLASHINFER_JIT_DIR)


sm90a_nvcc_flags = ["-gencode=arch=compute_90a,code=sm_90a"]
sm100a_nvcc_flags = [
    "-gencode=arch=compute_100a,code=sm_100a",
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
]


def write_ninja(spec: JitSpec) -> None:
    spec.ninja_path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_ninja_build_for_op(spec)
    write_if_different(spec.ninja_path, content)


def build(spec: JitSpec, verbose: bool) -> None:
    tmpdir = get_tmpdir()
    with FileLock(tmpdir / f"{spec.name}.lock", thread_local=False):
        run_ninja(jit_env.FLASHINFER_JIT_DIR, spec.ninja_path, verbose)


def build_and_load(spec: JitSpec, class_name: Optional[str] = None):
    if spec.aot_path.exists():
        so_path = spec.aot_path
    else:
        so_path = spec.jit_library_path
        verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
        build(spec, verbose)
    load_class = class_name is not None
    loader = torch.classes if load_class else torch.ops
    loader.load_library(str(so_path))
    if load_class:
        cls = torch._C._get_custom_class_python_wrapper(spec.name, class_name)
        return cls
    return getattr(loader, spec.name)


def gen_jit_spec(
    name: str,
    sources: Sequence[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    check_cuda_arch()
    verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        f"--threads={min(os.cpu_count() or 4, 32)}",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    if verbose:
        cuda_cflags += [
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    spec = JitSpec(
        name=name,
        sources=[Path(x) for x in sources],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=(
            [Path(x) for x in extra_include_paths]
            if extra_include_paths is not None
            else None
        ),
        needs_device_linking=needs_device_linking,
    )
    write_ninja(spec)
    return spec


def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.FLASHINFER_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def build_jit_specs(
    specs: Iterable[JitSpec],
    verbose: bool = False,
    skip_prebuilt: bool = True,
) -> None:
    """
    JIT compiles the operations defined by the @param specs

    @param specs: A collection of operation specifications to compile.
    @param verbose: whether or not to build with verbose output for the compilation itself and the runtime. Also builds with line and debug info and without optimizations.
    @param skip_prebuilt: whether or not to skip building operations when AOT (Ahead-Of-Time) compiled versions are available.
    """
    lines: List[str] = []
    for spec in specs:
        if skip_prebuilt and spec.aot_path.exists():
            continue
        lines.append(f"subninja {spec.ninja_path}")
    if not lines:
        return

    lines = ["ninja_required_version = 1.3"] + lines + [""]

    tmpdir = get_tmpdir()
    with FileLock(tmpdir / "flashinfer_jit.lock", thread_local=False):
        ninja_path = tmpdir / "flashinfer_jit.ninja"
        write_if_different(ninja_path, "\n".join(lines))
        run_ninja(jit_env.FLASHINFER_JIT_DIR, ninja_path, verbose)
