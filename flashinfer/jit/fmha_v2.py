import os
import subprocess
import shutil
from flashinfer.jit.core import logger
from flashinfer.jit import JitSpec, gen_jit_spec
from flashinfer.jit import env as jit_env


def gen_fmha_v2_module() -> JitSpec:
    # First attempt: reproduce what the makefile is doing as best we can.
    # FIXME: brittle at best.
    logger.info("Generating fmha_v2 module")
    logger.info("clearing fmha_v2 gen dir")
    fmha_v2_gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2"
    fmha_v2_src_dir = jit_env.FLASHINFER_CSRC_DIR / "fmha_v2"
    shutil.rmtree(fmha_v2_gen_dir, ignore_errors=True)

    # Create the required directories.
    os.makedirs(fmha_v2_gen_dir, exist_ok=True)
    os.mkdir(fmha_v2_gen_dir / "generated")
    os.mkdir(fmha_v2_gen_dir / "temp")
    os.mkdir(fmha_v2_gen_dir / "cubin")

    # setup.py needs those there I think.
    logger.info("copying src directory")
    shutil.copytree(
        fmha_v2_src_dir / "src",
        fmha_v2_gen_dir / "src",
        dirs_exist_ok=True,
    )
    logger.info("running setup.py")
    subprocess.run(
        [
            "python",
            fmha_v2_src_dir / "setup.py",
            "--output_dir",
            fmha_v2_gen_dir / "generated",
        ]
    )

    # Remove the superfluous print_kernel_traits.cu file.
    logger.info("removing print_kernel_traits.cu file")
    shutil.rmtree(fmha_v2_gen_dir / "generated" / "print_kernel_traits.cu")

    # Gather all the .cu files in the src directory recursively.
    fmha_v2_cuda_files = [
        f for f in (fmha_v2_gen_dir / "src").glob("**/*.cu") if f.is_file()
    ]

    # Gather all the .cu files in the generated directory.
    kernel_cuda_files = [
        f for f in (fmha_v2_gen_dir / "generated").glob("*.cu") if f.is_file()
    ]

    logger.info("generating jit spec")
    return gen_jit_spec(
        "fmha_v2",
        [
            *fmha_v2_cuda_files,
            *kernel_cuda_files,
            jit_env.FLASHINFER_CSRC_DIR / "fmha_v2_runner.cu",
        ],
        extra_include_paths=[fmha_v2_gen_dir / "src", fmha_v2_gen_dir / "generated"],
    )
