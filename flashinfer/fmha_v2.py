import os
import subprocess
import shutil
from flashinfer.jit import JitSpec
from flashinfer.jit import env as jit_env


def gen_fmha_v2_module() -> JitSpec:
    # First attempt: reproduce what the makefile is doing as best we can.
    # FIXME: brittle at best.
    shutil.rmtree(jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2", ignore_errors=True)

    # Create the required directories.
    os.mkdir(jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2")
    os.mkdir(jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2" / "generated")
    os.mkdir(jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2" / "temp")
    os.mkdir(jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2" / "cubin")

    shutil.copytree(
        jit_env.FLASHINFER_CSRC_DIR / "fmha_v2" / "src",
        jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2" / "src",
        dirs_exist_ok=True,
    )
    subprocess.run(
        [
            "python",
            jit_env.FLASHINFER_CSRC_DIR / "setup.py",
            "--output-dir",
            jit_env.FLASHINFER_GEN_SRC_DIR / "fmha_v2" / "generated",
        ]
    )
