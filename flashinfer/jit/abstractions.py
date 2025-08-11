from typing import List, Optional
import dataclasses

from pathlib import Path

from . import env as jit_env


@dataclasses.dataclass
class JitSpec:
    """
    JIT compilation specification for a FlashInfer operation.

    Attributes:
    name: A unique string identifying the operation.
    sources: A list of source files for the JIT operation.
    extra_cflags: Additional compiler flags for the JIT operation.
    """

    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]
    is_class: bool = False
    needs_device_linking: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"

    def get_library_path(self) -> Path:
        if self.aot_path.exists():
            return self.aot_path
        return self.jit_library_path

    @property
    def aot_path(self) -> Path:
        return jit_env.FLASHINFER_AOT_DIR / self.name / f"{self.name}.so"
