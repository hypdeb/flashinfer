from typing import Optional

from fmha_v2.abstractions import *
from fmha_v2.constants import *

from jinja2 import Environment, PackageLoader

JinjaEnv = Environment(loader=PackageLoader("fmha_v2"))


def generate(fmha_kernel_spec: Optional[FMHAKernelSpec] = None) -> str:
    templates = JinjaEnv.list_templates()
    print(f"Available templates: {templates}")
    return ""
