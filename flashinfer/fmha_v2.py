from flashinfer.jit.fmha_v2 import gen_fmha_v2_module


def fmha_v2():
    module = gen_fmha_v2_module().build_and_load()
    module.fmha_v2()
