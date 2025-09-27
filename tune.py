import torch

from flashinfer import bmm_fp8 
import flashinfer

def tune_fp8_gemm(max_m: int, n: int, k: int) -> None:
    a = torch.rand(1, max_m, k, device="cuda:0").to(torch.float8_e4m3fn).contiguous()
    b = torch.rand(1, k, n, device="cuda:0").to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.ones(1, device="cuda:0").contiguous()
    scale_b = torch.ones(1, device="cuda:0").contiguous()
    out = torch.ones(1, max_m, n, device="cuda:0").to(torch.bfloat16).contiguous()
    runner_names = "cudnn"
    with flashinfer.autotune():
        bmm_fp8(a, b, scale_a, scale_b, torch.bfloat16, out, runner_names)

def tune_fp8_gemm_swap_ab(max_m: int, n: int, k: int) -> None:
    a = torch.rand(1, n, k, device="cuda:0").to(torch.float8_e4m3fn).contiguous()
    b = torch.rand(1, k, max_m, device="cuda:0").to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.ones(1, device="cuda:0").contiguous()
    scale_b = torch.ones(1, device="cuda:0").contiguous()
    out = torch.ones(1, n, max_m, device="cuda:0").to(torch.bfloat16).contiguous()
    runner_names = "cudnn"
    with flashinfer.autotune():
        bmm_fp8(a, b, scale_a, scale_b, torch.bfloat16, out, runner_names)

if __name__ == '__main__':
    max_m = 128
    n = 5120
    k = 32768
    tune_fp8_gemm(max_m, n, k)
    tune_fp8_gemm_swap_ab(4, 8192, k)
