sm2name = {
    70: "volta",
    72: "volta",
    75: "turing",
    80: "ampere",
    86: "ampere",
    87: "ampere",
    89: "ada",
    90: "hopper",
    120: "blackwell",
}

dtype2traits = {
    "int8": "imma_int8_int32_traits",
    "fp16": "hmma_fp16_traits",
    "fp16_fp32": "hmma_fp32_traits",
    "bf16": "hmma_bf16_traits",
    "e4m3": "qmma_e4m3_fp32_traits",
    "e4m3_fp32": "qmma_e4m3_fp32_traits",
    "e4m3_fp16": "qmma_e4m3_fp16_traits",
}

dtype2OutputType = {
    "int8": "int8_t",
    "fp16": "fp16_t",
    "fp16_fp32": "fp16_t",
    "bf16": "bf16_t",
    "e4m3": "e4m3_t",
    "e4m3_fp32": "e4m3_t",
    "e4m3_fp16": "e4m3_t",
}

dtype2bytes = {
    "int8": 1,
    "fp16": 2,
    "fp16_fp32": 2,
    "bf16": 2,
    "e4m3": 1,
    "e4m3_fp32": 1,
    "e4m3_fp16": 1,
}

# TODO merge with above?
hopper_dtype2traits = {
    "int8": "igmma_int8_int32_traits",
    "fp16": "hgmma_fp16_traits",
    "fp16_fp32": "hgmma_fp32_traits",
    "bf16": "hgmma_bf16_traits",
    "e4m3": "qgmma_e4m3_fp32_traits",
    "e4m3_fp32": "qgmma_e4m3_fp32_traits",
}

# The minimal instruction shapes per warp group.
# TODO should this not be known to the trait itself?
hopper_traits2shape = {
    "Hopper_igmma_int8_int32_traits": (64, 8, 32),
    "Hopper_hgmma_fp16_traits": (64, 8, 16),
    "Hopper_hgmma_fp32_traits": (64, 8, 16),
    "Hopper_hgmma_bf16_traits": (64, 8, 16),
    "Hopper_qgmma_e4m3_fp32_traits": (64, 8, 32),
}

dtype2typename = {
    "int8": "DATA_TYPE_INT8",
    "fp16": "DATA_TYPE_FP16",
    "fp16_fp32": "DATA_TYPE_FP16",
    "bf16": "DATA_TYPE_BF16",
    "e4m3": "DATA_TYPE_E4M3",
    "e4m3_fp16": "DATA_TYPE_E4M3",
    "e4m3_fp32": "DATA_TYPE_E4M3",
}

MAX_STGS_PER_LOOP = 4
