#include <string>
#include <stdexcept>

constexpr int kCudaMemAlign = 128;

/**
 * @brief TODO: Document properly.
 */
inline constexpr std::int32_t kMinHistoryTokensPerBlock = 128;

/**
 * @brief TODO: Document properly.
 */
inline constexpr std::int32_t kTargetWaveFactor = 8;

// For multi-block mode. We reserve workspace for this amount of sub-sequences.
// This should be enough. Huge batch size may result in larger value, but for
// large batch size, multi-block mode is not useful. For llama v2 70b, 6000
// results in ~12MB multi-block workspace, and is enough for > 10 waves.
inline constexpr std::int32_t kXQAMaxNumSubSequences = 6000;

enum DeviceDataType : uint8_t
{
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP4_E2M1,
    UINT32,
    UINT16,
    UINT8,
    INT32,
    INT16,
    INT8,
    PTR,
    BOOL,
    INT2,

    _SENTINEL_COUNT_  // NOLINT: Sentinel value to count the number of elements in the enum and ensure all enum values
                      // are handled in tests.
};

constexpr uint32_t kNumBitsInFloat32  = 32;
constexpr uint32_t kNumBitsInFloat16  = 16;
constexpr uint32_t kNumBitsInBF16     = 16;
constexpr uint32_t kNumBitsInFP8_E4M3 = 8;
constexpr uint32_t kNumBitsInFP8_E5M2 = 8;
constexpr uint32_t kNumBitsInFP4_E2M1 = 4;
constexpr uint32_t kNumBitsInUINT32   = 32;
constexpr uint32_t kNumBitsInUINT16   = 16;
constexpr uint32_t kNumBitsInUINT8    = 8;
constexpr uint32_t kNumBitsInINT32    = 32;
constexpr uint32_t kNumBitsInINT16    = 16;
constexpr uint32_t kNumBitsInINT8     = 8;
constexpr uint32_t kNumBitsInPTR      = 64;
constexpr uint32_t kNumBitsInBOOL     = 1;
constexpr uint32_t kNumBitsInINT2     = 64;

// Constexpr function to map torch dtype strings to DeviceDataType
constexpr DeviceDataType getTorchDeviceDataType(std::string torch_dtype) {
    // Floating point types
    if constexpr (torch_dtype == "torch.float32" || torch_dtype == "torch.float") {
        return DeviceDataType::FP32;
    }
    if constexpr (torch_dtype == "torch.float16" || torch_dtype == "torch.half") {
        return DeviceDataType::FP16;
    }
    if constexpr (torch_dtype == "torch.bfloat16") {
        return DeviceDataType::BF16;
    }
    if constexpr (torch_dtype == "torch.float8_e4m3fn") {
        return DeviceDataType::FP8_E4M3;
    }
    
    // Integer types
    if constexpr (torch_dtype == "torch.uint8") {
        return DeviceDataType::UINT8;
    }
    if constexpr (torch_dtype == "torch.int8") {
        return DeviceDataType::INT8;
    }
    if constexpr (torch_dtype == "torch.int16" || torch_dtype == "torch.short") {
        return DeviceDataType::INT16;
    }
    if constexpr (torch_dtype == "torch.int32" || torch_dtype == "torch.int") {
        return DeviceDataType::INT32;
    }
    
    // Boolean
    if constexpr (torch_dtype == "torch.bool") {
        return DeviceDataType::BOOL;
    }
    
    // Unsupported or invalid dtype
    throw std::invalid_argument("Unsupported torch dtype");
}

template <typename UIntT>
__host__ __device__ inline void* advancePtr(void* ptr, UIntT offset, DeviceDataType dtype)
{
    auto const deltaInBytes = getSizeInBytes(offset, dtype);
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + deltaInBytes);
}

/**
 * @brief A segment of workspace required internally by a kernel.
 * @note Usage:
 * - Implement an internal / private function which returns a collection of segments.
 * - Implement a function which uses above function to get the workspace size. The user of the kernel can then call this
 * function to know how much memory they have to allocate.
 * - In the kernel launcher, use the SAME function to get the segments and then use the getWorkspacePtrs function to get
 * the pointers to the workspace.
 * The goal is to avoid discrepancies between the advertised, required workspace size and the actual workspace size.
 */
struct WorkspaceSegmentDescriptor
{
    /**
     * @brief The type of elements contained in the segment.
     */
    DeviceDataType dataType;

    /**
     * @brief The number of elements contained in the segment.
     */
    std::size_t numElements;

    /**
     * @brief The alignment of the segment in bytes.
     */
    std::size_t alignment = extensions::cuda::kCudaMemAlign;
};

/**
 * @brief Rounds up a value to the nearest multiple of an alignment.
 *
 * @tparam TIntegral The type of the integers to round up.
 * @param value The value to round up.
 * @param alignment The alignment to round up to.
 * @return TIntegral The result of the rounding.
 */
template <typename TIntegral>
inline constexpr TIntegral roundUp(TIntegral const& value, TIntegral const& alignment)
{
    return divUp(value, alignment) * alignment;
}

/**
 * @brief Get the pointers to the workspace segments.
 * @tparam RangeT The type of the range of segments.
 * @param basePtr The base pointer to the workspace.
 * @param range The range of segments.
 * @return The pointers to the workspace segments, aligned with the input segments.
 */
template <std::ranges::input_range RangeT>
    requires std::same_as<std::ranges::range_value_t<RangeT>, WorkspaceSegmentDescriptor>
std::vector<void*> getWorkspacePtrs(void* basePtr, RangeT const& range)
{
    std::vector<void*> ptrs{};
    ptrs.reserve(range.size());
    for (auto const& segment : range)
    {
        basePtr = alignPtr(basePtr, segment.alignment);
        ptrs.push_back(basePtr);
        basePtr = extensions::cuda::advancePtr(basePtr, segment.numElements, segment.dataType);
    }
    return ptrs;
}