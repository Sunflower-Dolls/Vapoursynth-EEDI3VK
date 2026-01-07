#ifndef EEDI3VK_SHADERS_HPP
#define EEDI3VK_SHADERS_HPP

#include <cstdint>
#include <span>

// NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)

namespace eedi3vk::shaders {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

inline constexpr uint8_t COPY_FIELD_SPV[] = {
#embed "copy_field.spv"
};

inline constexpr uint8_t DILATE_MASK_SPV[] = {
#embed "dilate_mask.spv"
};

inline constexpr uint8_t CALC_COSTS_SPV[] = {
#embed "calc_costs.spv"
};

inline constexpr uint8_t VITERBI_SCAN_SPV[] = {
#embed "viterbi_scan.spv"
};

inline constexpr uint8_t INTERPOLATE_SPV[] = {
#embed "interpolate.spv"
};

inline constexpr uint8_t COPY_BUFFER_SPV[] = {
#embed "copy_buffer.spv"
};

#pragma clang diagnostic pop

inline std::span<const uint32_t> get_copy_field_spv() {
    return {reinterpret_cast<const uint32_t*>(COPY_FIELD_SPV),
            sizeof(COPY_FIELD_SPV) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> get_dilate_mask_spv() {
    return {reinterpret_cast<const uint32_t*>(DILATE_MASK_SPV),
            sizeof(DILATE_MASK_SPV) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> get_calc_costs_spv() {
    return {reinterpret_cast<const uint32_t*>(CALC_COSTS_SPV),
            sizeof(CALC_COSTS_SPV) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> get_viterbi_scan_spv() {
    return {reinterpret_cast<const uint32_t*>(VITERBI_SCAN_SPV),
            sizeof(VITERBI_SCAN_SPV) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> get_interpolate_spv() {
    return {reinterpret_cast<const uint32_t*>(INTERPOLATE_SPV),
            sizeof(INTERPOLATE_SPV) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> get_copy_buffer_spv() {
    return {reinterpret_cast<const uint32_t*>(COPY_BUFFER_SPV),
            sizeof(COPY_BUFFER_SPV) / sizeof(uint32_t)};
}

} // namespace eedi3vk::shaders

// NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)

#endif // EEDI3VK_SHADERS_HPP
