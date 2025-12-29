#ifndef EEDI3VK_SHADERS_HPP
#define EEDI3VK_SHADERS_HPP

#include <cstdint>
#include <span>

// NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)

namespace eedi3vk::shaders {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

inline constexpr uint8_t copy_field_spv[] = {
#embed "copy_field.spv"
};

inline constexpr uint8_t dilate_mask_spv[] = {
#embed "dilate_mask.spv"
};

inline constexpr uint8_t calc_costs_spv[] = {
#embed "calc_costs.spv"
};

inline constexpr uint8_t viterbi_scan_spv[] = {
#embed "viterbi_scan.spv"
};

inline constexpr uint8_t interpolate_spv[] = {
#embed "interpolate.spv"
};

inline constexpr uint8_t copy_buffer_spv[] = {
#embed "copy_buffer.spv"
};

#pragma clang diagnostic pop

inline std::span<const uint32_t> getCopyFieldSpv() {
    return {reinterpret_cast<const uint32_t*>(copy_field_spv),
            sizeof(copy_field_spv) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> getDilateMaskSpv() {
    return {reinterpret_cast<const uint32_t*>(dilate_mask_spv),
            sizeof(dilate_mask_spv) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> getCalcCostsSpv() {
    return {reinterpret_cast<const uint32_t*>(calc_costs_spv),
            sizeof(calc_costs_spv) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> getViterbiScanSpv() {
    return {reinterpret_cast<const uint32_t*>(viterbi_scan_spv),
            sizeof(viterbi_scan_spv) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> getInterpolateSpv() {
    return {reinterpret_cast<const uint32_t*>(interpolate_spv),
            sizeof(interpolate_spv) / sizeof(uint32_t)};
}

inline std::span<const uint32_t> getCopyBufferSpv() {
    return {reinterpret_cast<const uint32_t*>(copy_buffer_spv),
            sizeof(copy_buffer_spv) / sizeof(uint32_t)};
}

} // namespace eedi3vk::shaders

// NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)

#endif // EEDI3VK_SHADERS_HPP
