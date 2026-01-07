#ifndef EEDI3VK_PARAMS_HPP
#define EEDI3VK_PARAMS_HPP

#include <cstdint>

namespace eedi3vk {

// calc_costs.comp
struct CalcCostsParams {
    int32_t width;
    int32_t height;
    int32_t mdis;
    int32_t tpitch;
    float alpha;
    float beta;
    float gamma;
    float remaining_weight;
    int32_t cost3;
    int32_t ucubic;
    int32_t has_mclip;
    int32_t nrad;
    int32_t field;
    int32_t stride;
    int32_t dh;
};

// viterbi_scan.comp
struct ViterbiScanParams {
    int32_t width;
    int32_t height;
    int32_t mdis;
    int32_t tpitch;
    float gamma;
    int32_t has_mclip;
};

// interpolate.comp
struct InterpolateParams {
    int32_t width;
    int32_t height;
    int32_t mdis;
    int32_t ucubic;
    int32_t has_mclip;
    int32_t field;
    int32_t stride;
    int32_t dh;
};

// copy_field.comp
struct CopyFieldParams {
    int32_t width;
    int32_t height;
    int32_t field;
    int32_t stride;
    int32_t dh;
};

// dilate_mask.comp
struct DilateMaskParams {
    int32_t width;
    int32_t height;
    int32_t mdis;
    int32_t field;
    int32_t stride;
    int32_t dh;
    float mclip_offset;
};

// copy_buffer.comp
struct CopyBufferParams {
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t width_bytes;
};

} // namespace eedi3vk

#endif // EEDI3VK_PARAMS_HPP
