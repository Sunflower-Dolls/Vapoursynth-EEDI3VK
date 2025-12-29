#ifndef EEDI3VK_PIPELINES_HPP
#define EEDI3VK_PIPELINES_HPP

#include "VulkanComputePipeline.hpp"

#include <memory>

namespace eedi3vk {

class VulkanContext;

class EEDI3Pipelines {
  public:
    explicit EEDI3Pipelines(VulkanContext& ctx, int tpitch);
    ~EEDI3Pipelines() = default;

    EEDI3Pipelines(const EEDI3Pipelines&) = delete;
    EEDI3Pipelines& operator=(const EEDI3Pipelines&) = delete;
    EEDI3Pipelines(EEDI3Pipelines&&) = default;
    EEDI3Pipelines& operator=(EEDI3Pipelines&&) = delete;

    [[nodiscard]] VulkanComputePipeline& getCopyField() { return *copy_field; }
    [[nodiscard]] VulkanComputePipeline& getDilateMask() {
        return *dilate_mask;
    }
    [[nodiscard]] VulkanComputePipeline& getCalcCosts() { return *calc_costs; }
    [[nodiscard]] VulkanComputePipeline& getViterbiScan() {
        return *viterbi_scan;
    }
    [[nodiscard]] VulkanComputePipeline& getInterpolate() {
        return *interpolate;
    }
    [[nodiscard]] VulkanComputePipeline& getCopyBuffer() {
        return *copy_buffer;
    }

    vk::raii::DescriptorSet allocateCopyFieldSet(DescriptorPool& pool);
    vk::raii::DescriptorSet allocateDilateMaskSet(DescriptorPool& pool);
    vk::raii::DescriptorSet allocateCalcCostsSet(DescriptorPool& pool);
    vk::raii::DescriptorSet allocateViterbiScanSet(DescriptorPool& pool);
    vk::raii::DescriptorSet allocateInterpolateSet(DescriptorPool& pool);
    vk::raii::DescriptorSet allocateCopyBufferSet(DescriptorPool& pool);

  private:
    VulkanContext& context;

    std::unique_ptr<VulkanComputePipeline> copy_field;
    std::unique_ptr<VulkanComputePipeline> dilate_mask;
    std::unique_ptr<VulkanComputePipeline> calc_costs;
    std::unique_ptr<VulkanComputePipeline> viterbi_scan;
    std::unique_ptr<VulkanComputePipeline> interpolate;
    std::unique_ptr<VulkanComputePipeline> copy_buffer;
};

} // namespace eedi3vk

#endif // EEDI3VK_PIPELINES_HPP
