#ifndef EEDI3VK_VULKAN_COMPUTE_PIPELINE_HPP
#define EEDI3VK_VULKAN_COMPUTE_PIPELINE_HPP

#define VK_NO_PROTOTYPES

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

#include <span>
#include <vulkan/vulkan_raii.hpp>

namespace eedi3vk {

class VulkanContext;

class VulkanComputePipeline {
  public:
    VulkanComputePipeline(
        VulkanContext& ctx, std::span<const uint32_t> spirv_code,
        std::span<const vk::DescriptorSetLayoutBinding> bindings,
        uint32_t push_constant_size = 0,
        const vk::SpecializationInfo* spec_info = nullptr);
    ~VulkanComputePipeline() = default;

    VulkanComputePipeline(const VulkanComputePipeline&) = delete;
    VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;
    VulkanComputePipeline(VulkanComputePipeline&&) = default;
    VulkanComputePipeline& operator=(VulkanComputePipeline&&) = delete;

    void dispatch(vk::raii::CommandBuffer& cmd,
                  vk::DescriptorSet descriptor_set, uint32_t group_count_x,
                  uint32_t group_count_y = 1, uint32_t group_count_z = 1,
                  const void* push_constants = nullptr,
                  uint32_t push_constants_size = 0);

    [[nodiscard]] vk::DescriptorSetLayout getDescriptorSetLayout() const {
        return *descriptor_set_layout;
    }
    [[nodiscard]] vk::PipelineLayout getPipelineLayout() const {
        return *pipeline_layout;
    }
    [[nodiscard]] vk::Pipeline getPipeline() const { return *pipeline; }

  private:
    VulkanContext& context;
    vk::raii::ShaderModule shader_module = nullptr;
    vk::raii::DescriptorSetLayout descriptor_set_layout = nullptr;
    vk::raii::PipelineLayout pipeline_layout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
    uint32_t push_constant_size = 0;
};

// Descriptor pool helper for allocating descriptor sets
class DescriptorPool {
  public:
    DescriptorPool(VulkanContext& ctx, uint32_t max_sets,
                   std::span<const vk::DescriptorPoolSize> pool_sizes);
    ~DescriptorPool() = default;

    vk::raii::DescriptorSet allocateSet(vk::DescriptorSetLayout layout);
    void reset();

  private:
    VulkanContext& context;
    vk::raii::DescriptorPool pool = nullptr;
};

} // namespace eedi3vk

#endif // EEDI3VK_VULKAN_COMPUTE_PIPELINE_HPP
