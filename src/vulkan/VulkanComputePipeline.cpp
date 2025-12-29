#include "VulkanComputePipeline.hpp"
#include "VulkanContext.hpp"

namespace eedi3vk {

VulkanComputePipeline::VulkanComputePipeline(
    VulkanContext& ctx, std::span<const uint32_t> spirv_code,
    std::span<const vk::DescriptorSetLayoutBinding> bindings,
    uint32_t push_constant_size, const vk::SpecializationInfo* spec_info)
    : context(ctx), push_constant_size(push_constant_size) {

    vk::ShaderModuleCreateInfo shader_info{
        {}, spirv_code.size() * sizeof(uint32_t), spirv_code.data()};
    shader_module = vk::raii::ShaderModule(context.getDevice(), shader_info);

    vk::DescriptorSetLayoutCreateInfo layout_info{
        {}, static_cast<uint32_t>(bindings.size()), bindings.data()};
    descriptor_set_layout =
        vk::raii::DescriptorSetLayout(context.getDevice(), layout_info);

    std::vector<vk::PushConstantRange> push_ranges;
    if (push_constant_size > 0) {
        push_ranges.emplace_back(vk::ShaderStageFlagBits::eCompute, 0,
                                 push_constant_size);
    }

    vk::PipelineLayoutCreateInfo pipeline_layout_info{
        {},
        1,
        &*descriptor_set_layout,
        static_cast<uint32_t>(push_ranges.size()),
        push_ranges.data()};
    pipeline_layout =
        vk::raii::PipelineLayout(context.getDevice(), pipeline_layout_info);

    vk::PipelineShaderStageCreateInfo stage_info{
        {},
        vk::ShaderStageFlagBits::eCompute,
        *shader_module,
        "main",
        spec_info};

    vk::ComputePipelineCreateInfo pipeline_info{
        {}, stage_info, *pipeline_layout};

    pipeline = vk::raii::Pipeline(context.getDevice(), nullptr, pipeline_info);
}

void VulkanComputePipeline::dispatch(
    vk::raii::CommandBuffer& cmd, vk::DescriptorSet descriptor_set,
    uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z,
    const void* push_constants, uint32_t push_constants_size) {

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0,
                           descriptor_set, nullptr);

    if (push_constants != nullptr && push_constants_size > 0) {
        vk::ArrayProxy<const uint8_t> data(
            push_constants_size, static_cast<const uint8_t*>(push_constants));
        cmd.pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eCompute,
                          0, data);
    }

    cmd.dispatch(group_count_x, group_count_y, group_count_z);
}

DescriptorPool::DescriptorPool(
    VulkanContext& ctx, uint32_t max_sets,
    std::span<const vk::DescriptorPoolSize> pool_sizes)
    : context(ctx) {

    vk::DescriptorPoolCreateInfo pool_info{
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, max_sets,
        static_cast<uint32_t>(pool_sizes.size()), pool_sizes.data()};
    pool = vk::raii::DescriptorPool(context.getDevice(), pool_info);
}

vk::raii::DescriptorSet
DescriptorPool::allocateSet(vk::DescriptorSetLayout layout) {
    vk::DescriptorSetAllocateInfo alloc_info{*pool, 1, &layout};
    auto sets = vk::raii::DescriptorSets(context.getDevice(), alloc_info);
    return std::move(sets[0]);
}

void DescriptorPool::reset() { pool.reset({}); }

} // namespace eedi3vk
