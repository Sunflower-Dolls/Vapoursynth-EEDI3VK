#include "EEDI3Pipelines.hpp"
#include "Params.hpp"
#include "Shaders.hpp"
#include "VulkanContext.hpp"

#include <array>
#include <cstddef>

namespace eedi3vk {

EEDI3Pipelines::EEDI3Pipelines(VulkanContext& ctx, int tpitch) : context(ctx) {
    // copy_field: 2 buffers (src, dst), push constants
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        }};
        copy_field = std::make_unique<VulkanComputePipeline>(
            context, shaders::getCopyFieldSpv(), bindings,
            sizeof(CopyFieldParams));
    }

    // dilate_mask: 2 buffers (mclip, bmask), push constants
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        }};
        dilate_mask = std::make_unique<VulkanComputePipeline>(
            context, shaders::getDilateMaskSpv(), bindings,
            sizeof(DilateMaskParams));
    }

    // calc_costs: 3 buffers (src, cost, bmask), push constants
    {
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        }};
        calc_costs = std::make_unique<VulkanComputePipeline>(
            context, shaders::getCalcCostsSpv(), bindings,
            sizeof(CalcCostsParams));
    }

    // viterbi_scan: 4 buffers (cost, dmap, pbackt, bmask), push constants
    {
        std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {{
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {3, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        }};

        uint32_t subgroup_size = context.getSubgroupSize();
        auto states_per_thread = (tpitch + subgroup_size - 1) / subgroup_size;

        struct SpecData {
            uint32_t local_size_x;
            uint32_t states_per_thread;
        } spec_data;
        spec_data.local_size_x = subgroup_size;
        spec_data.states_per_thread = states_per_thread;

        std::array<vk::SpecializationMapEntry, 2> entries = {{
            {0, offsetof(SpecData, local_size_x), sizeof(uint32_t)},
            {1, offsetof(SpecData, states_per_thread), sizeof(int)},
        }};

        vk::SpecializationInfo spec_info;
        spec_info.mapEntryCount = static_cast<uint32_t>(entries.size());
        spec_info.pMapEntries = entries.data();
        spec_info.dataSize = sizeof(spec_data);
        spec_info.pData = &spec_data;

        viterbi_scan = std::make_unique<VulkanComputePipeline>(
            context, shaders::getViterbiScanSpv(), bindings,
            sizeof(ViterbiScanParams), &spec_info);
    }

    // interpolate: 4 buffers (src, dst, dmap, bmask), push constants
    {
        std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {{
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {3, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        }};
        interpolate = std::make_unique<VulkanComputePipeline>(
            context, shaders::getInterpolateSpv(), bindings,
            sizeof(InterpolateParams));
    }

    // copy_buffer: 2 buffers (src, dst), push constants
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {0, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1,
             vk::ShaderStageFlagBits::eCompute},
        }};
        copy_buffer = std::make_unique<VulkanComputePipeline>(
            context, shaders::getCopyBufferSpv(), bindings,
            sizeof(CopyBufferParams));
    }
}

vk::raii::DescriptorSet
EEDI3Pipelines::allocateCopyFieldSet(DescriptorPool& pool) {
    return pool.allocateSet(copy_field->getDescriptorSetLayout());
}

vk::raii::DescriptorSet
EEDI3Pipelines::allocateDilateMaskSet(DescriptorPool& pool) {
    return pool.allocateSet(dilate_mask->getDescriptorSetLayout());
}

vk::raii::DescriptorSet
EEDI3Pipelines::allocateCalcCostsSet(DescriptorPool& pool) {
    return pool.allocateSet(calc_costs->getDescriptorSetLayout());
}

vk::raii::DescriptorSet
EEDI3Pipelines::allocateViterbiScanSet(DescriptorPool& pool) {
    return pool.allocateSet(viterbi_scan->getDescriptorSetLayout());
}

vk::raii::DescriptorSet
EEDI3Pipelines::allocateInterpolateSet(DescriptorPool& pool) {
    return pool.allocateSet(interpolate->getDescriptorSetLayout());
}

vk::raii::DescriptorSet
EEDI3Pipelines::allocateCopyBufferSet(DescriptorPool& pool) {
    return pool.allocateSet(copy_buffer->getDescriptorSetLayout());
}

} // namespace eedi3vk
