#include "VulkanResourcePool.hpp"
#include "VulkanContext.hpp"

#include <cstdint>

namespace eedi3vk {

ResourceHolder::ResourceHolder(VulkanResourcePool& p)
    : pool(p), resource(pool.acquire()) {}

ResourceHolder::~ResourceHolder() { pool.release(std::move(resource)); }

VulkanResourcePool::VulkanResourcePool(VulkanContext& ctx, VulkanMemory& mem,
                                       int num_streams, size_t vk_stride,
                                       int max_width, int max_height,
                                       int tpitch, bool has_mclip)
    : context(ctx), memory(mem), semaphore(num_streams) {

    int max_field_height = (max_height + 1) / 2;
    size_t plane_size_bytes = vk_stride * max_height;

    for (int i = 0; i < num_streams; ++i) {
        vk::CommandPoolCreateInfo compute_pool_info{
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            context.getQueueFamilyIndex()};
        vk::raii::CommandPool compute_command_pool(context.getDevice(),
                                                   compute_pool_info);

        vk::CommandBufferAllocateInfo alloc_info{
            *compute_command_pool, vk::CommandBufferLevel::ePrimary, 1};
        auto cmd_buffers =
            vk::raii::CommandBuffers(context.getDevice(), alloc_info);
        vk::raii::CommandBuffer compute_command_buffer =
            std::move(cmd_buffers[0]);

        vk::CommandPoolCreateInfo transfer_pool_info{
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            context.getTransferQueueFamilyIndex()};
        vk::raii::CommandPool transfer_command_pool(context.getDevice(),
                                                    transfer_pool_info);

        vk::CommandBufferAllocateInfo transfer_alloc_info{
            *transfer_command_pool, vk::CommandBufferLevel::ePrimary, 2};
        auto transfer_cmd_buffers =
            vk::raii::CommandBuffers(context.getDevice(), transfer_alloc_info);
        vk::raii::CommandBuffer transfer_upload_command_buffer =
            std::move(transfer_cmd_buffers[0]);
        vk::raii::CommandBuffer transfer_download_command_buffer =
            std::move(transfer_cmd_buffers[1]);

        vk::FenceCreateInfo fence_info{};
        vk::raii::Fence fence(context.getDevice(), fence_info);

        vk::SemaphoreCreateInfo semaphore_info{};
        vk::raii::Semaphore upload_semaphore(context.getDevice(),
                                             semaphore_info);
        vk::raii::Semaphore compute_semaphore(context.getDevice(),
                                              semaphore_info);

        auto params_buffer = memory.createStagingBuffer(256, true);

        size_t pbackt_size = static_cast<size_t>(max_field_height) * max_width *
                             tpitch * sizeof(int32_t);
        auto pbackt_buffer = memory.createGPUBuffer(pbackt_size);

        size_t dmap_size =
            static_cast<size_t>(max_width) * max_field_height * sizeof(int32_t);
        auto dmap_buffer = memory.createGPUBuffer(
            dmap_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        auto dmap_staging = memory.createStagingBuffer(dmap_size, false);

        VulkanBuffer bmask_buffer;
        VulkanBuffer mclip_buffer;
        VulkanBuffer mclip_staging;

        if (has_mclip) {
            size_t bmask_size = static_cast<size_t>(max_width) *
                                max_field_height * sizeof(uint32_t);
            bmask_buffer = memory.createGPUBuffer(bmask_size);
            mclip_buffer = memory.createGPUBuffer(plane_size_bytes);
            mclip_staging = memory.createStagingBuffer(plane_size_bytes, true);
        } else {
            bmask_buffer = memory.createGPUBuffer(256);
        }

        size_t cost_size = static_cast<size_t>(max_width) * max_field_height *
                           tpitch * sizeof(float);
        auto cost_buffer = memory.createGPUBuffer(cost_size);

        auto src_buffer = memory.createGPUBuffer(plane_size_bytes);
        auto dst_buffer = memory.createGPUBuffer(plane_size_bytes);
        auto src_staging = memory.createStagingBuffer(plane_size_bytes, true);
        auto dst_staging = memory.createStagingBuffer(plane_size_bytes, false);

        std::array<vk::DescriptorPoolSize, 1> pool_sizes = {
            {{vk::DescriptorType::eStorageBuffer, 32}}};
        auto descriptor_pool =
            std::make_unique<DescriptorPool>(context, 8, pool_sizes);

        resources.emplace_back(
            std::move(compute_command_pool), std::move(compute_command_buffer),
            std::move(transfer_command_pool),
            std::move(transfer_upload_command_buffer),
            std::move(transfer_download_command_buffer), std::move(fence),
            std::move(upload_semaphore), std::move(compute_semaphore),
            std::move(params_buffer), std::move(pbackt_buffer),
            std::move(dmap_buffer), std::move(bmask_buffer),
            std::move(cost_buffer), std::move(dmap_staging),
            std::move(src_buffer), std::move(dst_buffer),
            std::move(src_staging), std::move(dst_staging),
            std::move(mclip_buffer), std::move(mclip_staging),
            std::move(descriptor_pool));
    }
}

VulkanResourcePool::~VulkanResourcePool() {
    for (auto& res : resources) {
        memory.destroyBuffer(res.params_buffer);
        memory.destroyBuffer(res.pbackt_buffer);
        memory.destroyBuffer(res.dmap_buffer);
        memory.destroyBuffer(res.bmask_buffer);
        memory.destroyBuffer(res.cost_buffer);
        memory.destroyBuffer(res.dmap_staging);

        memory.destroyBuffer(res.src_buffer);
        memory.destroyBuffer(res.dst_buffer);
        memory.destroyBuffer(res.src_staging);
        memory.destroyBuffer(res.dst_staging);
        if (res.mclip_buffer.isValid()) {
            memory.destroyBuffer(res.mclip_buffer);
        }
        if (res.mclip_staging.isValid()) {
            memory.destroyBuffer(res.mclip_staging);
        }
    }
}

FrameResources VulkanResourcePool::acquire() {
    semaphore.acquire();
    std::lock_guard<std::mutex> lock(resources_lock);
    FrameResources res = std::move(resources.front());
    resources.pop_front();
    return res;
}

void VulkanResourcePool::release(FrameResources&& res) {
    std::lock_guard<std::mutex> lock(resources_lock);
    resources.push_back(std::move(res));
    semaphore.release();
}

} // namespace eedi3vk
