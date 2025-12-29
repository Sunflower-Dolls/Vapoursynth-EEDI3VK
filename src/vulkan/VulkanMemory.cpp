#include "VulkanMemory.hpp"
#include "VulkanContext.hpp"

#include <cstring>
#include <stdexcept>

#include <volk.h>

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

#include <vk_mem_alloc.h>

namespace eedi3vk {

VulkanMemory::VulkanMemory(VulkanContext& ctx) : context(ctx) {
    VmaVulkanFunctions vulkan_functions = {};
    vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_1;
    allocator_info.physicalDevice = *context.getPhysicalDevice();
    allocator_info.device = *context.getDevice();
    allocator_info.instance = *context.getInstanceRef();
    allocator_info.pVulkanFunctions = &vulkan_functions;

    if (vmaCreateAllocator(&allocator_info, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }

    vk::CommandPoolCreateInfo pool_info{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        context.getQueueFamilyIndex()};
    transfer_pool = vk::raii::CommandPool(context.getDevice(), pool_info);

    vk::CommandBufferAllocateInfo alloc_info{
        *transfer_pool, vk::CommandBufferLevel::ePrimary, 1};
    auto cmd_buffers =
        vk::raii::CommandBuffers(context.getDevice(), alloc_info);
    transfer_cmd = std::move(cmd_buffers[0]);

    vk::FenceCreateInfo fence_info{};
    transfer_fence = vk::raii::Fence(context.getDevice(), fence_info);
}

VulkanMemory::~VulkanMemory() {
    if (allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator);
    }
}

VulkanBuffer VulkanMemory::createGPUBuffer(VkDeviceSize size,
                                           VkBufferUsageFlags usage) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkBuffer buffer = nullptr;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocation_info;

    if (vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer,
                        &allocation, &allocation_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create GPU buffer");
    }

    return {buffer, allocation, allocation_info, size};
}

VulkanBuffer VulkanMemory::createStagingBuffer(VkDeviceSize size,
                                               bool for_upload) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = for_upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                   : VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = nullptr;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocation_info;

    if (vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer,
                        &allocation, &allocation_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create staging buffer");
    }

    return VulkanBuffer(buffer, allocation, allocation_info, size);
}

void VulkanMemory::destroyBuffer(VulkanBuffer& buffer) {
    if (buffer.isValid()) {
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
        buffer.buffer = VK_NULL_HANDLE;
        buffer.allocation = VK_NULL_HANDLE;
        buffer.size = 0;
    }
}

void VulkanMemory::uploadToBuffer(VulkanBuffer& gpu_buffer, const void* data,
                                  VkDeviceSize size, VkDeviceSize offset) {
    auto staging = createStagingBuffer(size, true);
    std::memcpy(staging.getMappedData(), data, size);

    transfer_cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy copy_region{0, offset, size};
    transfer_cmd.copyBuffer(staging.buffer, gpu_buffer.buffer, copy_region);
    transfer_cmd.end();

    vk::SubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &*transfer_cmd;

    context.submit(submit_info, *transfer_fence);

    auto result =
        context.getDevice().waitForFences(*transfer_fence, VK_TRUE, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed waiting for upload fence");
    }
    context.getDevice().resetFences(*transfer_fence);

    destroyBuffer(staging);
}

void VulkanMemory::downloadFromBuffer(VulkanBuffer& gpu_buffer, void* data,
                                      VkDeviceSize size, VkDeviceSize offset) {
    auto staging = createStagingBuffer(size, false);

    transfer_cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy copy_region{offset, 0, size};
    transfer_cmd.copyBuffer(gpu_buffer.buffer, staging.buffer, copy_region);
    transfer_cmd.end();

    vk::SubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &*transfer_cmd;

    context.submit(submit_info, *transfer_fence);

    auto result =
        context.getDevice().waitForFences(*transfer_fence, VK_TRUE, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed waiting for download fence");
    }
    context.getDevice().resetFences(*transfer_fence);

    vmaInvalidateAllocation(allocator, staging.allocation, 0, size);
    std::memcpy(data, staging.getMappedData(), size);

    destroyBuffer(staging);
}

void VulkanMemory::copyBuffer(VulkanBuffer& src, VulkanBuffer& dst,
                              VkDeviceSize size) {
    transfer_cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy copy_region{0, 0, size};
    transfer_cmd.copyBuffer(src.buffer, dst.buffer, copy_region);
    transfer_cmd.end();

    vk::SubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &*transfer_cmd;

    context.submit(submit_info, *transfer_fence);

    auto result =
        context.getDevice().waitForFences(*transfer_fence, VK_TRUE, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed waiting for copy fence");
    }
    context.getDevice().resetFences(*transfer_fence);
}

} // namespace eedi3vk
