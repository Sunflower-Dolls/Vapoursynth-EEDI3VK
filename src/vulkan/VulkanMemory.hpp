#ifndef EEDI3VK_VULKAN_MEMORY_HPP
#define EEDI3VK_VULKAN_MEMORY_HPP

#define VK_NO_PROTOTYPES

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>

#include <cstddef>

namespace eedi3vk {

class VulkanContext;

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo alloc_info = {};
    VkDeviceSize size = 0;

    VulkanBuffer() = default;
    VulkanBuffer(VkBuffer buf, VmaAllocation alloc, VmaAllocationInfo info,
                 VkDeviceSize sz)
        : buffer(buf), allocation(alloc), alloc_info(info), size(sz) {}

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    ~VulkanBuffer() = default;

    VulkanBuffer(VulkanBuffer&& other) noexcept
        : buffer(other.buffer), allocation(other.allocation),
          alloc_info(other.alloc_info), size(other.size) {
        other.buffer = VK_NULL_HANDLE;
        other.allocation = VK_NULL_HANDLE;
        other.size = 0;
    }

    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept {
        if (this != &other) {
            buffer = other.buffer;
            allocation = other.allocation;
            alloc_info = other.alloc_info;
            size = other.size;
            other.buffer = VK_NULL_HANDLE;
            other.allocation = VK_NULL_HANDLE;
            other.size = 0;
        }
        return *this;
    }

    [[nodiscard]] bool isValid() const { return buffer != VK_NULL_HANDLE; }
    [[nodiscard]] void* getMappedData() const { return alloc_info.pMappedData; }
};

class VulkanMemory {
  public:
    explicit VulkanMemory(VulkanContext& ctx);
    ~VulkanMemory();

    VulkanMemory(const VulkanMemory&) = delete;
    VulkanMemory& operator=(const VulkanMemory&) = delete;
    VulkanMemory(VulkanMemory&&) = delete;
    VulkanMemory& operator=(VulkanMemory&&) = delete;

    VulkanBuffer createGPUBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    VulkanBuffer createStagingBuffer(VkDeviceSize size, bool for_upload = true);

    void destroyBuffer(VulkanBuffer& buffer);

    void uploadToBuffer(VulkanBuffer& gpu_buffer, const void* data,
                        VkDeviceSize size, VkDeviceSize offset = 0);

    void downloadFromBuffer(VulkanBuffer& gpu_buffer, void* data,
                            VkDeviceSize size, VkDeviceSize offset = 0);

    void copyBuffer(VulkanBuffer& src, VulkanBuffer& dst, VkDeviceSize size);

    [[nodiscard]] VmaAllocator getAllocator() const { return allocator; }

  private:
    VulkanContext& context;
    VmaAllocator allocator = VK_NULL_HANDLE;
    vk::raii::CommandPool transfer_pool = nullptr;
    vk::raii::CommandBuffer transfer_cmd = nullptr;
    vk::raii::Fence transfer_fence = nullptr;
};

} // namespace eedi3vk

#endif // EEDI3VK_VULKAN_MEMORY_HPP
