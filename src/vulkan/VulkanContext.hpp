#ifndef EEDI3VK_VULKAN_CONTEXT_HPP
#define EEDI3VK_VULKAN_CONTEXT_HPP

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

#include <mutex>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>

namespace eedi3vk {

class VulkanContext {
  public:
    static VulkanContext& getInstance(int device_id = -1);

    explicit VulkanContext(int device_id = -1);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    vk::raii::Context& getContext() { return context; }
    vk::raii::Instance& getInstanceRef() { return instance; }
    vk::raii::PhysicalDevice& getPhysicalDevice() { return physical_device; }
    vk::raii::Device& getDevice() { return device; }
    vk::raii::Queue& getComputeQueue() { return compute_queue; }
    [[nodiscard]] uint32_t getQueueFamilyIndex() const {
        return queue_family_index;
    }
    [[nodiscard]] uint32_t getSubgroupSize() const { return subgroup_size; }

    void submit(const vk::SubmitInfo& submit_info, const vk::Fence& fence);
    void waitIdle();

  private:
    void createInstance();
    void pickPhysicalDevice();
    void createDevice();

    int device_id = -1;
    vk::raii::Context context;
    vk::raii::Instance instance;
    vk::raii::PhysicalDevice physical_device = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue compute_queue = nullptr;

    uint32_t queue_family_index = static_cast<uint32_t>(-1);
    uint32_t subgroup_size = 32;
    std::mutex queue_mutex;
};

} // namespace eedi3vk

#endif // EEDI3VK_VULKAN_CONTEXT_HPP
