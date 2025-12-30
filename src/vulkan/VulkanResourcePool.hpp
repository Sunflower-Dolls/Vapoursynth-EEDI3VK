#ifndef EEDI3VK_VULKAN_RESOURCE_POOL_HPP
#define EEDI3VK_VULKAN_RESOURCE_POOL_HPP

#define VK_NO_PROTOTYPES

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

#include <atomic>
#include <deque>
#include <mutex>
#include <vulkan/vulkan_raii.hpp>

#include "VulkanComputePipeline.hpp"
#include "VulkanMemory.hpp"

namespace eedi3vk {

class VulkanContext;

class TicketSemaphore {
  public:
    explicit TicketSemaphore(int max_tickets)
        : ticket(0), current(max_tickets - 1) {}

    void acquire() noexcept {
        intptr_t tk = ticket.fetch_add(1, std::memory_order_acquire);
        while (true) {
            intptr_t curr = current.load(std::memory_order_acquire);
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order_relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order_release);
        current.notify_all();
    }

  private:
    std::atomic<intptr_t> ticket;
    std::atomic<intptr_t> current;
};

struct FrameResources {
    vk::raii::CommandPool compute_command_pool;
    vk::raii::CommandBuffer compute_command_buffer = nullptr;
    vk::raii::CommandPool transfer_command_pool;
    vk::raii::CommandBuffer transfer_upload_command_buffer = nullptr;
    vk::raii::CommandBuffer transfer_download_command_buffer = nullptr;
    vk::raii::Fence fence;
    vk::raii::Semaphore upload_semaphore;
    vk::raii::Semaphore compute_semaphore;

    VulkanBuffer params_buffer;
    VulkanBuffer pbackt_buffer;
    VulkanBuffer dmap_buffer;
    VulkanBuffer bmask_buffer;
    VulkanBuffer cost_buffer;

    VulkanBuffer dmap_staging;

    VulkanBuffer src_buffer;
    VulkanBuffer dst_buffer;
    VulkanBuffer src_staging;
    VulkanBuffer dst_staging;
    VulkanBuffer mclip_buffer;
    VulkanBuffer mclip_staging;

    std::unique_ptr<DescriptorPool> descriptor_pool;
    std::vector<vk::raii::DescriptorSet> descriptor_sets;

    FrameResources(
        vk::raii::CommandPool&& ccp, vk::raii::CommandBuffer&& ccb,
        vk::raii::CommandPool&& tcp, vk::raii::CommandBuffer&& tcb_upload,
        vk::raii::CommandBuffer&& tcb_download, vk::raii::Fence&& f,
        vk::raii::Semaphore&& upload_sem, vk::raii::Semaphore&& compute_sem,
        VulkanBuffer&& params, VulkanBuffer&& pbackt, VulkanBuffer&& dmap,
        VulkanBuffer&& bmask, VulkanBuffer&& cost, VulkanBuffer&& dmap_stg,
        VulkanBuffer&& src, VulkanBuffer&& dst, VulkanBuffer&& src_stg,
        VulkanBuffer&& dst_stg, VulkanBuffer&& mclip, VulkanBuffer&& mclip_stg,
        std::unique_ptr<DescriptorPool>&& dp)
        : compute_command_pool(std::move(ccp)),
          compute_command_buffer(std::move(ccb)),
          transfer_command_pool(std::move(tcp)),
          transfer_upload_command_buffer(std::move(tcb_upload)),
          transfer_download_command_buffer(std::move(tcb_download)),
          fence(std::move(f)), upload_semaphore(std::move(upload_sem)),
          compute_semaphore(std::move(compute_sem)),
          params_buffer(std::move(params)), pbackt_buffer(std::move(pbackt)),
          dmap_buffer(std::move(dmap)), bmask_buffer(std::move(bmask)),
          cost_buffer(std::move(cost)), dmap_staging(std::move(dmap_stg)),
          src_buffer(std::move(src)), dst_buffer(std::move(dst)),
          src_staging(std::move(src_stg)), dst_staging(std::move(dst_stg)),
          mclip_buffer(std::move(mclip)), mclip_staging(std::move(mclip_stg)),
          descriptor_pool(std::move(dp)) {}

    ~FrameResources() = default;

    FrameResources(const FrameResources&) = delete;
    FrameResources& operator=(const FrameResources&) = delete;

    FrameResources(FrameResources&& other) noexcept = default;
    FrameResources& operator=(FrameResources&& other) noexcept = default;
};

class ResourceHolder {
  public:
    ResourceHolder(class VulkanResourcePool& pool);
    ~ResourceHolder();

    ResourceHolder(const ResourceHolder&) = delete;
    ResourceHolder& operator=(const ResourceHolder&) = delete;
    ResourceHolder(ResourceHolder&&) = delete;
    ResourceHolder& operator=(ResourceHolder&&) = delete;

    FrameResources& get() { return resource; }

  private:
    VulkanResourcePool& pool;
    FrameResources resource;
};

class VulkanResourcePool {
  public:
    VulkanResourcePool(VulkanContext& ctx, VulkanMemory& memory,
                       int num_streams, size_t vk_stride, int max_width,
                       int max_height, int tpitch, bool has_mclip);
    ~VulkanResourcePool();

    VulkanResourcePool(const VulkanResourcePool&) = delete;
    VulkanResourcePool& operator=(const VulkanResourcePool&) = delete;
    VulkanResourcePool(VulkanResourcePool&&) = delete;
    VulkanResourcePool& operator=(VulkanResourcePool&&) = delete;

    FrameResources acquire();
    void release(FrameResources&& res);

    [[nodiscard]] VulkanContext& getContext() { return context; }
    [[nodiscard]] VulkanMemory& getMemory() { return memory; }

  private:
    VulkanContext& context;
    VulkanMemory& memory;
    TicketSemaphore semaphore;
    std::deque<FrameResources> resources;
    std::mutex resources_lock;
};

} // namespace eedi3vk

#endif // EEDI3VK_VULKAN_RESOURCE_POOL_HPP
