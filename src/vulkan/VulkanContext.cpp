#include "VulkanContext.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define VOLK_IMPLEMENTATION
#include <volk.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace eedi3vk {

VulkanContext& VulkanContext::getInstance(int device_id) {
    struct NoDestroy {
        void operator()(VulkanContext* /*ptr*/) const {}
    };
    using ContextPtr = std::unique_ptr<VulkanContext, NoDestroy>;

    static std::mutex context_mutex;
    static std::unordered_map<int, ContextPtr> contexts;

    int key = device_id;
    if (key < 0) {
        key = -1;
    }

    std::lock_guard<std::mutex> lock(context_mutex);
    auto it = contexts.find(key);
    if (it != contexts.end()) {
        return *it->second;
    }

    auto created = std::make_unique<VulkanContext>(key);
    VulkanContext& ref = *created;

    // Leaky Singleton
    contexts.emplace(key, ContextPtr(created.release()));
    return ref;
}

VulkanContext::VulkanContext(int device_id)
    : device_id(device_id), instance(nullptr) {

    struct VolkInit {
        VolkInit() {
            if (volkInitialize() != VK_SUCCESS) {
                throw std::runtime_error("Failed to initialize volk");
            }
        }
    };
    static VolkInit volk_initer;

    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    createInstance();
    pickPhysicalDevice();
    createDevice();
}

VulkanContext::~VulkanContext() = default;

void VulkanContext::createInstance() {
    vk::ApplicationInfo app_info{"EEDI3VK", VK_MAKE_VERSION(1, 0, 0),
                                 "EEDI3VK Engine", VK_MAKE_VERSION(1, 0, 0),
                                 VK_API_VERSION_1_1};

    std::vector<const char*> instance_extensions;
    vk::InstanceCreateFlags instance_flags{};

    auto available_extensions = context.enumerateInstanceExtensionProperties();
    bool has_portability_enumeration = std::ranges::any_of(
        available_extensions, [](const vk::ExtensionProperties& ext) {
            return strcmp(ext.extensionName,
                          VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        });

    if (has_portability_enumeration) {
        instance_extensions.push_back(
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }

    vk::InstanceCreateInfo create_info{
        instance_flags,
        &app_info,
        0,
        nullptr, // layers
        static_cast<uint32_t>(instance_extensions.size()),
        instance_extensions.data()};

    instance = vk::raii::Instance(context, create_info);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    volkLoadInstance(*instance);
}

void VulkanContext::pickPhysicalDevice() {
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    size_t selected_index = 0;
    if (device_id >= 0) {
        if (static_cast<size_t>(device_id) < devices.size()) {
            selected_index = device_id;
        } else {
            throw std::runtime_error("Invalid device_id: " +
                                     std::to_string(device_id));
        }
    } else {
        for (size_t i = 0; i < devices.size(); i++) {
            auto props = devices[i].getProperties();
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                selected_index = i;
                break;
            }
        }
    }

    physical_device = std::move(devices[selected_index]);

    auto queue_families = physical_device.getQueueFamilyProperties();

    compute_queue_family_index = static_cast<uint32_t>(-1);
    for (uint32_t i = 0; i < queue_families.size(); i++) {
        if (queue_families[i].queueFlags & vk::QueueFlagBits::eCompute) {
            compute_queue_family_index = i;
            break;
        }
    }

    if (compute_queue_family_index == static_cast<uint32_t>(-1)) {
        throw std::runtime_error("No compute queue family found");
    }

    transfer_queue_family_index = compute_queue_family_index;

    int separate_transfer_idx = -1;
    bool found_dedicated = false;

    for (uint32_t i = 0; i < queue_families.size(); i++) {
        const auto& flags = queue_families[i].queueFlags;

        if (!(flags & vk::QueueFlagBits::eTransfer)) {
            continue;
        }

        if (!(flags & vk::QueueFlagBits::eGraphics) &&
            !(flags & vk::QueueFlagBits::eCompute)) {
            transfer_queue_family_index = i;
            found_dedicated = true;
            break;
        }

        if (i != compute_queue_family_index) {
            separate_transfer_idx = static_cast<int>(i);
        }
    }

    if (!found_dedicated && separate_transfer_idx >= 0) {
        transfer_queue_family_index =
            static_cast<uint32_t>(separate_transfer_idx);
    }
}

void VulkanContext::createDevice() {
    std::array<float, 1> compute_queue_priority = {1.0F};
    std::array<float, 1> transfer_queue_priority = {1.0F};

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    queue_create_infos.emplace_back(vk::DeviceQueueCreateInfo{
        {}, compute_queue_family_index, 1, compute_queue_priority.data()});

    if (transfer_queue_family_index != compute_queue_family_index) {
        queue_create_infos.emplace_back(
            vk::DeviceQueueCreateInfo{{},
                                      transfer_queue_family_index,
                                      1,
                                      transfer_queue_priority.data()});
    }

    std::vector<const char*> device_extensions = {};

    auto subgroup_props =
        physical_device.getProperties2<vk::PhysicalDeviceProperties2,
                                       vk::PhysicalDeviceSubgroupProperties>();
    subgroup_size =
        subgroup_props.get<vk::PhysicalDeviceSubgroupProperties>().subgroupSize;

    vk::PhysicalDeviceSubgroupSizeControlFeatures subgroup_size_features{};
    vk::PhysicalDeviceFeatures2 features2{};

    bool has_subgroup_size_control = false;
    std::vector<vk::ExtensionProperties> available_extensions =
        physical_device.enumerateDeviceExtensionProperties();

    const bool has_portability_subset = std::ranges::any_of(
        available_extensions, [](const vk::ExtensionProperties& ext) {
            return strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0;
        });
    if (has_portability_subset) {
        device_extensions.push_back("VK_KHR_portability_subset");
    }

    for (const auto& ext : available_extensions) {
        if (std::string(ext.extensionName.data()) ==
            VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) {
            has_subgroup_size_control = true;
            device_extensions.push_back(
                VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
            break;
        }
    }

    vk::DeviceCreateInfo device_create_info{
        {},
        static_cast<uint32_t>(queue_create_infos.size()),
        queue_create_infos.data(),
        0,
        nullptr,
        static_cast<uint32_t>(device_extensions.size()),
        device_extensions.data()};

    if (has_subgroup_size_control) {
        subgroup_size_features.subgroupSizeControl = VK_TRUE;
        subgroup_size_features.computeFullSubgroups = VK_TRUE;
        features2.pNext = &subgroup_size_features;
        device_create_info.pNext = &features2;
    }

    device = vk::raii::Device(physical_device, device_create_info);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    volkLoadDevice(*device);

    compute_queue = vk::raii::Queue(device, compute_queue_family_index, 0);
    transfer_queue = vk::raii::Queue(device, transfer_queue_family_index, 0);
}

void VulkanContext::submitCompute(const vk::SubmitInfo& submit_info,
                                  vk::Fence fence) {
    std::lock_guard<std::mutex> lock(compute_queue_mutex);
    compute_queue.submit(submit_info, fence);
}

void VulkanContext::submitTransfer(const vk::SubmitInfo& submit_info,
                                   vk::Fence fence) {
    if (transfer_queue_family_index == compute_queue_family_index) {
        std::lock_guard<std::mutex> lock(compute_queue_mutex);
        transfer_queue.submit(submit_info, fence);
        return;
    }

    std::lock_guard<std::mutex> lock(transfer_queue_mutex);
    transfer_queue.submit(submit_info, fence);
}

void VulkanContext::waitIdle() { device.waitIdle(); }

} // namespace eedi3vk
