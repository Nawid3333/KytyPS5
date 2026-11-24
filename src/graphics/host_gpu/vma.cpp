#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <fmt/format.h>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

static std::atomic<uint64_t> g_lab_host_image_bytes {0};
static std::atomic<uint64_t> g_lab_host_image_count {0};

struct LabImageRecord {
	uint64_t            size;
	uint32_t            heap;
	vk::Extent3D        extent;
	vk::Format          format;
	vk::ImageUsageFlags usage;
	uint32_t            layers;
	uint32_t            mips;
};

static constexpr size_t                                  LabTopImages = 12;
static std::mutex                                        g_lab_images_mutex;
static std::unordered_map<VmaAllocation, LabImageRecord> g_lab_images;

static void LabTrackImage(VmaAllocator allocator, VmaAllocation allocation,
                          const vk::ImageCreateInfo& info) {
	VmaAllocationInfo alloc {};
	vmaGetAllocationInfo(allocator, allocation, &alloc);
	const VkPhysicalDeviceMemoryProperties* props = nullptr;
	vmaGetMemoryProperties(allocator, &props);
	std::scoped_lock lock(g_lab_images_mutex);
	g_lab_images[allocation] = {alloc.size,    props->memoryTypes[alloc.memoryType].heapIndex,
	                            info.extent,   info.format,
	                            info.usage,    info.arrayLayers,
	                            info.mipLevels};
}

static void LabForgetImage(VmaAllocation allocation) {
	std::scoped_lock lock(g_lab_images_mutex);
	g_lab_images.erase(allocation);
}

static void LabLogImageCensus(VmaAllocator allocator, const vk::ImageCreateInfo& info) {
	const VkPhysicalDeviceMemoryProperties* props = nullptr;
	vmaGetMemoryProperties(allocator, &props);
	uint32_t allowed = 0;
	for (uint32_t type = 0; type < props->memoryTypeCount; type++) {
		VmaAllocationCreateInfo probe {};
		probe.memoryTypeBits = 1u << type;
		uint32_t index       = 0;
		if (vmaFindMemoryTypeIndexForImageInfo(
		        allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(info), &probe,
		        &index) == VK_SUCCESS) {
			allowed |= 1u << type;
		}
	}
	Log::WriteToConsoleAndLog(
	    fmt::format("[lab-2331] memory types this image may use: 0x{:x}\n", allowed));
	std::vector<LabImageRecord> images;
	{
		std::scoped_lock lock(g_lab_images_mutex);
		images.reserve(g_lab_images.size());
		for (const auto& entry: g_lab_images) {
			images.push_back(entry.second);
		}
	}
	std::array<uint64_t, VK_MAX_MEMORY_HEAPS> bytes {};
	std::array<uint64_t, VK_MAX_MEMORY_HEAPS> count {};
	for (const auto& record: images) {
		bytes[record.heap] += record.size;
		count[record.heap]++;
	}
	for (uint32_t heap = 0; heap < props->memoryHeapCount; heap++) {
		Log::WriteToConsoleAndLog(fmt::format("[lab-2331] live images in heap {}: {} ({} bytes)\n",
		                                      heap, count[heap], bytes[heap]));
	}
	const auto shown = std::min(images.size(), LabTopImages);
	std::partial_sort(
	    images.begin(), images.begin() + static_cast<std::ptrdiff_t>(shown), images.end(),
	    [](const LabImageRecord& a, const LabImageRecord& b) { return a.size > b.size; });
	for (size_t i = 0; i < shown; i++) {
		const auto& record = images[i];
		Log::WriteToConsoleAndLog(
		    fmt::format("[lab-2331] image {} bytes heap {}: {}x{}x{} layers {} mips {} format {} "
		                "usage 0x{:x}\n",
		                record.size, record.heap, record.extent.width, record.extent.height,
		                record.extent.depth, record.layers, record.mips,
		                static_cast<int>(record.format), static_cast<uint32_t>(record.usage)));
	}
}

static bool LabIsHostHeap(VmaAllocator allocator, VmaAllocation allocation, uint64_t* size) {
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	const VkPhysicalDeviceMemoryProperties* props = nullptr;
	vmaGetMemoryProperties(allocator, &props);
	const auto heap = props->memoryTypes[info.memoryType].heapIndex;
	*size           = info.size;
	return (props->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0;
}

static void LabLogImageFailure(VmaAllocator allocator, const vk::ImageCreateInfo& info,
                               vk::Result result) {
	Log::WriteToConsoleAndLog(
	    fmt::format("[lab-2331] vmaCreateImage result {} usage 0x{:x} tiling {} flags 0x{:x} "
	                "samples {} format {}\n",
	                vk::to_string(result), static_cast<uint32_t>(info.usage),
	                vk::to_string(info.tiling), static_cast<uint32_t>(info.flags),
	                static_cast<uint32_t>(info.samples), static_cast<int>(info.format)));
	Log::WriteToConsoleAndLog(fmt::format("[lab-2331] images in host heaps: {} ({} bytes)\n",
	                                      g_lab_host_image_count.load(),
	                                      g_lab_host_image_bytes.load()));
	VmaTotalStatistics stats {};
	vmaCalculateStatistics(allocator, &stats);
	const VkPhysicalDeviceMemoryProperties* props = nullptr;
	vmaGetMemoryProperties(allocator, &props);
	for (uint32_t i = 0; i < props->memoryTypeCount; i++) {
		const auto& type = stats.memoryType[i];
		if (type.statistics.blockCount == 0) {
			continue;
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "[lab-2331] type {} heap {} flags 0x{:x}: blocks {} ({} bytes), allocations {} ({} "
		    "bytes), "
		    "largest {}\n",
		    i, props->memoryTypes[i].heapIndex, props->memoryTypes[i].propertyFlags,
		    type.statistics.blockCount, type.statistics.blockBytes, type.statistics.allocationCount,
		    type.statistics.allocationBytes, type.allocationSizeMax));
	}
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	const auto        result       = static_cast<vk::Result>(
	    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	                   &alloc_info, &native_image, &image.allocation, nullptr));
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		LabLogImageFailure(allocator, image_info, result);
		LabLogImageCensus(allocator, image_info);
		return false;
	}
	if (uint64_t size = 0; LabIsHostHeap(allocator, image.allocation, &size)) {
		g_lab_host_image_bytes += size;
		g_lab_host_image_count++;
	}
	LabTrackImage(allocator, image.allocation, image_info);

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (uint64_t size = 0; LabIsHostHeap(allocator, image.allocation, &size)) {
		g_lab_host_image_bytes -= size;
		g_lab_host_image_count--;
	}
	LabForgetImage(image.allocation);
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
