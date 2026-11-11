#include "graphics/host_gpu/vulkanCommon.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cinttypes>
#include <iterator>
#include <map>
#include <mutex>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace Libs::Graphics {
namespace {

struct FormatMapping {
	Prospero::BufferFormat guest;
	vk::Format             host;
};

constexpr FormatMapping kFormatMappings[] = {
    {Prospero::BufferFormat::k8UNorm, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8SNorm, vk::Format::eR8Snorm},
    {Prospero::BufferFormat::k8UScaled, vk::Format::eR8Uscaled},
    {Prospero::BufferFormat::k8SScaled, vk::Format::eR8Sscaled},
    {Prospero::BufferFormat::k8UInt, vk::Format::eR8Uint},
    {Prospero::BufferFormat::k8SInt, vk::Format::eR8Sint},
    {Prospero::BufferFormat::k16UNorm, vk::Format::eR16Unorm},
    {Prospero::BufferFormat::k16SNorm, vk::Format::eR16Snorm},
    {Prospero::BufferFormat::k16UScaled, vk::Format::eR16Uscaled},
    {Prospero::BufferFormat::k16SScaled, vk::Format::eR16Sscaled},
    {Prospero::BufferFormat::k16UInt, vk::Format::eR16Uint},
    {Prospero::BufferFormat::k16SInt, vk::Format::eR16Sint},
    {Prospero::BufferFormat::k16Float, vk::Format::eR16Sfloat},
    {Prospero::BufferFormat::k8_8UNorm, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8SNorm, vk::Format::eR8G8Snorm},
    {Prospero::BufferFormat::k8_8UScaled, vk::Format::eR8G8Uscaled},
    {Prospero::BufferFormat::k8_8SScaled, vk::Format::eR8G8Sscaled},
    {Prospero::BufferFormat::k8_8UInt, vk::Format::eR8G8Uint},
    {Prospero::BufferFormat::k8_8SInt, vk::Format::eR8G8Sint},
    {Prospero::BufferFormat::k32UInt, vk::Format::eR32Uint},
    {Prospero::BufferFormat::k32SInt, vk::Format::eR32Sint},
    {Prospero::BufferFormat::k32Float, vk::Format::eR32Sfloat},
    {Prospero::BufferFormat::k16_16UNorm, vk::Format::eR16G16Unorm},
    {Prospero::BufferFormat::k16_16SNorm, vk::Format::eR16G16Snorm},
    {Prospero::BufferFormat::k16_16UScaled, vk::Format::eR16G16Uscaled},
    {Prospero::BufferFormat::k16_16SScaled, vk::Format::eR16G16Sscaled},
    {Prospero::BufferFormat::k16_16UInt, vk::Format::eR16G16Uint},
    {Prospero::BufferFormat::k16_16SInt, vk::Format::eR16G16Sint},
    {Prospero::BufferFormat::k16_16Float, vk::Format::eR16G16Sfloat},
    {Prospero::BufferFormat::k11_11_10Float, vk::Format::eB10G11R11UfloatPack32},
    {Prospero::BufferFormat::k10_10_10_2UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k10_10_10_2SNorm, vk::Format::eA2B10G10R10SnormPack32},
    {Prospero::BufferFormat::k10_10_10_2UScaled, vk::Format::eA2B10G10R10UscaledPack32},
    {Prospero::BufferFormat::k10_10_10_2UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k8_8_8_8UNorm, vk::Format::eR8G8B8A8Unorm},
    {Prospero::BufferFormat::k8_8_8_8SNorm, vk::Format::eR8G8B8A8Snorm},
    {Prospero::BufferFormat::k8_8_8_8UScaled, vk::Format::eR8G8B8A8Uscaled},
    {Prospero::BufferFormat::k8_8_8_8SScaled, vk::Format::eR8G8B8A8Sscaled},
    {Prospero::BufferFormat::k8_8_8_8UInt, vk::Format::eR8G8B8A8Uint},
    {Prospero::BufferFormat::k8_8_8_8SInt, vk::Format::eR8G8B8A8Sint},
    {Prospero::BufferFormat::k32_32UInt, vk::Format::eR32G32Uint},
    {Prospero::BufferFormat::k32_32SInt, vk::Format::eR32G32Sint},
    {Prospero::BufferFormat::k32_32Float, vk::Format::eR32G32Sfloat},
    {Prospero::BufferFormat::k16_16_16_16UNorm, vk::Format::eR16G16B16A16Unorm},
    {Prospero::BufferFormat::k16_16_16_16SNorm, vk::Format::eR16G16B16A16Snorm},
    {Prospero::BufferFormat::k16_16_16_16UScaled, vk::Format::eR16G16B16A16Uscaled},
    {Prospero::BufferFormat::k16_16_16_16SScaled, vk::Format::eR16G16B16A16Sscaled},
    {Prospero::BufferFormat::k16_16_16_16UInt, vk::Format::eR16G16B16A16Uint},
    {Prospero::BufferFormat::k16_16_16_16SInt, vk::Format::eR16G16B16A16Sint},
    {Prospero::BufferFormat::k16_16_16_16Float, vk::Format::eR16G16B16A16Sfloat},
    {Prospero::BufferFormat::k32_32_32UInt, vk::Format::eR32G32B32Uint},
    {Prospero::BufferFormat::k32_32_32SInt, vk::Format::eR32G32B32Sint},
    {Prospero::BufferFormat::k32_32_32Float, vk::Format::eR32G32B32Sfloat},
    {Prospero::BufferFormat::k32_32_32_32UInt, vk::Format::eR32G32B32A32Uint},
    {Prospero::BufferFormat::k32_32_32_32SInt, vk::Format::eR32G32B32A32Sint},
    {Prospero::BufferFormat::k32_32_32_32Float, vk::Format::eR32G32B32A32Sfloat},
    // 8 and 8_8 sRGB views are optional in Vulkan, so these images store UNorm.
    {Prospero::BufferFormat::k8Srgb, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8_8Srgb, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8_8_8Srgb, vk::Format::eR8G8B8A8Srgb},
    {Prospero::BufferFormat::k9_9_9_5Float, vk::Format::eE5B9G9R9UfloatPack32},
    {Prospero::BufferFormat::k5_6_5UNorm, vk::Format::eB5G6R5UnormPack16},
    {Prospero::BufferFormat::k5_5_5_1UNorm, vk::Format::eA1R5G5B5UnormPack16},
    {Prospero::BufferFormat::k1_5_5_5UNorm, vk::Format::eR5G5B5A1UnormPack16},
    {Prospero::BufferFormat::k4_4_4_4UNorm, vk::Format::eR4G4B4A4UnormPack16},
    {Prospero::BufferFormat::kBc1UNorm, vk::Format::eBc1RgbaUnormBlock},
    {Prospero::BufferFormat::kBc1Srgb, vk::Format::eBc1RgbaSrgbBlock},
    {Prospero::BufferFormat::kBc2UNorm, vk::Format::eBc2UnormBlock},
    {Prospero::BufferFormat::kBc2Srgb, vk::Format::eBc2SrgbBlock},
    {Prospero::BufferFormat::kBc3UNorm, vk::Format::eBc3UnormBlock},
    {Prospero::BufferFormat::kBc3Srgb, vk::Format::eBc3SrgbBlock},
    {Prospero::BufferFormat::kBc4UNorm, vk::Format::eBc4UnormBlock},
    {Prospero::BufferFormat::kBc4SNorm, vk::Format::eBc4SnormBlock},
    {Prospero::BufferFormat::kBc5UNorm, vk::Format::eBc5UnormBlock},
    {Prospero::BufferFormat::kBc5SNorm, vk::Format::eBc5SnormBlock},
    {Prospero::BufferFormat::kBc6UFloat, vk::Format::eBc6HUfloatBlock},
    {Prospero::BufferFormat::kBc6SFloat, vk::Format::eBc6HSfloatBlock},
    {Prospero::BufferFormat::kBc7UNorm, vk::Format::eBc7UnormBlock},
    {Prospero::BufferFormat::kBc7Srgb, vk::Format::eBc7SrgbBlock},
};

constexpr auto MakeFormatLookup() {
	constexpr auto kMaxFormat = static_cast<size_t>(Prospero::BufferFormat::kBc7Srgb);
	std::array<vk::Format, kMaxFormat + 1> lookup {};
	lookup.fill(vk::Format::eUndefined);
	for (const auto& mapping: kFormatMappings) {
		lookup[static_cast<size_t>(mapping.guest)] = mapping.host;
	}
	return lookup;
}

constexpr auto kFormatLookup = MakeFormatLookup();

} // namespace

vk::Format VulkanFormat(Prospero::BufferFormat guest_format) {
	const auto index = static_cast<size_t>(guest_format);
	return index < kFormatLookup.size() ? kFormatLookup[index] : vk::Format::eUndefined;
}

namespace {

struct LabFaultRange {
	uint64_t size;
	uint64_t guest;
	int      usage;
	bool     alive;
};

std::mutex                        g_lab_fault_mutex;
std::map<uint64_t, LabFaultRange> g_lab_fault_ranges;
vk::Device                        g_lab_fault_device   = nullptr;
bool                              g_lab_fault_reported = false;

void LabDeviceFaultLine(const char* where, uint64_t address, uint64_t start,
                        const LabFaultRange& range) {
	LOGF("[lab-fault]   %s va=0x%016" PRIx64 " size=0x%" PRIx64 " offset=%s0x%" PRIx64
	     " guest=0x%016" PRIx64 " usage=%d %s\n",
	     where, start, range.size, address < start ? "-" : "",
	     address < start ? start - address : address - start, range.guest, range.usage,
	     range.alive ? "alive" : "destroyed");
}

void LabDeviceFaultOwner(uint64_t address, uint64_t precision) {
	const auto low   = address & ~(precision - 1);
	const auto high  = low + precision < low ? UINT64_MAX : low + precision;
	bool       found = false;
	for (const auto& [start, range]: g_lab_fault_ranges) {
		if (start >= high || start + range.size <= low) continue;
		found = true;
		LabDeviceFaultLine("in buffer", address, start, range);
	}
	if (found) return;
	const auto above = g_lab_fault_ranges.upper_bound(address);
	if (above != g_lab_fault_ranges.begin()) {
		const auto below = std::prev(above);
		LabDeviceFaultLine("in no tracked buffer; below it:", address, below->first,
		                   below->second);
	}
	if (above != g_lab_fault_ranges.end()) {
		LabDeviceFaultLine("in no tracked buffer; above it:", address, above->first,
		                   above->second);
	}
}

} // namespace

namespace {

constexpr uint64_t kLabWorkSlots     = 8192;
constexpr uint64_t kLabWorkContext   = 4;
constexpr uint32_t kLabWorkMaxListed = 48;

struct LabWork {
	uint64_t        id        = 0;
	uint64_t        tick      = 0;
	uint64_t        hash0     = 0;
	uint64_t        hash1     = 0;
	uint32_t        arg0      = 0;
	uint32_t        arg1      = 0;
	uint32_t        arg2      = 0;
	int64_t         ms        = 0;
	VkCommandBuffer command   = VK_NULL_HANDLE;
	VkSemaphore     semaphore = VK_NULL_HANDLE;
	char            kind      = 0;
};

struct LabWorkDone {
	VkSemaphore semaphore;
	uint64_t    tick;
};

std::array<LabWork, kLabWorkSlots> g_lab_work {};
uint64_t                           g_lab_work_next  = 1;
uint64_t                           g_lab_work_floor = 1;
std::vector<LabWorkDone>           g_lab_work_done;
VkDevice                           g_lab_work_device   = VK_NULL_HANDLE;
VkBuffer                           g_lab_marker_buffer = VK_NULL_HANDLE;
const uint32_t*                    g_lab_marker_words  = nullptr;

int64_t LabWorkNow() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void LabWorkMark(vk::CommandBuffer command, VkPipelineStageFlagBits stage, uint64_t id,
                 uint32_t word) {
	const auto write = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD;
	if (g_lab_marker_buffer == VK_NULL_HANDLE || write == nullptr) return;
	write(static_cast<VkCommandBuffer>(command), stage, g_lab_marker_buffer,
	      ((id % kLabWorkSlots) * 2 + word) * sizeof(uint32_t), static_cast<uint32_t>(id));
}

uint64_t LabWorkDoneTick(VkSemaphore semaphore) {
	for (const auto& done: g_lab_work_done) {
		if (done.semaphore == semaphore) return done.tick;
	}
	return 0;
}

const char* LabWorkMarkerState(const LabWork& work) {
	if (g_lab_marker_words == nullptr) return "off";
	const auto slot = (work.id % kLabWorkSlots) * 2;
	const auto id   = static_cast<uint32_t>(work.id);
	if (g_lab_marker_words[slot + 1] == id) return "ended";
	return g_lab_marker_words[slot] == id ? "begun" : "not-begun";
}

void LabWorkReport() {
	const auto now = LabWorkNow();
	for (size_t index = 0; index < g_lab_work_done.size(); ++index) {
		auto&      done    = g_lab_work_done[index];
		uint64_t   counter = 0;
		const auto read    = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetSemaphoreCounterValue;
		const auto status  = g_lab_work_device == VK_NULL_HANDLE || read == nullptr
		                         ? VK_ERROR_UNKNOWN
		                         : read(g_lab_work_device, done.semaphore, &counter);
		LOGF("[lab-work] semaphore %zu: known_tick=%" PRIu64 " counter_read=%d counter=%" PRIu64 "\n",
		     index, done.tick, static_cast<int>(status), counter);
		if (status == VK_SUCCESS && counter > done.tick) done.tick = counter;
	}
	const auto first = g_lab_work_next > kLabWorkSlots ? g_lab_work_next - kLabWorkSlots : 1;
	const auto is_open = [](const LabWork& work) {
		return work.tick != 0 && work.tick > LabWorkDoneTick(work.semaphore);
	};
	const auto is_begun = [](const LabWork& work) {
		return g_lab_marker_words != nullptr && g_lab_marker_words[(work.id % kLabWorkSlots) * 2] ==
		                                            static_cast<uint32_t>(work.id) &&
		       g_lab_marker_words[(work.id % kLabWorkSlots) * 2 + 1] != static_cast<uint32_t>(work.id);
	};
	uint64_t unsubmitted = 0;
	uint64_t open_count  = 0;
	uint64_t begun_count = 0;
	uint64_t first_open  = g_lab_work_next;
	for (auto id = first; id < g_lab_work_next; ++id) {
		const auto& work = g_lab_work[id % kLabWorkSlots];
		unsubmitted += work.tick == 0 ? 1 : 0;
		open_count += is_open(work) ? 1 : 0;
		begun_count += is_begun(work) ? 1 : 0;
		if ((is_open(work) || is_begun(work)) && id < first_open) first_open = id;
	}
	LOGF("[lab-work] device lost; next_id=%" PRIu64 " markers=%s entries=%" PRIu64
	     " unsubmitted=%" PRIu64 " open_by_tick=%" PRIu64 " begun_not_ended=%" PRIu64 "\n",
	     g_lab_work_next, g_lab_marker_words != nullptr ? "on" : "off", g_lab_work_next - first,
	     unsubmitted, open_count, begun_count);
	uint32_t listed = 0;
	uint64_t more   = 0;
	for (auto id = std::max(first, first_open > kLabWorkContext ? first_open - kLabWorkContext : 1);
	     id < g_lab_work_next; ++id) {
		const auto& work = g_lab_work[id % kLabWorkSlots];
		if (id >= first_open && !is_open(work) && !is_begun(work)) continue;
		if (id >= first_open && listed >= kLabWorkMaxListed) {
			++more;
			continue;
		}
		listed += id >= first_open ? 1 : 0;
		LOGF("[lab-work] id=%" PRIu64 " kind=%c tick=%" PRIu64 " state=%s marker=%s hashes=0x%016" PRIx64
		     ",0x%016" PRIx64 " args=%u,%u,%u age_ms=%" PRId64 "\n",
		     work.id, work.kind, work.tick,
		     work.tick == 0 ? "unsubmitted" : is_open(work) ? "open" : "done", LabWorkMarkerState(work),
		     work.hash0, work.hash1, work.arg0, work.arg1, work.arg2, now - work.ms);
	}
	if (more != 0) LOGF("[lab-work] ... and %" PRIu64 " more open entries\n", more);
}

} // namespace

void LabWorkEnable(vk::PhysicalDevice physical_device, vk::Device device, bool markers) {
	const auto& vk_api = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto  handle = static_cast<VkDevice>(device);
	{
		std::lock_guard lock(g_lab_fault_mutex);
		g_lab_work_device = handle;
	}
	if (!markers) return;
	VkBufferCreateInfo info {};
	info.sType  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size   = kLabWorkSlots * 2 * sizeof(uint32_t);
	info.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	auto buffer = VkBuffer {VK_NULL_HANDLE};
	if (vk_api.vkCreateBuffer(handle, &info, nullptr, &buffer) != VK_SUCCESS) return;
	VkMemoryRequirements requirements {};
	vk_api.vkGetBufferMemoryRequirements(handle, buffer, &requirements);
	VkPhysicalDeviceMemoryProperties memory {};
	vk_api.vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(physical_device),
	                                           &memory);
	constexpr VkMemoryPropertyFlags wanted =
	    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
		if ((requirements.memoryTypeBits & (1u << type)) == 0 ||
		    (memory.memoryTypes[type].propertyFlags & wanted) != wanted) {
			continue;
		}
		VkMemoryAllocateInfo allocate {};
		allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate.allocationSize  = requirements.size;
		allocate.memoryTypeIndex = type;
		auto  block              = VkDeviceMemory {VK_NULL_HANDLE};
		void* mapped             = nullptr;
		if (vk_api.vkAllocateMemory(handle, &allocate, nullptr, &block) != VK_SUCCESS) continue;
		if (vk_api.vkBindBufferMemory(handle, buffer, block, 0) != VK_SUCCESS ||
		    vk_api.vkMapMemory(handle, block, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
			return;
		}
		std::memset(mapped, 0, static_cast<size_t>(info.size));
		std::lock_guard lock(g_lab_fault_mutex);
		g_lab_marker_buffer = buffer;
		g_lab_marker_words  = static_cast<const uint32_t*>(mapped);
		return;
	}
}

uint64_t LabWorkBegin(vk::CommandBuffer command, char kind, uint64_t hash0, uint64_t hash1,
                      uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	std::lock_guard lock(g_lab_fault_mutex);
	const auto      id = g_lab_work_next++;
	auto&           work = g_lab_work[id % kLabWorkSlots];
	work.id        = id;
	work.tick      = 0;
	work.hash0     = hash0;
	work.hash1     = hash1;
	work.arg0      = arg0;
	work.arg1      = arg1;
	work.arg2      = arg2;
	work.ms        = LabWorkNow();
	work.command   = static_cast<VkCommandBuffer>(command);
	work.semaphore = VK_NULL_HANDLE;
	work.kind      = kind;
	LabWorkMark(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, id, 0);
	return id;
}

void LabWorkEnd(vk::CommandBuffer command, uint64_t id) {
	std::lock_guard lock(g_lab_fault_mutex);
	LabWorkMark(command,
	            g_lab_work[id % kLabWorkSlots].kind == 'D' ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
	                                                       : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	            id, 1);
}

void LabWorkSubmitted(vk::CommandBuffer command, vk::Semaphore semaphore, uint64_t tick) {
	std::lock_guard lock(g_lab_fault_mutex);
	const auto      first = g_lab_work_next > kLabWorkSlots ? g_lab_work_next - kLabWorkSlots : 1;
	auto            floor = g_lab_work_next;
	for (auto id = g_lab_work_next; id-- > std::max(first, g_lab_work_floor);) {
		auto& work = g_lab_work[id % kLabWorkSlots];
		if (work.tick != 0) continue;
		if (work.command == static_cast<VkCommandBuffer>(command)) {
			work.tick      = tick;
			work.semaphore = static_cast<VkSemaphore>(semaphore);
		} else {
			floor = id;
		}
	}
	g_lab_work_floor = floor;
}

void LabWorkCompleted(vk::Semaphore semaphore, uint64_t tick) {
	std::lock_guard lock(g_lab_fault_mutex);
	for (auto& done: g_lab_work_done) {
		if (done.semaphore != static_cast<VkSemaphore>(semaphore)) continue;
		done.tick = std::max(done.tick, tick);
		return;
	}
	g_lab_work_done.push_back({static_cast<VkSemaphore>(semaphore), tick});
}

void LabDeviceFaultEnable(vk::Device device) {
	std::lock_guard lock(g_lab_fault_mutex);
	g_lab_fault_device = device;
}

void LabDeviceFaultTrack(uint64_t device_address, uint64_t size, uint64_t guest, int usage,
                         bool alive) {
	std::lock_guard lock(g_lab_fault_mutex);
	g_lab_fault_ranges[device_address] = {size, guest, usage, alive};
}

void LabDeviceFaultReport(vk::Result result) {
	if (result != vk::Result::eErrorDeviceLost) return;
	std::lock_guard lock(g_lab_fault_mutex);
	if (g_lab_fault_reported) return;
	g_lab_fault_reported = true;
	LabWorkReport();
	const auto query     = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT;
	if (g_lab_fault_device == nullptr || query == nullptr) {
		LOGF("[lab-fault] device lost; VK_EXT_device_fault is not enabled on this device\n");
		return;
	}
	const auto             device = static_cast<VkDevice>(g_lab_fault_device);
	VkDeviceFaultCountsEXT counts {};
	counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
	auto status  = query(device, &counts, nullptr);
	LOGF("[lab-fault] device lost; status=%d addresses=%u vendor_infos=%u vendor_binary=%" PRIu64
	     " tracked_buffers=%zu\n",
	     static_cast<int>(status), counts.addressInfoCount, counts.vendorInfoCount,
	     static_cast<uint64_t>(counts.vendorBinarySize), g_lab_fault_ranges.size());
	if (status != VK_SUCCESS) return;
	counts.vendorBinarySize = 0;
	std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	VkDeviceFaultInfoEXT                     info {};
	info.sType         = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	status             = query(device, &counts, &info);
	if (status != VK_SUCCESS && status != VK_INCOMPLETE) {
		LOGF("[lab-fault] fault info query failed: %d\n", static_cast<int>(status));
		return;
	}
	info.description[VK_MAX_DESCRIPTION_SIZE - 1] = 0;
	LOGF("[lab-fault] description: %s%s\n", info.description,
	     status == VK_INCOMPLETE ? " (incomplete)" : "");
	for (uint32_t index = 0; index < counts.addressInfoCount && index < addresses.size(); ++index) {
		const auto& fault = addresses[index];
		LOGF("[lab-fault] address %u: type=%s reported=0x%016" PRIx64 " precision=0x%" PRIx64 "\n",
		     index,
		     vk::to_string(static_cast<vk::DeviceFaultAddressTypeEXT>(fault.addressType)).c_str(),
		     static_cast<uint64_t>(fault.reportedAddress),
		     static_cast<uint64_t>(fault.addressPrecision));
		LabDeviceFaultOwner(fault.reportedAddress,
		                    fault.addressPrecision == 0 ? 1 : fault.addressPrecision);
	}
	for (uint32_t index = 0; index < counts.vendorInfoCount && index < vendors.size(); ++index) {
		auto& vendor = vendors[index];
		vendor.description[VK_MAX_DESCRIPTION_SIZE - 1] = 0;
		LOGF("[lab-fault] vendor %u: %s code=0x%" PRIx64 " data=0x%" PRIx64 "\n", index,
		     vendor.description, static_cast<uint64_t>(vendor.vendorFaultCode),
		     static_cast<uint64_t>(vendor.vendorFaultData));
	}
}

void RequireVulkanSuccess(vk::Result result, const char* operation) {
	if (result != vk::Result::eSuccess) {
		LabDeviceFaultReport(result);
		EXIT("%s failed: %s (%d)\n", operation, vk::to_string(result).c_str(),
		     static_cast<int>(result));
	}
}

vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device) {
	vk::ShaderModuleCreateInfo create_info {};
	create_info.codeSize    = code.size_bytes();
	create_info.pCode       = code.data();
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
	                     "create SPIR-V shader module");
	return module;
}

vk::Pipeline CreateComputePipeline(vk::Device device, std::span<const uint32_t> code,
                                   vk::PipelineLayout layout, const char* operation) {
	const auto                    shader = CompileSPV(code, device);
	vk::ComputePipelineCreateInfo create {};
	create.stage.stage    = vk::ShaderStageFlagBits::eCompute;
	create.stage.module   = shader;
	create.stage.pName    = "main";
	create.layout         = layout;
	vk::Pipeline pipeline = nullptr;
	const auto   result   = device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
	device.destroyShaderModule(shader, nullptr);
	RequireVulkanSuccess(result, operation);
	return pipeline;
}

} // namespace Libs::Graphics
