#include "graphics/host_gpu/renderer/deviceLossDiag.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace Libs::Graphics::DeviceLossDiag {

namespace {

constexpr uint32_t RingSize       = 1u << 16u;
constexpr uint32_t ContextEntries = 24;

struct Entry {
	std::atomic<uint32_t> id {0};
	Kind                  kind  = Kind::Draw;
	uint64_t              hash0 = 0;
	uint64_t              hash1 = 0;
	uint64_t              tick  = 0;
	const char*           site  = nullptr;
};

GraphicContext*             g_graphics = nullptr;
std::array<Entry, RingSize> g_ring;
std::atomic<uint32_t>       g_next_id {0};
std::atomic<bool>           g_reported {false};
vk::Buffer                  g_marker_buffer = nullptr;
vk::DeviceMemory            g_marker_memory = nullptr;
volatile uint32_t*          g_markers       = nullptr;

constexpr vk::DeviceSize StartedOffset  = 0;
constexpr vk::DeviceSize FinishedOffset = 4;

const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::Draw: return "draw";
		case Kind::DrawIndirect: return "draw-indirect";
		case Kind::Mesh: return "mesh";
		case Kind::Dispatch: return "dispatch";
		case Kind::DispatchIndirect: return "dispatch-indirect";
		case Kind::Internal: return "kyty-internal";
	}
	return "?";
}

void Write(const std::string& text) {
	Log::WriteToConsoleAndLog(text);
}

std::string Describe(uint32_t id) {
	if (id == 0) {
		return "none";
	}
	const auto& e = g_ring[id % RingSize];
	if (e.id.load(std::memory_order_acquire) != id) {
		return fmt::format("#{} (overwritten)", id);
	}
	if (e.kind == Kind::Internal) {
		return fmt::format("#{} {} site={}", id, KindName(e.kind),
		                   e.site != nullptr ? e.site : "?");
	}
	return fmt::format("#{} {} tick={} hash0=0x{:016x} hash1=0x{:016x}", id, KindName(e.kind),
	                   e.tick, e.hash0, e.hash1);
}

void CreateMarkerBuffer(GraphicContext& graphics) {
	vk::BufferCreateInfo info {};
	info.size        = 64;
	info.usage       = vk::BufferUsageFlagBits::eTransferDst;
	info.sharingMode = vk::SharingMode::eExclusive;
	if (graphics.device.createBuffer(&info, nullptr, &g_marker_buffer) != vk::Result::eSuccess) {
		g_marker_buffer = nullptr;
		return;
	}
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(g_marker_buffer, &requirements);
	vk::PhysicalDeviceMemoryProperties memory {};
	graphics.physical_device.getMemoryProperties(&memory);
	const auto wanted =
	    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
			type = i;
			break;
		}
	}
	vk::MemoryAllocateInfo allocate {};
	allocate.allocationSize  = requirements.size;
	allocate.memoryTypeIndex = type;
	void* mapped             = nullptr;
	if (type == UINT32_MAX ||
	    graphics.device.allocateMemory(&allocate, nullptr, &g_marker_memory) !=
	        vk::Result::eSuccess ||
	    graphics.device.bindBufferMemory(g_marker_buffer, g_marker_memory, 0) !=
	        vk::Result::eSuccess ||
	    graphics.device.mapMemory(g_marker_memory, 0, VK_WHOLE_SIZE, {}, &mapped) !=
	        vk::Result::eSuccess) {
		graphics.device.destroyBuffer(g_marker_buffer, nullptr);
		g_marker_buffer = nullptr;
		return;
	}
	g_markers    = static_cast<volatile uint32_t*>(mapped);
	g_markers[0] = 0;
	g_markers[1] = 0;
}

void ReportFault() {
	if (!g_graphics->device_fault_enabled) {
		Write("[DEVLOST] VK_EXT_device_fault: not available on this device/driver\n");
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	auto                     result = g_graphics->device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		Write(
		    fmt::format("[DEVLOST] vkGetDeviceFaultInfoEXT(counts): {}\n", vk::to_string(result)));
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	result             = g_graphics->device.getFaultInfoEXT(&counts, &info);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		Write(fmt::format("[DEVLOST] vkGetDeviceFaultInfoEXT(info): {}\n", vk::to_string(result)));
		return;
	}
	Write(fmt::format("[DEVLOST] device fault: \"{}\" addresses={} vendor_infos={}\n",
	                  info.description.data(), counts.addressInfoCount, counts.vendorInfoCount));
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& a = addresses[i];
		Write(fmt::format("[DEVLOST]   address[{}] {} reported=0x{:016x} precision=0x{:x}\n", i,
		                  vk::to_string(a.addressType), a.reportedAddress, a.addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& v = vendors[i];
		Write(fmt::format("[DEVLOST]   vendor[{}] \"{}\" code=0x{:016x} data=0x{:016x}\n", i,
		                  v.description.data(), v.vendorFaultCode, v.vendorFaultData));
	}
}

void ReportNvCheckpoints() {
	uint32_t count = 0;
	g_graphics->queue.getCheckpointDataNV(&count, nullptr);
	std::vector<vk::CheckpointDataNV> data(count);
	g_graphics->queue.getCheckpointDataNV(&count, data.data());
	Write(fmt::format("[DEVLOST] NV checkpoints: {}\n", count));
	for (uint32_t i = 0; i < count; i++) {
		const auto marker = reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker);
		const auto id     = static_cast<uint32_t>(marker >> 1u);
		Write(fmt::format("[DEVLOST]   stage {} reached {} of {}\n", vk::to_string(data[i].stage),
		                  (marker & 1u) != 0 ? "END" : "BEGIN", Describe(id)));
	}
}

} // namespace

void Init(GraphicContext& graphics) {
	g_graphics = &graphics;
	if (graphics.amd_buffer_marker_enabled) {
		CreateMarkerBuffer(graphics);
	}
	Write(fmt::format("[DEVLOST] device-loss diagnostics: device_fault={} nv_checkpoints={} "
	                  "amd_buffer_marker={}\n",
	                  graphics.device_fault_enabled, graphics.nv_checkpoints_enabled,
	                  g_markers != nullptr));
}

static uint32_t Record(vk::CommandBuffer buffer, Kind kind, uint64_t hash0, uint64_t hash1,
                       uint64_t tick, const char* site) {
	if (g_graphics == nullptr) {
		return 0;
	}
	uint32_t id = g_next_id.fetch_add(1, std::memory_order_relaxed) + 1;
	if (id == 0) {
		id = g_next_id.fetch_add(1, std::memory_order_relaxed) + 1;
	}
	auto& e = g_ring[id % RingSize];
	e.id.store(0, std::memory_order_relaxed);
	e.kind  = kind;
	e.hash0 = hash0;
	e.hash1 = hash1;
	e.tick  = tick;
	e.site  = site;
	e.id.store(id, std::memory_order_release);
	if (g_graphics->nv_checkpoints_enabled) {
		buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(id) << 1u));
	}
	if (g_markers != nullptr) {
		buffer.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_marker_buffer,
		                            StartedOffset, id);
	}
	return id;
}

uint32_t Begin(vk::CommandBuffer buffer, Kind kind, uint64_t hash0, uint64_t hash1, uint64_t tick) {
	return Record(buffer, kind, hash0, hash1, tick, nullptr);
}

uint32_t BeginInternal(vk::CommandBuffer buffer, const char* site) {
	return Record(buffer, Kind::Internal, 0, 0, 0, site);
}

void End(vk::CommandBuffer buffer, uint32_t id) {
	if (g_graphics == nullptr || id == 0) {
		return;
	}
	if (g_graphics->nv_checkpoints_enabled) {
		buffer.setCheckpointNV(
		    reinterpret_cast<const void*>((static_cast<uintptr_t>(id) << 1u) | 1u));
	}
	if (g_markers != nullptr) {
		buffer.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_marker_buffer,
		                            FinishedOffset, id);
	}
}

void Report(const char* where, vk::Result result, uint64_t known_gpu_tick) {
	if (g_graphics == nullptr || g_reported.exchange(true)) {
		return;
	}
	const auto last = g_next_id.load(std::memory_order_relaxed);
	Write(fmt::format("[DEVLOST] ===== {} returned {} ; last known finished tick={} ; recorded "
	                  "commands={} =====\n",
	                  where, vk::to_string(result), known_gpu_tick, last));
	ReportFault();
	uint32_t focus = 0;
	if (g_graphics->nv_checkpoints_enabled) {
		ReportNvCheckpoints();
	}
	if (g_markers != nullptr) {
		const uint32_t started  = g_markers[0];
		const uint32_t finished = g_markers[1];
		Write(fmt::format("[DEVLOST] AMD markers: last started {} | last finished {}\n",
		                  Describe(started), Describe(finished)));
		focus = finished + 1;
	}
	if (focus == 0 || focus > last) {
		focus = last > ContextEntries / 2 ? last - ContextEntries / 2 : 1;
	}
	const uint32_t from = focus > ContextEntries / 2 ? focus - ContextEntries / 2 : 1;
	const uint32_t to   = std::min(last, from + ContextEntries - 1);
	Write(fmt::format("[DEVLOST] recorded commands #{}..#{}:\n", from, to));
	for (uint32_t id = from; id <= to && id != 0; id++) {
		Write(fmt::format("[DEVLOST]   {}{}\n", Describe(id),
		                  id == focus ? "   <== first unfinished" : ""));
	}
	Write("[DEVLOST] ===== end =====\n");
}

} // namespace Libs::Graphics::DeviceLossDiag
