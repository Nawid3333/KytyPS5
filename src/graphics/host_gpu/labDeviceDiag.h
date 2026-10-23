#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_LABDEVICEDIAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_LABDEVICEDIAG_H_

// Lab (#1511, not for upstream): KYTY_LAB_DEVICE_DIAG=1 enables VK_EXT_device_fault and
// VK_AMD_buffer_marker. Every guest dispatch and draw, helper blit and flip copy (render stream, slot 0)
// and every presenter command buffer (present stream, slot 1; same queue) writes its serial before it
// (top of pipe) and after it (bottom of pipe) into a host buffer; a lost device logs the fault
// records and, per stream, the commands that started but did not finish.

#include "common/logging/log.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::LabDeviceDiag {

inline bool Requested() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LAB_DEVICE_DIAG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

struct Dispatch {
	uint64_t hash  = 0; // CS, or VS/mesh for draws
	uint64_t hash2 = 0; // PS for draws
	uint32_t x     = 0; // groups, or vertex/index count and instances for draws
	uint32_t y     = 0;
	uint32_t z     = 0;
	uint32_t kind  = 0; // 1 dispatch, 2 dispatch indirect, 3 ordered dispatchBase loop, 4 draw,
	                    // 5 mesh draw, 6 draw indirect, 7 helper blit, 9 present commands, 10 flip copy
	uint32_t slot  = 0; // 0 render stream, 1 present stream
};

struct State {
	bool                      fault  = false;
	vk::Device                device = nullptr;
	vk::Buffer                buffer = nullptr;
	vk::DeviceMemory          memory = nullptr;
	volatile uint32_t*        mapped = nullptr;
	std::atomic<uint32_t>     serial {0};
	std::array<Dispatch, 4096> ring {};
};

inline State& Get() {
	static State state;
	return state;
}

// Call after the device dispatcher is initialized. fault: VK_EXT_device_fault and its feature are on.
inline void Init(vk::PhysicalDevice physical, vk::Device device, bool fault, bool marker) {
	auto& state  = Get();
	state.fault  = fault;
	state.device = device;
	if (marker) {
		vk::BufferCreateInfo info {};
		info.size        = 16;
		info.usage       = vk::BufferUsageFlagBits::eTransferDst;
		info.sharingMode = vk::SharingMode::eExclusive;
		vk::MemoryRequirements requirements {};
		vk::PhysicalDeviceMemoryProperties properties {};
		physical.getMemoryProperties(&properties);
		const auto wanted =
		    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
		if (device.createBuffer(&info, nullptr, &state.buffer) == vk::Result::eSuccess) {
			device.getBufferMemoryRequirements(state.buffer, &requirements);
			for (uint32_t index = 0; index < properties.memoryTypeCount; index++) {
				if ((requirements.memoryTypeBits & (1u << index)) == 0 ||
				    (properties.memoryTypes[index].propertyFlags & wanted) != wanted) {
					continue;
				}
				vk::MemoryAllocateInfo allocate {};
				allocate.allocationSize  = requirements.size;
				allocate.memoryTypeIndex = index;
				void* pointer            = nullptr;
				if (device.allocateMemory(&allocate, nullptr, &state.memory) == vk::Result::eSuccess &&
				    device.bindBufferMemory(state.buffer, state.memory, 0) == vk::Result::eSuccess &&
				    device.mapMemory(state.memory, 0, VK_WHOLE_SIZE, {}, &pointer) ==
				        vk::Result::eSuccess) {
					state.mapped = static_cast<volatile uint32_t*>(pointer);
					state.mapped[0] = state.mapped[1] = state.mapped[2] = state.mapped[3] = 0;
				}
				break;
			}
		}
	}
	LOGF("Lab device diag (KYTY_LAB_DEVICE_DIAG): requested=%s device_fault=%s buffer_marker=%s\n",
	     Requested() ? "true" : "false", fault ? "true" : "false",
	     state.mapped != nullptr ? "true" : "false");
}

// Before a guest dispatch; returns the serial for End (0 = markers off).
inline uint32_t Begin(vk::CommandBuffer cmd, uint64_t hash, uint32_t x, uint32_t y, uint32_t z,
                      uint32_t kind, uint64_t hash2 = 0, uint32_t slot = 0) {
	auto& state = Get();
	if (state.mapped == nullptr) {
		return 0;
	}
	const auto serial = state.serial.fetch_add(1, std::memory_order_relaxed) + 1;
	state.ring[serial % state.ring.size()] = {hash, hash2, x, y, z, kind, slot};
	cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, state.buffer, slot * 8u, serial);
	return serial;
}

inline void End(vk::CommandBuffer cmd, uint32_t serial, uint32_t slot = 0) {
	if (serial != 0) {
		cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, Get().buffer, slot * 8u + 4u,
		                         serial);
	}
}

inline void Report(const char* where) {
	auto& state = Get();
	LOGF("Lab device diag: device lost in %s\n", where);
	if (state.fault) {
		vk::DeviceFaultCountsEXT counts {};
		const auto counted = state.device.getFaultInfoEXT(&counts, nullptr);
		std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
		std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
		vk::DeviceFaultInfoEXT                      info {};
		info.pAddressInfos     = addresses.data();
		info.pVendorInfos      = vendors.data();
		counts.vendorBinarySize = 0;
		const auto read = state.device.getFaultInfoEXT(&counts, &info);
		LOGF("Lab device diag: fault counts=%s info=%s addresses=%u vendor=%u description=\"%s\"\n",
		     vk::to_string(counted).c_str(), vk::to_string(read).c_str(), counts.addressInfoCount,
		     counts.vendorInfoCount, info.description.data());
		for (uint32_t i = 0; i < counts.addressInfoCount && i < addresses.size(); i++) {
			LOGF("Lab device diag:   address[%u] type=%s address=0x%016" PRIx64
			     " precision=0x%" PRIx64 "\n",
			     i, vk::to_string(addresses[i].addressType).c_str(),
			     static_cast<uint64_t>(addresses[i].reportedAddress),
			     static_cast<uint64_t>(addresses[i].addressPrecision));
		}
		for (uint32_t i = 0; i < counts.vendorInfoCount && i < vendors.size(); i++) {
			LOGF("Lab device diag:   vendor[%u] code=0x%016" PRIx64 " data=0x%016" PRIx64 " \"%s\"\n",
			     i, static_cast<uint64_t>(vendors[i].vendorFaultCode),
			     static_cast<uint64_t>(vendors[i].vendorFaultData), vendors[i].description.data());
		}
	}
	if (state.mapped != nullptr) {
		const uint32_t last = state.serial.load(std::memory_order_relaxed);
		for (uint32_t slot = 0; slot < 2u; slot++) {
			const uint32_t top    = state.mapped[slot * 2u];
			const uint32_t bottom = state.mapped[slot * 2u + 1u];
			LOGF("Lab device diag: %s markers started=%u finished=%u recorded=%u\n",
			     slot == 0 ? "render stream" : "present stream", top, bottom, last);
			uint32_t shown = 0;
			uint32_t after = 0;
			for (uint32_t serial = bottom > 2u ? bottom - 2u : 1u; serial <= last && shown < 24u && after < 3u;
			     serial++) {
				const auto& entry = state.ring[serial % state.ring.size()];
				if (entry.slot != slot || serial + state.ring.size() <= last) {
					continue; // other stream, or overwritten in the ring
				}
				shown++;
				after += serial > top ? 1u : 0u;
				LOGF("Lab device diag:   command %u %s kind=%u hash=0x%016" PRIx64 " hash2=0x%016" PRIx64
				     " size=%ux%ux%u\n",
				     serial,
				     serial <= bottom ? "finished" : (serial <= top ? "STARTED, NOT FINISHED" : "not started"),
				     entry.kind, entry.hash, entry.hash2, entry.x, entry.y, entry.z);
			}
		}
	}
}

} // namespace Libs::Graphics::LabDeviceDiag

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_LABDEVICEDIAG_H_ */
