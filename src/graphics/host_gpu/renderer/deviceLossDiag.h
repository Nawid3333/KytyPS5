#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

namespace DeviceLossDiag {

enum class Kind : uint8_t { Draw, DrawIndirect, Mesh, Dispatch, DispatchIndirect, Internal };

struct IndirectArgs {
	vk::Buffer              buffer;
	vk::DeviceSize          offset        = 0;
	uint32_t                size          = 0;
	uint64_t                guest_address = 0;
	bool                    cpu_read      = false;
	std::array<uint32_t, 3> cpu_words {};
	bool                    gpu_modified = false;
	bool                    cpu_modified = false;
	bool                    image        = false;
};

enum class WriteSource : uint8_t { Bound, Fill, CpuFill, Copy, CpuCopy, WriteData, Label };

void     Init(GraphicContext& graphics);
void     NoteWrite(WriteSource source, uint64_t vaddr, uint64_t size, uint64_t value = 0);
uint32_t Begin(vk::CommandBuffer buffer, Kind kind, uint64_t hash0, uint64_t hash1, uint64_t tick,
               const IndirectArgs& args = {});
uint32_t BeginInternal(vk::CommandBuffer buffer, const char* site);
void     End(vk::CommandBuffer buffer, uint32_t id);
void     Report(const char* where, vk::Result result, uint64_t known_gpu_tick);

} // namespace DeviceLossDiag

} // namespace Libs::Graphics
