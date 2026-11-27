#include "graphics/host_gpu/renderer/deviceLossDiag.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace Libs::Graphics::DeviceLossDiag {

namespace {

constexpr uint32_t RingSize          = 1u << 16u;
constexpr uint32_t ContextEntries    = 24;
constexpr uint32_t ArgsSlots         = 4096;
constexpr uint32_t ArgsSlotWords     = 80;
constexpr uint32_t ArgsWindowBefore  = 0x120;
constexpr uint32_t WindowLineWords   = 8;
constexpr uint32_t ArgsReportedEarly = 8;
constexpr uint32_t SameShaderArgs    = 24;
constexpr uint32_t WriteRingSize     = 1u << 20u;
constexpr uint32_t WritersReported   = 64;
constexpr uint32_t WindowCommands    = 4096;
constexpr uint32_t CpRingSize        = 1u << 16u;
constexpr uint32_t BindingsReported  = 32;
constexpr uint32_t BindingWriters    = 4;
constexpr uint32_t ListingsReported  = 4;

struct Entry {
	std::atomic<uint32_t> id {0};
	Kind                  kind  = Kind::Draw;
	uint64_t              hash0 = 0;
	uint64_t              hash1 = 0;
	uint64_t              tick  = 0;
	const char*           site  = nullptr;
	IndirectArgs          args;
	uint32_t              args_slot = 0;
	std::array<uint32_t, 3> groups {};
};

struct WriteEntry {
	std::atomic<uint64_t> serial {0};
	uint32_t              next_command = 0;
	WriteSource           source       = WriteSource::Bound;
	uint64_t              vaddr        = 0;
	uint64_t              size         = 0;
	uint64_t              value        = 0;
};

struct CpEntry {
	std::atomic<uint64_t> serial {0};
	uint32_t              next_command = 0;
	CpEvent               event        = CpEvent::CondExecRun;
	uint64_t              address      = 0;
	uint64_t              value        = 0;
	uint64_t              count        = 0;
};

GraphicContext*             g_graphics = nullptr;
std::vector<CpEntry>        g_cp(CpRingSize);
std::atomic<uint64_t>       g_next_cp {0};
std::vector<WriteEntry>     g_writes(WriteRingSize);
std::atomic<uint64_t>       g_next_write {0};
std::array<Entry, RingSize> g_ring;
std::atomic<uint32_t>       g_next_id {0};
std::atomic<uint32_t>       g_next_args_slot {0};
std::atomic<bool>           g_reported {false};
vk::Buffer                  g_marker_buffer = nullptr;
vk::DeviceMemory            g_marker_memory = nullptr;
volatile uint32_t*          g_markers       = nullptr;
vk::Buffer                  g_args_buffer   = nullptr;
vk::DeviceMemory            g_args_memory   = nullptr;
volatile uint32_t*          g_args          = nullptr;

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

const char* SourceName(WriteSource source) {
	switch (source) {
		case WriteSource::Bound: return "bound-written";
		case WriteSource::Fill: return "fill-gpu";
		case WriteSource::CpuFill: return "fill-cpu";
		case WriteSource::Copy: return "copy-gpu";
		case WriteSource::CpuCopy: return "copy-cpu";
		case WriteSource::WriteData: return "write-data";
		case WriteSource::Label: return "eop-label";
		case WriteSource::BoundRead: return "bound-read";
	}
	return "?";
}

const char* CpEventName(CpEvent event) {
	switch (event) {
		case CpEvent::CondExecRun: return "COND_EXEC run";
		case CpEvent::CondExecSkip: return "COND_EXEC skip";
		case CpEvent::BranchThen: return "BRANCH then";
		case CpEvent::BranchElse: return "BRANCH else";
		case CpEvent::BranchNone: return "BRANCH none";
		case CpEvent::Predicate: return "SET_PREDICATION";
		case CpEvent::PredicatedSkip: return "predicated packet skipped";
		case CpEvent::DispatchDropped: return "dispatch dropped";
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
	auto text = fmt::format("#{} {} tick={} hash0=0x{:016x} hash1=0x{:016x}", id, KindName(e.kind),
	                        e.tick, e.hash0, e.hash1);
	if (e.groups[0] != 0 || e.groups[1] != 0 || e.groups[2] != 0) {
		text += fmt::format(" groups={}x{}x{}", e.groups[0], e.groups[1], e.groups[2]);
	}
	return text;
}

volatile uint32_t* CreateHostBuffer(GraphicContext& graphics, vk::DeviceSize size,
                                    vk::Buffer& buffer, vk::DeviceMemory& memory_out) {
	vk::BufferCreateInfo info {};
	info.size        = size;
	info.usage       = vk::BufferUsageFlagBits::eTransferDst;
	info.sharingMode = vk::SharingMode::eExclusive;
	if (graphics.device.createBuffer(&info, nullptr, &buffer) != vk::Result::eSuccess) {
		buffer = nullptr;
		return nullptr;
	}
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(buffer, &requirements);
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
	    graphics.device.allocateMemory(&allocate, nullptr, &memory_out) != vk::Result::eSuccess ||
	    graphics.device.bindBufferMemory(buffer, memory_out, 0) != vk::Result::eSuccess ||
	    graphics.device.mapMemory(memory_out, 0, VK_WHOLE_SIZE, {}, &mapped) !=
	        vk::Result::eSuccess) {
		graphics.device.destroyBuffer(buffer, nullptr);
		buffer = nullptr;
		return nullptr;
	}
	std::memset(mapped, 0, static_cast<size_t>(size));
	return static_cast<volatile uint32_t*>(mapped);
}

vk::DeviceSize WindowBefore(const IndirectArgs& args) {
	return args.guest_mapped ? std::min<vk::DeviceSize>(args.offset, ArgsWindowBefore) : 0;
}

void CaptureArgs(vk::CommandBuffer buffer, uint32_t id, uint32_t args_slot,
                 const IndirectArgs& args) {
	const vk::DeviceSize    slot = vk::DeviceSize {args_slot} * ArgsSlotWords * sizeof(uint32_t);
	const vk::MemoryBarrier before {.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
	                                                 vk::AccessFlagBits::eTransferWrite,
	                                .dstAccessMask = vk::AccessFlagBits::eTransferRead};
	buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader |
	                           vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 1, &before,
	                       0, nullptr, 0, nullptr);
	buffer.fillBuffer(g_args_buffer, slot, sizeof(uint32_t), id);
	const vk::DeviceSize window = WindowBefore(args);
	const vk::BufferCopy copy {
	    args.offset - window, slot + sizeof(uint32_t),
	    std::min<vk::DeviceSize>(window + args.size, (ArgsSlotWords - 1) * sizeof(uint32_t))};
	buffer.copyBuffer(args.buffer, g_args_buffer, 1, &copy);
	buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eDrawIndirect, vk::DependencyFlags {}, 0,
	                       nullptr, 0, nullptr, 0, nullptr);
}

bool HasArgs(uint32_t id) {
	const auto& e = g_ring[id % RingSize];
	return e.id.load(std::memory_order_acquire) == id && e.args.buffer;
}

std::string DescribeArgs(uint32_t id) {
	const auto& e = g_ring[id % RingSize];
	const auto& a = e.args;
	const auto  handle =
	    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkBuffer>(a.buffer)));
	auto text =
	    fmt::format("#{} {} hash0=0x{:016x} guest=0x{:016x} buffer=0x{:016x}+0x{:x} words=", id,
	                KindName(e.kind), e.hash0, a.guest_address, handle, a.offset);
	if (g_args == nullptr) {
		return text + "not captured";
	}
	const auto*    slot   = g_args + vk::DeviceSize {e.args_slot} * ArgsSlotWords;
	const uint32_t holder = slot[0];
	if (holder != id) {
		return text + fmt::format("not copied (slot holds #{})", holder);
	}
	const auto     first = static_cast<uint32_t>(WindowBefore(a) / sizeof(uint32_t));
	const uint32_t words = std::min<uint32_t>(a.size / sizeof(uint32_t), ArgsSlotWords - 1 - first);
	for (uint32_t i = 0; i < words; i++) {
		const uint32_t word = slot[1 + first + i];
		text += fmt::format("{}0x{:08x}", i == 0 ? "" : " ", word);
	}
	text += a.cpu_read ? fmt::format(" | guest memory at record 0x{:08x} 0x{:08x} 0x{:08x}",
	                                 a.cpu_words[0], a.cpu_words[1], a.cpu_words[2])
	                   : " | guest memory at record unreadable";
	text += fmt::format(" gpu_modified={} cpu_modified={} image={}", a.gpu_modified,
	                    a.cpu_modified, a.image);
	return text;
}

void ReportArgsWindow(uint32_t id) {
	if (g_args == nullptr) {
		return;
	}
	const auto& e = g_ring[id % RingSize];
	if (!e.args.guest_mapped) {
		Write(fmt::format(
		    "[DEVLOST]     no guest window for #{}: its arguments were read through the "
		    "stream ring\n",
		    id));
		return;
	}
	const auto* slot = g_args + vk::DeviceSize {e.args_slot} * ArgsSlotWords;
	if (slot[0] != id) {
		return;
	}
	const vk::DeviceSize before = WindowBefore(e.args);
	const auto           words  = static_cast<uint32_t>(
	    std::min<vk::DeviceSize>((before + e.args.size) / sizeof(uint32_t), ArgsSlotWords - 1));
	Write(fmt::format("[DEVLOST]     GPU words of #{} from guest 0x{:016x} ({} bytes before the "
	                  "arguments):\n",
	                  id, e.args.guest_address - before, before));
	for (uint32_t i = 0; i < words; i += WindowLineWords) {
		auto line = fmt::format("[DEVLOST]     +0x{:03x}:", i * sizeof(uint32_t));
		for (uint32_t j = i; j < std::min(words, i + WindowLineWords); j++) {
			line += fmt::format(" {:08x}", slot[1 + j]);
		}
		Write(line + "\n");
	}
}

void ReportIndirectArgs(uint32_t focus) {
	Write(fmt::format("[DEVLOST] dispatch-indirect arguments copied on the GPU before each "
	                  "dispatch-indirect ({} slots):\n",
	                  ArgsSlots));
	if (HasArgs(focus)) {
		Write(fmt::format("[DEVLOST]   {}   <== first unfinished\n", DescribeArgs(focus)));
		ReportArgsWindow(focus);
	} else {
		Write(fmt::format("[DEVLOST]   #{} has no indirect arguments   <== first unfinished\n",
		                  focus));
	}
	uint32_t shown = 0;
	for (uint32_t id = focus - 1; id != 0 && shown < ArgsReportedEarly && focus - id < RingSize;
	     id--) {
		if (g_ring[id % RingSize].id.load(std::memory_order_acquire) != id) {
			break;
		}
		if (HasArgs(id)) {
			Write(fmt::format("[DEVLOST]   {}\n", DescribeArgs(id)));
			shown++;
		}
	}
	if (!HasArgs(focus)) {
		return;
	}
	const uint64_t hash    = g_ring[focus % RingSize].hash0;
	const uint64_t address = g_ring[focus % RingSize].args.guest_address;
	Write(fmt::format("[DEVLOST] earlier dispatch-indirects of shader 0x{:016x} (newest first):\n",
	                  hash));
	shown             = 0;
	bool window_shown = false;
	for (uint32_t id = focus - 1; id != 0 && shown < SameShaderArgs && focus - id < RingSize;
	     id--) {
		const auto& e = g_ring[id % RingSize];
		if (e.id.load(std::memory_order_acquire) != id) {
			break;
		}
		if (e.hash0 == hash && HasArgs(id)) {
			Write(fmt::format("[DEVLOST]   tick={} {}\n", e.tick, DescribeArgs(id)));
			if (!window_shown && e.args.guest_address == address) {
				ReportArgsWindow(id);
				window_shown = true;
			}
			shown++;
		}
	}
	if (shown == 0) {
		Write("[DEVLOST]   none in the recorded commands\n");
	}
}

void ReportListing(uint64_t hash, const char* role) {
	const auto code = ShaderFindCodeByHash(hash);
	if (code.empty()) {
		Write(fmt::format("[DEVLOST] {} shader hash=0x{:016x}: code not in the shader map\n", role,
		                  hash));
		return;
	}
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
	Write(fmt::format("[DEVLOST] {} shader hash=0x{:016x} address=0x{:016x} listing:\n{}\n", role,
	                  hash, reinterpret_cast<uint64_t>(code.data()),
	                  ShaderRecompiler::Decoder::ProgramToString(decoded)));
}

void ReportWriters(uint32_t focus) {
	if (!HasArgs(focus)) {
		return;
	}
	const auto& entry = g_ring[focus % RingSize];
	const auto  begin = entry.args.guest_mapped && entry.args.guest_address >= ArgsWindowBefore
	                        ? entry.args.guest_address - ArgsWindowBefore
	                        : entry.args.guest_address;
	const auto  end   = entry.args.guest_address + entry.args.size;
	Write(fmt::format("[DEVLOST] writes recorded over 0x{:016x}..0x{:016x} (newest first, ring of "
	                  "{}):\n",
	                  begin, end, WriteRingSize));
	const uint64_t        last  = g_next_write.load(std::memory_order_acquire);
	const uint64_t        first = last > WriteRingSize ? last - WriteRingSize : 0;
	std::vector<uint64_t> listed;
	uint32_t              shown = 0;
	for (uint64_t serial = last; serial > first && shown < WritersReported; serial--) {
		const auto& w = g_writes[serial % WriteRingSize];
		if (w.serial.load(std::memory_order_acquire) != serial || w.source == WriteSource::BoundRead ||
		    w.vaddr >= end || w.vaddr + w.size <= begin) {
			continue;
		}
		const auto command = w.next_command;
		Write(fmt::format("[DEVLOST]   {} 0x{:016x}+0x{:x} value=0x{:x} before command #{} | next "
		                  "recorded {}\n",
		                  SourceName(w.source), w.vaddr, w.size, w.value, command,
		                  Describe(command)));
		shown++;
		const auto& next = g_ring[command % RingSize];
		if (w.source == WriteSource::Bound && next.id.load(std::memory_order_acquire) == command &&
		    next.kind != Kind::Internal && listed.size() < ListingsReported &&
		    std::find(listed.begin(), listed.end(), next.hash0) == listed.end()) {
			listed.push_back(next.hash0);
		}
	}
	if (shown == 0) {
		Write("[DEVLOST]   none\n");
	}
	ReportListing(entry.hash0, "dispatch-indirect");
	for (const auto hash : listed) {
		if (hash != entry.hash0) {
			ReportListing(hash, "writer");
		}
	}
}

void ReportWindow(uint32_t focus) {
	const auto& entry = g_ring[focus % RingSize];
	if (entry.id.load(std::memory_order_acquire) != focus || entry.kind == Kind::Internal) {
		return;
	}
	uint32_t previous  = 0;
	uint32_t bind_from = 0;
	for (uint32_t id = focus - 1; id != 0 && focus - id < std::min(RingSize, WindowCommands); id--) {
		const auto& e = g_ring[id % RingSize];
		if (e.id.load(std::memory_order_acquire) != id) {
			break;
		}
		if (bind_from == 0 && e.kind != Kind::Internal) {
			bind_from = id + 1;
		}
		if (e.kind == entry.kind && e.hash0 == entry.hash0) {
			previous = id;
			break;
		}
	}
	if (bind_from == 0) {
		bind_from = focus;
	}
	const uint32_t from =
	    previous != 0 ? previous : (focus > WindowCommands ? focus - WindowCommands : 1);
	Write(fmt::format("[DEVLOST] commands #{}..#{} ({}):\n", from, focus,
	                  previous != 0 ? "since the previous run of this shader"
	                                : "no earlier run of this shader in the window"));
	for (uint32_t id = from; id <= focus && id != 0; id++) {
		Write(fmt::format("[DEVLOST]   {}\n", Describe(id)));
	}
	Write(fmt::format("[DEVLOST] command processor decisions before commands #{}..#{}:\n", from + 1,
	                  focus));
	const uint64_t last_cp  = g_next_cp.load(std::memory_order_acquire);
	const uint64_t first_cp = last_cp > CpRingSize ? last_cp - CpRingSize : 0;
	uint32_t       shown    = 0;
	for (uint64_t serial = first_cp + 1; serial <= last_cp; serial++) {
		const auto& c = g_cp[serial % CpRingSize];
		if (c.serial.load(std::memory_order_acquire) != serial || c.next_command <= from ||
		    c.next_command > focus) {
			continue;
		}
		Write(fmt::format("[DEVLOST]   cp {} address=0x{:016x} value=0x{:x} count={} before command "
		                  "#{}\n",
		                  CpEventName(c.event), c.address, c.value, c.count, c.next_command));
		shown++;
	}
	if (shown == 0) {
		Write("[DEVLOST]   none\n");
	}
	Write(fmt::format("[DEVLOST] buffers bound by #{} and their newest writers:\n", focus));
	const uint64_t last     = g_next_write.load(std::memory_order_acquire);
	const uint64_t first    = last > WriteRingSize ? last - WriteRingSize : 0;
	uint32_t       bindings = 0;
	for (uint64_t serial = last; serial > first && bindings < BindingsReported; serial--) {
		const auto& b = g_writes[serial % WriteRingSize];
		if (b.serial.load(std::memory_order_acquire) != serial || b.next_command < bind_from ||
		    b.next_command > focus ||
		    (b.source != WriteSource::Bound && b.source != WriteSource::BoundRead)) {
			continue;
		}
		Write(fmt::format("[DEVLOST]   {} 0x{:016x}+0x{:x}\n", SourceName(b.source), b.vaddr,
		                  b.size));
		bindings++;
		uint32_t writers = 0;
		for (uint64_t s = serial - 1; s > first && writers < BindingWriters; s--) {
			const auto& w = g_writes[s % WriteRingSize];
			if (w.serial.load(std::memory_order_acquire) != s ||
			    w.source == WriteSource::BoundRead || w.next_command >= bind_from ||
			    w.vaddr >= b.vaddr + b.size || w.vaddr + w.size <= b.vaddr) {
				continue;
			}
			Write(fmt::format("[DEVLOST]     {} 0x{:016x}+0x{:x} value=0x{:x} before command #{} | "
			                  "{}\n",
			                  SourceName(w.source), w.vaddr, w.size, w.value, w.next_command,
			                  Describe(w.next_command)));
			writers++;
		}
		if (writers == 0) {
			Write("[DEVLOST]     no write in the ring\n");
		}
	}
	if (bindings == 0) {
		Write("[DEVLOST]   none recorded\n");
	}
	if (!HasArgs(focus)) {
		ReportListing(entry.hash0, KindName(entry.kind));
	}
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
		g_markers = CreateHostBuffer(graphics, 64, g_marker_buffer, g_marker_memory);
	}
	g_args =
	    CreateHostBuffer(graphics, vk::DeviceSize {ArgsSlots} * ArgsSlotWords * sizeof(uint32_t),
	                     g_args_buffer, g_args_memory);
	Write(fmt::format("[DEVLOST] device-loss diagnostics: device_fault={} nv_checkpoints={} "
	                  "amd_buffer_marker={} indirect_args={}\n",
	                  graphics.device_fault_enabled, graphics.nv_checkpoints_enabled,
	                  g_markers != nullptr, g_args != nullptr));
}

static uint32_t Record(vk::CommandBuffer buffer, Kind kind, uint64_t hash0, uint64_t hash1,
                       uint64_t tick, const char* site, const IndirectArgs& args,
                       const std::array<uint32_t, 3>& groups) {
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
	e.args  = args;
	e.groups = groups;
	if (args.buffer) {
		e.args_slot = g_next_args_slot.fetch_add(1, std::memory_order_relaxed) % ArgsSlots;
	}
	e.id.store(id, std::memory_order_release);
	if (g_args != nullptr && args.buffer) {
		CaptureArgs(buffer, id, e.args_slot, args);
	}
	if (g_graphics->nv_checkpoints_enabled) {
		buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(id) << 1u));
	}
	if (g_markers != nullptr) {
		buffer.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_marker_buffer,
		                            StartedOffset, id);
	}
	return id;
}

void NoteWrite(WriteSource source, uint64_t vaddr, uint64_t size, uint64_t value) {
	if (g_graphics == nullptr || size == 0) {
		return;
	}
	const uint64_t serial = g_next_write.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&          w      = g_writes[serial % WriteRingSize];
	w.serial.store(0, std::memory_order_relaxed);
	w.next_command = g_next_id.load(std::memory_order_relaxed) + 1;
	w.source       = source;
	w.vaddr        = vaddr;
	w.size         = size;
	w.value        = value;
	w.serial.store(serial, std::memory_order_release);
}

void NoteCp(CpEvent event, uint64_t address, uint64_t value, uint64_t count) {
	if (g_graphics == nullptr) {
		return;
	}
	const uint64_t serial = g_next_cp.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&          c      = g_cp[serial % CpRingSize];
	c.serial.store(0, std::memory_order_relaxed);
	c.next_command = g_next_id.load(std::memory_order_relaxed) + 1;
	c.event        = event;
	c.address      = address;
	c.value        = value;
	c.count        = count;
	c.serial.store(serial, std::memory_order_release);
}

uint32_t Begin(vk::CommandBuffer buffer, Kind kind, uint64_t hash0, uint64_t hash1, uint64_t tick,
               const IndirectArgs& args, const std::array<uint32_t, 3>& groups) {
	return Record(buffer, kind, hash0, hash1, tick, nullptr, args, groups);
}

uint32_t BeginInternal(vk::CommandBuffer buffer, const char* site) {
	return Record(buffer, Kind::Internal, 0, 0, 0, site, {}, {});
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
	ReportIndirectArgs(focus);
	ReportWriters(focus);
	ReportWindow(focus);
	Write("[DEVLOST] ===== end =====\n");
}

} // namespace Libs::Graphics::DeviceLossDiag
