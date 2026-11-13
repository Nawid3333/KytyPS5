#pragma once

#include "common/logging/log.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>

#include <fmt/format.h>

namespace Libs::Graphics::Lab {

inline void TraceWriteData(const char* where, uint32_t control, uint64_t address, uint32_t dwords,
                           const uint32_t* data, const void* caller) {
	static std::mutex                                              mutex;
	static std::set<std::tuple<std::string, uint32_t, uint32_t>> seen;
	static uint32_t                                                lines = 0;
	constexpr uint32_t                                             max_lines = 256;
	std::lock_guard lock(mutex);
	if (lines >= max_lines ||
	    !seen.emplace(where, control, static_cast<uint32_t>(address >> 32u)).second) {
		return;
	}
	lines++;
	auto text = fmt::format("[LAB 2233] {} control=0x{:08x} address=0x{:016x} dwords={} caller=0x{:016x}",
	                        where, control, address, dwords, reinterpret_cast<uintptr_t>(caller));
	if (data != nullptr) {
		text += " data=";
		for (uint32_t i = 0; i < dwords && i < 4u; i++) {
			text += fmt::format("{}{:08x}", i == 0 ? "" : ",", data[i]);
		}
	}
	text += '\n';
	Log::WriteToConsoleAndLog(text);
	Log::Flush();
}

inline void TraceRegList(const char* where, const uint32_t* packet, uint32_t packet_dwords,
                         uint64_t list, uint32_t num_regs, const void* caller, bool force) {
	static std::mutex  mutex;
	static uint32_t    lines     = 0;
	constexpr uint32_t max_lines = 96;
	constexpr uint32_t max_pairs = 24;
	std::lock_guard    lock(mutex);
	if (!force && lines >= max_lines) {
		return;
	}
	lines++;
	auto text = fmt::format("[LAB 2233] {} list=0x{:016x} num_regs={} caller=0x{:016x}", where,
	                        list, num_regs, reinterpret_cast<uintptr_t>(caller));
	if (packet != nullptr) {
		text += " packet=";
		for (uint32_t i = 0; i < packet_dwords; i++) {
			text += fmt::format("{}{:08x}", i == 0 ? "" : ",", packet[i]);
		}
	}
	if (list != 0) {
		const auto* pairs = reinterpret_cast<const volatile uint32_t*>(list);
		text += " regs=";
		for (uint32_t i = 0; i < num_regs && i < max_pairs; i++) {
			text += fmt::format("{}{:x}:{:08x}", i == 0 ? "" : ",", pairs[i * 2], pairs[i * 2 + 1]);
		}
		if (num_regs > max_pairs) {
			text += ",...";
		}
	}
	text += '\n';
	Log::WriteToConsoleAndLog(text);
	Log::Flush();
}

inline void TraceLine(const std::string& text) {
	static std::mutex  mutex;
	static uint32_t    lines     = 0;
	constexpr uint32_t max_lines = 256;
	std::lock_guard    lock(mutex);
	if (lines >= max_lines) {
		return;
	}
	lines++;
	Log::WriteToConsoleAndLog("[LAB 2233] " + text + "\n");
	Log::Flush();
}

inline void TraceListMemory(const char* where, uint64_t list, uint32_t dwords_before, uint32_t dwords) {
	const auto* words = reinterpret_cast<const volatile uint32_t*>(list);
	auto        text  = fmt::format("[LAB 2233] {} memory at 0x{:016x} - {} dwords:", where, list, dwords_before);
	for (uint32_t i = 0; i < dwords_before + dwords; i++) {
		const auto index = static_cast<int64_t>(i) - static_cast<int64_t>(dwords_before);
		if (i % 8 == 0) {
			text += fmt::format("\n[LAB 2233]   {:016x}:", static_cast<uint64_t>(static_cast<int64_t>(list) + index * 4));
		}
		text += fmt::format(" {:08x}", words[index]);
	}
	text += '\n';
	Log::WriteToConsoleAndLog(text);
	Log::Flush();
}

inline void TracePacketsBefore(const uint32_t* base, uint32_t packet_offset) {
	auto text = fmt::format("[LAB 2233] packets before the failing one (buffer 0x{:016x}, failing at dword 0x{:05x}):",
	                        reinterpret_cast<uintptr_t>(base), packet_offset);
	for (uint32_t at = 0; at < packet_offset;) {
		const auto header = base[at];
		const auto len    = (header >> 30u) == 3u ? ((header >> 16u) & 0x3fffu) + 2u : 1u;
		text += fmt::format("\n[LAB 2233]   {:05x} op={:02x} len={}", at, (header >> 8u) & 0xffu, len);
		for (uint32_t i = 0; i < len && i < 8u && at + i < packet_offset; i++) {
			text += fmt::format(" {:08x}", base[at + i]);
		}
		at += len;
	}
	text += '\n';
	Log::WriteToConsoleAndLog(text);
	Log::Flush();
}

inline void TraceListAfterDelay(uint64_t list, uint32_t num_regs) {
	std::this_thread::sleep_for(std::chrono::milliseconds(250));
	TraceRegList("cp SET_SH_REG_INDIRECT FAILING list re-read after 250 ms", nullptr, 0, list, num_regs, nullptr, true);
}

} // namespace Libs::Graphics::Lab
