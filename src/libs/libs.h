#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <fmt/format.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLED g_print_name

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLE(flag) PRINT_NAME_ENABLED = flag;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_DEFINE(name) void name(Loader::SymbolDatabase* s)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_NAME(l, m)                                                                             \
	[[maybe_unused]] static thread_local bool PRINT_NAME_ENABLED = false;                          \
	static constexpr char                     g_library[]        = l;                              \
	static constexpr char                     g_module[]         = m;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_VERSION(l, lv, m, mv1, mv2)                                                            \
	LIB_NAME(l, m);                                                                                \
	static constexpr int g_library_version      = lv;                                              \
	static constexpr int g_module_version_major = mv1;                                             \
	static constexpr int g_module_version_minor = mv2;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_ADD(n, f, t)                                                                           \
	{                                                                                              \
		Loader::SymbolResolve sr {};                                                               \
		sr.name                 = n;                                                               \
		sr.library              = g_library;                                                       \
		sr.library_version      = g_library_version;                                               \
		sr.module               = g_module;                                                        \
		sr.module_version_major = g_module_version_major;                                          \
		sr.module_version_minor = g_module_version_minor;                                          \
		sr.type                 = t;                                                               \
		auto        func        = reinterpret_cast<uint64_t>(f);                                   \
		const char* dbg_name    = "" #f;                                                           \
		s->Add(sr, func, dbg_name);                                                                \
	}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_OBJECT(n, f) LIB_ADD(n, f, Loader::SymbolType::Object)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f) LIB_ADD(n, f, Loader::SymbolType::Func)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	if (PRINT_NAME_ENABLED) {                                                                      \
		Libs::PrintName(g_library, g_module, __func__);                                              \
	} else if (::Libs::g_lab_trace_all.load(std::memory_order_relaxed)) {                          \
		Libs::LabPrintNameCaller(g_library, g_module, __func__, __builtin_return_address(0));       \
	}

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

inline std::atomic<bool> g_lab_trace_all {false};

inline bool LabTraceStack(const char* function) {
	static const std::string names = [] {
		const char* value = std::getenv("KYTY_LAB_TRACE_STACK");
		return value != nullptr ? "," + std::string(value) + "," : std::string();
	}();
	return !names.empty() && names.find("," + std::string(function) + ",") != std::string::npos;
}

bool LabGuestStack(uint64_t* stack_addr, uint64_t* stack_size);

[[gnu::noinline]] inline void LabScanStack(char* out, size_t size) {
	uintptr_t stack_base  = 0;
	uintptr_t stack_limit = 0;
#if defined(_WIN32)
	asm volatile("movq %%gs:0x8, %0" : "=r"(stack_base));
	asm volatile("movq %%gs:0x10, %0" : "=r"(stack_limit));
#endif
	const auto frame = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
	if (frame < stack_limit || frame >= stack_base) {
		uint64_t guest_addr = 0;
		uint64_t guest_size = 0;
		if (!LabGuestStack(&guest_addr, &guest_size) || frame < guest_addr || frame >= guest_addr + guest_size) {
			snprintf(out, size, " stack=?");
			return;
		}
		stack_limit = guest_addr;
		stack_base  = guest_addr + guest_size;
	}
	const uintptr_t end   = stack_base - frame > 0x10000 ? frame + 0x10000 : stack_base;
	int             used  = snprintf(out, size, " stack=");
	int             found = 0;
	for (uintptr_t p = frame; p + 8 <= end && found < 12 && used < static_cast<int>(size) - 20; p += 8) {
		const auto value = *reinterpret_cast<const uint64_t*>(p);
		if (value >= 0x900000000ull && value < 0xa00000000ull) {
			used += snprintf(out + used, size - used, "%llx,", static_cast<unsigned long long>(value));
			found++;
		}
	}
}

[[gnu::noinline]] inline void LabPrintNameCaller(const char* library, const char* module, const char* function,
                                                 const void* caller) {
	const auto elapsed_ms = static_cast<unsigned long long>(Loader::Timer::GetTimeMs());
	char       stack[256] = "";
	if (LabTraceStack(function)) {
		LabScanStack(stack, sizeof(stack));
	}
	LOGF("[%d][%02llu:%02llu:%02llu.%03llu] %s::%s::%s() caller=0x%016llx%s\n", Common::Thread::GetThreadIdUnique(),
	     elapsed_ms / 3600000, (elapsed_ms / 60000) % 60, (elapsed_ms / 1000) % 60, elapsed_ms % 1000, library, module,
	     function, static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(caller)), stack);
}

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
[[gnu::noinline]] inline void PrintName(const char* library, const char* module, const char* function) {
	if (!Log::IsSilent()) {
		const auto elapsed_ms      = static_cast<uint64_t>(Loader::Timer::GetTimeMs());
		const auto print_name_time = fmt::format(
		    "{:02}:{:02}:{:02}.{:03}", elapsed_ms / 3600000, (elapsed_ms / 60000) % 60,
		    (elapsed_ms / 1000) % 60, elapsed_ms % 1000);
		LOGF_COLOR(Log::Color::Cyan, "[%d][%s] %s::%s::%s()\n",
		           Common::Thread::GetThreadIdUnique(), print_name_time.c_str(), library, module,
		           function);
	}
}

void InitAll(Loader::SymbolDatabase* s);

} // namespace Libs
#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_ */
