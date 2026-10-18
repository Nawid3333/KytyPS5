#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_LABPROFILE_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_LABPROFILE_H_

// LAB (KytyLab test build, not for upstream): CPU time per phase of the render path and of the
// fault/trap paths. A background thread prints one "[lab-prof]" summary every 2 s to the console:
// per zone, calls per second, milliseconds per second (summed over all threads) and microseconds
// per call. Zones nest (e.g. Draw contains DrawPrepareState), so do not add them up.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

namespace LabProfile {

enum Zone : uint32_t {
	GpuThreadCommand,
	GpuThreadSubmission,
	Draw,
	DrawPrepareState,
	DrawPrograms,
	DrawBindings,
	DrawVertexIndex,
	DrawPipeline,
	DrawTargets,
	DrawCommit,
	DrawRecord,
	Dispatch,
	DispatchProgram,
	DispatchPipeline,
	DispatchBindings,
	DispatchCommit,
	DispatchRecord,
	Submit,
	GpuWait,
	SyncReadback,
	HandleFault,
	Emulate,
	ShaderCompile,
	Count
};

inline constexpr std::array<const char*, Count> Names = {
    "GpuThreadCommand", "GpuThreadSubmission", "Draw",          "DrawPrepareState",
    "DrawPrograms",     "DrawBindings",        "DrawVertexIndex", "DrawPipeline",
    "DrawTargets",      "DrawCommit",          "DrawRecord",    "Dispatch",
    "DispatchProgram",  "DispatchPipeline",    "DispatchBindings", "DispatchCommit",
    "DispatchRecord",   "Submit",              "GpuWait",       "SyncReadback",
    "HandleFault",      "Emulate",             "ShaderCompile"};

struct alignas(64) Slot {
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> calls {0};
};

inline std::array<Slot, Count> g_slots {};

inline uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

inline void Report() {
	std::array<uint64_t, Count> last_ns {};
	std::array<uint64_t, Count> last_calls {};
	uint64_t                    last_time  = NowNs();
	const uint64_t              start_time = last_time;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(2));
		const uint64_t now     = NowNs();
		const double   seconds = static_cast<double>(now - last_time) / 1e9;
		std::string    line    = "[lab-prof] t=" +
		                    std::to_string(static_cast<uint64_t>((now - start_time) / 1000000000u)) +
		                    "s";
		bool any = false;
		for (uint32_t zone = 0; zone < Count; zone++) {
			const uint64_t ns    = g_slots[zone].ns.load(std::memory_order_relaxed);
			const uint64_t calls = g_slots[zone].calls.load(std::memory_order_relaxed);
			const uint64_t d_ns  = ns - last_ns[zone];
			const uint64_t d_cnt = calls - last_calls[zone];
			last_ns[zone]        = ns;
			last_calls[zone]     = calls;
			if (d_cnt == 0) {
				continue;
			}
			char buffer[160];
			std::snprintf(buffer, sizeof(buffer), " | %s %.0f/s %.1fms/s %.1fus", Names[zone],
			              static_cast<double>(d_cnt) / seconds,
			              static_cast<double>(d_ns) / 1e6 / seconds,
			              static_cast<double>(d_ns) / 1e3 / static_cast<double>(d_cnt));
			line += buffer;
			any = true;
		}
		last_time = now;
		if (any) {
			std::printf("%s\n", line.c_str());
			std::fflush(stdout);
		}
	}
}

inline void StartReporter() {
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Report).detach(); });
}

inline void Add(Zone zone, uint64_t ns) {
	g_slots[zone].ns.fetch_add(ns, std::memory_order_relaxed);
	g_slots[zone].calls.fetch_add(1, std::memory_order_relaxed);
}

// Charges the time from construction to destruction to one zone.
struct Scope {
	explicit Scope(Zone zone): zone(zone), start(NowNs()) { StartReporter(); }
	~Scope() { Add(zone, NowNs() - start); }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

	Zone     zone;
	uint64_t start;
};

// Same as Scope, but never starts the reporter thread (used inside the exception handler).
struct QuietScope {
	explicit QuietScope(Zone zone): zone(zone), start(NowNs()) {}
	~QuietScope() { Add(zone, NowNs() - start); }
	QuietScope(const QuietScope&)            = delete;
	QuietScope& operator=(const QuietScope&) = delete;

	Zone     zone;
	uint64_t start;
};

// Charges the time since the previous mark (or construction) to the given zone.
struct Lap {
	Lap(): last(NowNs()) {}
	void Mark(Zone zone) {
		const uint64_t now = NowNs();
		Add(zone, now - last);
		last = now;
	}
	uint64_t last;
};

} // namespace LabProfile

#endif /* EMULATOR_INCLUDE_EMULATOR_COMMON_LABPROFILE_H_ */
