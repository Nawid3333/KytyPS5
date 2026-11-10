#ifndef EMULATOR_SRC_COMMON_LABSYNC_H_
#define EMULATOR_SRC_COMMON_LABSYNC_H_

// LAB ONLY (KytyLab experiments/2026-11-10-des-frametime, section 8d): where BufferCache::SynchronizeBuffer
// spends its time. Not for upstream.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace LabSync {

inline std::atomic<uint64_t> g_spin_ns {0};
inline std::atomic<uint64_t> g_spin_count {0};
inline std::atomic<uint64_t> g_spin_max_ns {0};
inline std::atomic<uint64_t> g_other_spin_ns {0};
inline std::atomic<uint64_t> g_other_spin_count {0};
inline std::atomic<uint64_t> g_protect_ns {0};
inline std::atomic<uint64_t> g_protect_count {0};
inline thread_local bool     tl_in_sync = false;
inline std::atomic<uint64_t> g_park_count {0};
inline std::atomic<uint64_t> g_other_park_count {0};
inline std::atomic<uint64_t> g_long_hold_ns {0};
inline std::atomic<uint64_t> g_long_hold_count {0};
inline std::atomic<uint64_t> g_long_hold_max_ns {0};
inline std::atomic<uint64_t> g_other_long_hold_ns {0};
inline std::atomic<uint64_t> g_other_long_hold_count {0};
inline std::atomic<uint64_t> g_other_long_hold_max_ns {0};

// KYTY_LAB_SPIN_PARK=<n>: a waiter that saw the lock taken n times in a row sleeps until the holder unlocks
// (0 or unset: upstream behaviour, spin for ever).
inline const uint32_t g_park_spins = [] {
	const char* text = std::getenv("KYTY_LAB_SPIN_PARK");
	return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
}();

inline void AddPark() noexcept {
	(tl_in_sync ? g_park_count : g_other_park_count).fetch_add(1, std::memory_order_relaxed);
}

inline void AddHold(uint64_t ns) noexcept {
	if (ns < 1'000'000) {
		return;
	}
	auto& sum   = tl_in_sync ? g_long_hold_ns : g_other_long_hold_ns;
	auto& count = tl_in_sync ? g_long_hold_count : g_other_long_hold_count;
	auto& top   = tl_in_sync ? g_long_hold_max_ns : g_other_long_hold_max_ns;
	sum.fetch_add(ns, std::memory_order_relaxed);
	count.fetch_add(1, std::memory_order_relaxed);
	auto max = top.load(std::memory_order_relaxed);
	while (ns > max && !top.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
	}
}

inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

inline void AddSpin(uint64_t ns) noexcept {
	if (tl_in_sync) {
		g_spin_ns.fetch_add(ns, std::memory_order_relaxed);
		g_spin_count.fetch_add(1, std::memory_order_relaxed);
		auto max = g_spin_max_ns.load(std::memory_order_relaxed);
		while (ns > max && !g_spin_max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
		}
	} else {
		g_other_spin_ns.fetch_add(ns, std::memory_order_relaxed);
		g_other_spin_count.fetch_add(1, std::memory_order_relaxed);
	}
}

void PrepareDone(uint64_t prepare_ns, uint64_t ranges, uint64_t range_bytes);

} // namespace LabSync

#endif // EMULATOR_SRC_COMMON_LABSYNC_H_
