#include "common/logging/log.h"

#include "common/emulatorConfig.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <spdlog/logger.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

namespace Log {

static bool g_initialized = false;
static bool g_console     = false;
// Null when silent. Only Initialize sets it, before other threads start. Never destroyed, since
// threads keep logging while exit runs static destructors.
static spdlog::logger* g_logger = nullptr;

struct RepeatSite {
	std::atomic<const void*> site {nullptr};
	std::atomic<uint64_t>    calls {0};
};

static constexpr uint32_t REPEAT_SITE_BITS   = 13;
static constexpr size_t   REPEAT_SITE_SLOTS  = size_t {1} << REPEAT_SITE_BITS;
static constexpr size_t   REPEAT_SITE_PROBES = 64;
static constexpr uint64_t FIBONACCI_HASH     = 0x9e3779b97f4a7c15ull;

static std::array<RepeatSite, REPEAT_SITE_SLOTS> g_repeat_sites;

static RepeatSite* FindRepeatSite(const void* site) {
	const auto first =
	    (reinterpret_cast<uintptr_t>(site) * FIBONACCI_HASH) >> (64 - REPEAT_SITE_BITS);
	for (size_t probe = 0; probe < REPEAT_SITE_PROBES; probe++) {
		auto&       slot    = g_repeat_sites[(first + probe) % REPEAT_SITE_SLOTS];
		const void* claimed = slot.site.load(std::memory_order_acquire);
		if (claimed == nullptr && slot.site.compare_exchange_strong(claimed, site)) {
			claimed = site;
		}
		if (claimed == site) {
			return &slot;
		}
	}
	return nullptr;
}

static void WriteStdout(std::string_view text, fmt::text_style style) {
	if (style == fmt::text_style {}) {
		std::fwrite(text.data(), 1, text.size(), stdout);
	} else {
		fmt::print(stdout, style, "{}", text);
	}
	std::fflush(stdout);
}

void Initialize() {
	g_initialized        = true;
	const auto direction = Config::GetPrintfDirection();
	const auto path      = Config::GetPrintfOutputFile();
	g_console            = direction == Config::LogDirection::Console;
	Detail::g_repeat_limit = Config::GetLogRepeatLimit();
	for (auto& slot: g_repeat_sites) {
		slot.site.store(nullptr);
		slot.calls.store(0);
	}
	delete g_logger;
	g_logger = nullptr;

	spdlog::sink_ptr sink;
	if (g_console) {
		sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
	} else if (direction == Config::LogDirection::File && !path.empty()) {
		if (path.has_parent_path()) {
			std::filesystem::create_directories(path.parent_path());
		}
		const bool truncate = std::getenv(RelaunchedVariable) == nullptr;
		sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.native(), truncate);
	}
	if (sink != nullptr) {
		sink->set_formatter(std::make_unique<spdlog::pattern_formatter>(
		    "%v", spdlog::pattern_time_type::local, ""));
		g_logger = new spdlog::logger("kyty", std::move(sink));
	}
}

void Flush() {
	if (g_logger != nullptr) {
		g_logger->flush();
	}
}

bool IsSilent() {
	// Before Initialize the text goes to stdout.
	return g_initialized && g_logger == nullptr;
}

void Write(fmt::text_style style, std::string_view text) {
	if (!g_initialized) {
		WriteStdout(text, style);
		return;
	}
	if (g_logger == nullptr) {
		return;
	}
	if (g_console && style != fmt::text_style {}) {
		fmt::memory_buffer styled;
		fmt::format_to(std::back_inserter(styled), style, "{}", text);
		g_logger->log(spdlog::level::info, spdlog::string_view_t(styled.data(), styled.size()));
	} else {
		g_logger->log(spdlog::level::info, spdlog::string_view_t(text.data(), text.size()));
	}
}

bool AdmitRepeat(const void* site, uint64_t* skipped) {
	const uint64_t limit = Detail::g_repeat_limit;
	*skipped             = 0;
	auto* slot           = FindRepeatSite(site);
	if (slot == nullptr) {
		return true;
	}
	const uint64_t call = slot->calls.fetch_add(1, std::memory_order_relaxed) + 1;
	if (call <= limit) {
		return true;
	}
	if ((call & (call - 1)) != 0) {
		return false;
	}
	*skipped = call - std::max(limit, call / 2) - 1;
	return true;
}

void WriteRepeatNote(fmt::text_style style, uint64_t skipped, std::string_view text) {
	Write(style,
	      fmt::format("[log-repeat] {} calls of the next line's call site were not printed\n{}",
	                  skipped, text));
}

void WriteToConsoleAndLog(std::string_view text) {
	if (g_initialized && !g_console) {
		WriteStdout(text, {});
	}
	Write({}, text);
	Flush();
}

} // namespace Log
