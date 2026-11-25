#include "common/assert.h"

#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <utility>

namespace Common {

static thread_local DbgReportContextWriter g_context_writer = nullptr;
static thread_local const void*            g_context        = nullptr;
static thread_local const std::string*     g_pending_report = nullptr;

DbgReportContext::DbgReportContext(DbgReportContextWriter writer, const void* context)
    : m_previous_writer(std::exchange(g_context_writer, writer)),
      m_previous_context(std::exchange(g_context, context)) {}

DbgReportContext::~DbgReportContext() {
	g_context_writer = m_previous_writer;
	g_context        = m_previous_context;
}

static std::array<char, 256> g_game {};
static std::atomic<size_t>   g_game_size {0};

void SetDbgReportGame(std::string_view game) {
	const auto size = std::min(game.size(), g_game.size());
	std::copy_n(game.data(), size, g_game.data());
	g_game_size.store(size, std::memory_order_release);
}

[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line) {
	const auto game_size = g_game_size.load(std::memory_order_acquire);
	const auto game   = game_size != 0
	                        ? fmt::format("Game: {}\n", std::string_view(g_game.data(), game_size))
	                        : std::string();
	const auto report = fmt::format("--- Build ---\n{}\n{}{}\n{} in {}:{}\n", KYTY_BUILD_LABEL,
	                                game, title, text, file, line);
	if (g_pending_report != nullptr) {
		if (*g_pending_report != report) {
			Log::WriteToConsoleAndLog(*g_pending_report);
		}
	} else if (const auto writer = std::exchange(g_context_writer, nullptr); writer != nullptr) {
		g_pending_report = &report;
		Log::WriteToConsoleAndLog(writer(g_context));
		g_pending_report = nullptr;
	}
	Log::WriteToConsoleAndLog(report);
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(321);
}

void DbgExitIfHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr), file,
	          line);
}

void DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

void DbgExitHandler(const char* file, int line, std::string_view text) {
	DbgReport("--- Error ---", text, file, line);
}

} // namespace Common
