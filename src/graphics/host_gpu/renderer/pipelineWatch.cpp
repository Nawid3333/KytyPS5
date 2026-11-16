#include "graphics/host_gpu/renderer/pipelineWatch.h"

#include "common/logging/log.h"
#include "common/stringUtils.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace Libs::Graphics::PipelineWatch {

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto REPORT_AFTER = std::chrono::seconds(20);
constexpr auto POLL_PERIOD  = std::chrono::seconds(2);
constexpr auto REMIND_EVERY = std::chrono::seconds(60);

struct InFlight {
	std::string                   description;
	std::vector<vk::ShaderModule> modules;
	Clock::time_point             start;
	Clock::time_point             next_report;
	bool                          reported = false;
};

struct State {
	std::mutex                                                mutex;
	std::unordered_map<VkShaderModule, std::vector<uint32_t>> code;
	std::map<uint64_t, InFlight>                              in_flight;
	uint64_t                                                  next_id = 1;
	std::once_flag                                            watcher_started;
};

State& GetState() {
	static auto* state = new State;
	return *state;
}

void Write(const std::string& text) {
	Log::WriteToConsoleAndLog(text);
}

std::string DumpModules(uint64_t id, const InFlight& entry, const char* tag = "PIPELINE-SLOW",
                        const char* folder_name = "_SlowPipelines") {
	const std::filesystem::path folder = std::filesystem::absolute(folder_name);
	std::error_code             error;
	std::filesystem::create_directories(folder, error);
	std::string written;
	for (size_t index = 0; index < entry.modules.size(); index++) {
		const auto found = GetState().code.find(static_cast<VkShaderModule>(entry.modules[index]));
		if (found == GetState().code.end()) {
			continue;
		}
		const auto    path = folder / fmt::format("pipeline{}_stage{}.spv", id, index);
		std::ofstream file(path, std::ios::binary);
		file.write(reinterpret_cast<const char*>(found->second.data()),
		           static_cast<std::streamsize>(found->second.size() * sizeof(uint32_t)));
		written += fmt::format("[{}]   wrote {} ({} bytes)\n", tag,
		                       Common::PathToString(path), found->second.size() * sizeof(uint32_t));
	}
	return written.empty() ? fmt::format("[{}]   no SPIR-V kept for its modules\n", tag) : written;
}

void Watch() {
	for (;;) {
		std::this_thread::sleep_for(POLL_PERIOD);
		std::string text;
		{
			auto&           state = GetState();
			std::lock_guard lock(state.mutex);
			const auto      now = Clock::now();
			for (auto& [id, entry]: state.in_flight) {
				if (now < entry.next_report) {
					continue;
				}
				const auto seconds =
				    std::chrono::duration_cast<std::chrono::seconds>(now - entry.start).count();
				text +=
				    fmt::format("[PIPELINE-SLOW] pipeline #{} has been compiling for {} s: {}\n",
				                id, seconds, entry.description);
				if (!entry.reported) {
					text += DumpModules(id, entry);
					entry.reported = true;
				}
				entry.next_report = now + REMIND_EVERY;
			}
		}
		if (!text.empty()) {
			Write(text);
			std::fflush(nullptr);
		}
	}
}

} // namespace

void RememberModule(vk::ShaderModule module, std::span<const uint32_t> code) {
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	state.code[static_cast<VkShaderModule>(module)].assign(code.begin(), code.end());
}

Scope::Scope(std::string description, std::vector<vk::ShaderModule> modules) {
	auto& state = GetState();
	std::call_once(state.watcher_started, [] { std::thread(Watch).detach(); });
	const auto      now = Clock::now();
	std::lock_guard lock(state.mutex);
	m_id = state.next_id++;
	state.in_flight.emplace(m_id, InFlight {.description = std::move(description),
	                                        .modules     = std::move(modules),
	                                        .start       = now,
	                                        .next_report = now + REPORT_AFTER});
}

Scope::~Scope() {
	std::string text;
	{
		auto&           state = GetState();
		std::lock_guard lock(state.mutex);
		const auto      found = state.in_flight.find(m_id);
		if (found == state.in_flight.end()) {
			return;
		}
		if (found->second.reported) {
			const auto seconds =
			    std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - found->second.start)
			        .count();
			text = fmt::format("[PIPELINE-SLOW] pipeline #{} finished after {} s\n", m_id, seconds);
		}
		state.in_flight.erase(found);
	}
	if (!text.empty()) {
		Write(text);
	}
}

void Scope::Failed(vk::Result result) const {
	std::string text;
	{
		auto&           state = GetState();
		std::lock_guard lock(state.mutex);
		const auto      found = state.in_flight.find(m_id);
		if (found == state.in_flight.end()) {
			return;
		}
		text = fmt::format("[PIPELINE-FAIL] pipeline #{} returned {}: {}\n", m_id,
		                   vk::to_string(result), found->second.description);
		text += DumpModules(m_id, found->second, "PIPELINE-FAIL", "_FailedPipelines");
	}
	Write(text);
	std::fflush(nullptr);
}

} // namespace Libs::Graphics::PipelineWatch
