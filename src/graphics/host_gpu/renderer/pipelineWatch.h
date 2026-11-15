#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::PipelineWatch {

void RememberModule(vk::ShaderModule module, std::span<const uint32_t> code);

class Scope {
public:
	Scope(std::string description, std::vector<vk::ShaderModule> modules);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	uint64_t m_id;
};

} // namespace Libs::Graphics::PipelineWatch
