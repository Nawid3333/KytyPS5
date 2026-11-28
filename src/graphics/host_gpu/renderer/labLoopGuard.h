#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::Lab {

constexpr uint32_t LoopGuardCap = 4096;

std::vector<uint32_t> GuardLoops(std::span<const uint32_t> words, uint32_t cap, uint64_t address,
                                 std::vector<std::string>* report, std::string* error);

vk::ShaderModule CompileGuardedSPV(std::span<const uint32_t> spirv, vk::Device device, uint64_t hash);

} // namespace Libs::Graphics::Lab
