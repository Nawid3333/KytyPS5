#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <xxhash.h>

namespace {

using namespace Libs::Graphics;

struct LabCompileOptionsHeader {
  char magic[4];
  uint32_t version;
  uint32_t stage;
  uint32_t wave_size;
  uint32_t user_data_base;
  uint32_t input_size;
  uint32_t user_data_count;
  uint32_t back_code_count;
  uint64_t hash;
};

std::vector<uint8_t> ReadFile(const char *path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<uint32_t> Words(const std::vector<uint8_t> &bytes, size_t offset, size_t count) {
  std::vector<uint32_t> words(count);
  std::memcpy(words.data(), bytes.data() + offset, count * sizeof(uint32_t));
  return words;
}

template <typename Info>
bool LoadInfo(const std::vector<uint8_t> &bytes, const LabCompileOptionsHeader &header, Info &info) {
  if (header.input_size != sizeof(Info) || sizeof(header) + sizeof(Info) > bytes.size()) return false;
  std::memcpy(static_cast<void *>(&info), bytes.data() + sizeof(header), sizeof(Info));
  info.stage = {};
  return true;
}

long long Ms(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - from).count();
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: lab_shader_replay <stage_hash.bin> <stage_hash.opts> [--log]\n");
    return 2;
  }
  static Common::Subsystems subsystems;
  Common::InitializeThreads();
  subsystems.Initialize<Config::Lifecycle>();
  Config::ConfigOptions config;
  config.printf_direction =
      argc > 3 && std::string(argv[3]) == "--log" ? Config::LogDirection::Console : Config::LogDirection::Silent;
  Config::Load(config);
  subsystems.Initialize<Log::Lifecycle>();
  ShaderInit();

  const auto code_bytes = ReadFile(argv[1]);
  const auto opts = ReadFile(argv[2]);
  LabCompileOptionsHeader header{};
  if (code_bytes.empty() || code_bytes.size() % 4 != 0 || opts.size() < sizeof(header)) {
    std::fprintf(stderr, "replay: bad input files\n");
    return 2;
  }
  std::memcpy(&header, opts.data(), sizeof(header));
  if (std::memcmp(header.magic, "KLOP", 4) != 0 || header.version != 1 ||
      sizeof(header) + header.input_size + 4ull * (header.user_data_count + header.back_code_count) != opts.size()) {
    std::fprintf(stderr, "replay: options file does not match this build (input_size=%u)\n", header.input_size);
    return 2;
  }
  const auto code = Words(code_bytes, 0, code_bytes.size() / 4);
  const auto user_data = Words(opts, sizeof(header) + header.input_size, header.user_data_count);
  const auto back_code =
      Words(opts, sizeof(header) + header.input_size + 4ull * header.user_data_count, header.back_code_count);

  ShaderRecompiler::CompileOptions options;
  options.stage = static_cast<ShaderType>(header.stage);
  options.wave_size = header.wave_size;
  options.user_data_base = header.user_data_base;
  options.shader_hash = header.hash;
  options.dump_ir = false;
  options.user_data = user_data;
  options.back_code = back_code;
  ShaderComputeInputInfo compute{};
  ShaderPixelInputInfo pixel{};
  ShaderVertexInputInfo vertex{};
  bool loaded = false;
  if (options.stage == ShaderType::Compute) {
    loaded = LoadInfo(opts, header, compute);
    options.input_info.compute = &compute;
  } else if (options.stage == ShaderType::Pixel) {
    loaded = LoadInfo(opts, header, pixel);
    options.input_info.pixel = &pixel;
  } else {
    loaded = LoadInfo(opts, header, vertex);
    options.input_info.vertex = &vertex;
  }
  if (!loaded) {
    std::fprintf(stderr, "replay: input info size %u does not match this build\n", header.input_size);
    return 2;
  }
  std::printf("replay: hash=0x%016llx stage=%s words=%zu user_data=%u\n", static_cast<unsigned long long>(header.hash),
              ShaderRecompiler::StageName(options.stage), code.size(), header.user_data_count);
  std::fflush(stdout);
  const auto start = std::chrono::steady_clock::now();
  auto source = ShaderRecompiler::PrepareShaderSource(code, options);
  const auto prepared = Ms(start);
  if (source.unsupported_call_pc != UINT32_MAX || !source.calls.empty() || source.linked) {
    std::printf("replay: prepare_ms=%lld GPU-selected calls: not replayed\n", prepared);
    return 3;
  }
  std::printf("replay: prepare_ms=%lld\n", prepared);
  std::fflush(stdout);
  const auto translate_start = std::chrono::steady_clock::now();
  auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
  const auto translate_ms = Ms(translate_start);
  const auto plan = ShaderRecompiler::IR::SaveResourcePlan(translated.program);
  std::printf("replay: translate_ms=%lld blocks=%zu plan_bytes=%zu plan_hash=%016llx\n", translate_ms,
              translated.program.blocks.size(), plan.size(),
              static_cast<unsigned long long>(XXH3_64bits(plan.data(), plan.size())));
  std::fflush(stdout);
  return 0;
}
