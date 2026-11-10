#include "graphics/host_gpu/renderer/fsr.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/shaderBindings.h"

#include <SDL3/SDL_loadso.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include "api/include/vk/ffx_api_vk.h"
#include "upscalers/include/ffx_upscale.h"
#endif

namespace Libs::Graphics::Fsr {

bool Accepts(const Pssr::Request& request) {
	ShaderTextureResource depth {}, motion {};
	std::copy_n(request.depth, 8, depth.fields);
	std::copy_n(request.motion, 8, motion.fields);
	return (request.flags & Pssr::kSpatialOnly) == 0 && !depth.IsNull() && !motion.IsNull() &&
	       request.x == 0 && request.y == 0;
}

bool Viewable(VkFormat image_format, VkImageCreateFlags image_flags, VkFormat view_format,
              uint32_t samples) {
	return samples == 1 && (image_format == view_format ||
	                        (image_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) != 0);
}

bool IdentityChannels(VkComponentMapping mapping, uint32_t channels) {
	const std::array components {mapping.r, mapping.g, mapping.b, mapping.a};
	const std::array own {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
	                      VK_COMPONENT_SWIZZLE_A};
	for (uint32_t i = 0; i < std::min<uint32_t>(channels, 4); ++i) {
		if (components[i] != VK_COMPONENT_SWIZZLE_IDENTITY && components[i] != own[i]) {
			return false;
		}
	}
	return true;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr const char* LOADER_DLL   = "amd_fidelityfx_loader_vk.dll";
constexpr const char* UPSCALER_DLL = "amd_fidelityfx_upscaler_vk.dll";

constexpr float CAMERA_NEAR          = 0.1f;
constexpr float CAMERA_FAR           = 10000.0f;
constexpr float CAMERA_FOV_VERTICAL  = std::numbers::pi_v<float> / 3.0f;
constexpr float FIRST_FRAME_MS       = 1000.0f / 60.0f;

struct Api {
	PfnFfxCreateContext  create;
	PfnFfxDestroyContext destroy;
	PfnFfxQuery          query;
	PfnFfxDispatch       dispatch;
};

SDL_SharedObject* g_upscaler = nullptr;
SDL_SharedObject* g_loader   = nullptr;
Api               g_api {};
VkDeviceContext   g_device {};
bool              g_available = false;

void Warn(std::string_view text) {
	Log::WriteToConsoleAndLog(fmt::format("WARNING: FSR: {}; PSSR scales linearly.\n", text));
}

void OnMessage(uint32_t type, const wchar_t* message) noexcept {
	try {
		std::wstring_view text(message != nullptr ? message : L"");
		std::string       narrow;
		narrow.reserve(text.size());
		for (const wchar_t c: text) {
			narrow.push_back(c < 0x80 ? static_cast<char>(c) : '?');
		}
		Log::WriteToConsoleAndLog(fmt::format("FSR message {}: {}\n", type, narrow));
	} catch (...) {
	}
}

template <typename Function>
bool Bind(Function& function, const char* name) {
	function = reinterpret_cast<Function>(SDL_LoadFunction(g_loader, name));
	return function != nullptr;
}

void Unload() {
	SDL_UnloadObject(g_loader);
	SDL_UnloadObject(g_upscaler);
	g_loader   = nullptr;
	g_upscaler = nullptr;
	g_api      = {};
}

uint32_t SurfaceFormat(VkFormat format) {
	switch (format) {
		case VK_FORMAT_D32_SFLOAT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
		case VK_FORMAT_D16_UNORM: return FFX_API_SURFACE_FORMAT_R16_UNORM;
		default: return ffxApiGetSurfaceFormatVK(format);
	}
}

std::vector<std::string> Versions() {
	uint64_t                 count = 0;
	ffxQueryDescGetVersions  query {};
	query.header.type    = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
	query.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
	query.device         = &g_device;
	query.outputCount    = &count;
	if (g_api.query(nullptr, &query.header) != FFX_API_RETURN_OK || count == 0) {
		return {};
	}
	std::vector<uint64_t>    ids(count);
	std::vector<const char*> names(count);
	query.versionIds   = ids.data();
	query.versionNames = names.data();
	if (g_api.query(nullptr, &query.header) != FFX_API_RETURN_OK) {
		return {};
	}
	std::vector<std::string> result;
	for (uint64_t i = 0; i < count; ++i) {
		result.emplace_back(names[i] != nullptr ? names[i] : "?");
	}
	return result;
}

} // namespace

uint32_t CreationFlags(const Pssr::Request& request, bool exposure) {
	uint32_t flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
	if ((request.flags & Pssr::kReverseDepth) != 0) {
		flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
	}
	if ((request.flags & Pssr::kAutoExposure) != 0 || !exposure) {
		flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
	}
	return flags;
}

Creation Choose(const Pssr::Request& request, const Inputs& inputs) {
	return {inputs.output.width, inputs.output.height,
	        CreationFlags(request, inputs.exposure != nullptr)};
}

bool Takes(const Creation& creation, const Pssr::Request& request, const Inputs& inputs) {
	return creation == Choose(request, inputs) && request.width <= creation.output_width &&
	       request.height <= creation.output_height;
}

FfxApiResource MakeResource(const Resource* resource) {
	FfxApiResource result {};
	if (resource == nullptr) {
		return result;
	}
	result.resource             = reinterpret_cast<void*>(resource->image);
	result.description.type     = FFX_API_RESOURCE_TYPE_TEXTURE2D;
	result.description.format   = SurfaceFormat(resource->format);
	result.description.width    = resource->width;
	result.description.height   = resource->height;
	result.description.depth    = resource->layers;
	result.description.mipCount = resource->mip_levels;
	result.description.flags    = FFX_API_RESOURCE_FLAGS_NONE;
	result.description.usage    = FFX_API_RESOURCE_USAGE_READ_ONLY;
	if (resource->depth) {
		result.description.usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
	}
	if (resource->format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
	    resource->format == VK_FORMAT_D24_UNORM_S8_UINT ||
	    resource->format == VK_FORMAT_D16_UNORM_S8_UINT) {
		result.description.usage |= FFX_API_RESOURCE_USAGE_STENCILTARGET;
	}
	if ((resource->usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0) {
		result.description.usage |= FFX_API_RESOURCE_USAGE_UAV;
	}
	result.state = resource->output ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
	                                : FFX_API_RESOURCE_STATE_COMPUTE_READ;
	return result;
}

void SetCreation(ffxCreateContextDescUpscale& desc, const Creation& creation) {
	desc                = {};
	desc.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
	desc.flags          = creation.flags;
	desc.maxRenderSize  = {creation.output_width, creation.output_height};
	desc.maxUpscaleSize = {creation.output_width, creation.output_height};
	desc.fpMessage      = OnMessage;
}

void SetDispatch(ffxDispatchDescUpscale& desc, VkCommandBuffer command,
                 const Pssr::Request& request, const Inputs& inputs, bool reset, float frame_ms) {
	desc                         = {};
	desc.header.type             = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
	desc.commandList             = command;
	desc.color                   = MakeResource(&inputs.color);
	desc.depth                   = MakeResource(&inputs.depth);
	desc.motionVectors           = MakeResource(&inputs.motion);
	desc.exposure                = MakeResource(
	                   (request.flags & Pssr::kAutoExposure) != 0 ? nullptr : inputs.exposure);
	desc.output                  = MakeResource(&inputs.output);
	desc.jitterOffset            = {-request.jitter[0], -request.jitter[1]};
	desc.motionVectorScale       = {request.motion_scale[0], request.motion_scale[1]};
	desc.renderSize              = {request.width, request.height};
	desc.upscaleSize             = {inputs.output.width, inputs.output.height};
	desc.enableSharpening        = false;
	desc.sharpness               = 0.0f;
	desc.frameTimeDelta          = std::isfinite(frame_ms) && frame_ms > 0.0f ? frame_ms : FIRST_FRAME_MS;
	desc.preExposure             = std::isfinite(request.pre_exposure) && request.pre_exposure > 0.0f
	                                   ? request.pre_exposure
	                                   : 1.0f;
	desc.reset                   = reset;
	desc.cameraNear              = CAMERA_NEAR;
	desc.cameraFar               = CAMERA_FAR;
	desc.cameraFovAngleVertical  = CAMERA_FOV_VERTICAL;
	desc.viewSpaceToMetersFactor = 1.0f;
	desc.flags                   = 0;
}

void Load() {
	if (!Config::Ps5ProEnabled() || Config::GetFsrFolder().empty()) {
		return;
	}
	const auto folder = Config::GetFsrFolder();
	g_upscaler = SDL_LoadObject(Common::PathToString(folder / UPSCALER_DLL).c_str());
	g_loader   = SDL_LoadObject(Common::PathToString(folder / LOADER_DLL).c_str());
	if (g_upscaler == nullptr || g_loader == nullptr) {
		Warn(fmt::format("{} and {} are needed in {}", LOADER_DLL, UPSCALER_DLL,
		                 Common::PathToString(folder)));
		Unload();
		return;
	}
	if (!Bind(g_api.create, "ffxCreateContext") || !Bind(g_api.destroy, "ffxDestroyContext") ||
	    !Bind(g_api.query, "ffxQuery") || !Bind(g_api.dispatch, "ffxDispatch")) {
		Warn("the FSR loader lacks an entry point");
		Unload();
	}
}

bool Loaded() {
	return g_loader != nullptr;
}

DeviceFeatures AppendDeviceExtensions(vk::PhysicalDevice physical_device,
                                      std::vector<const char*>&                   out,
                                      const std::vector<vk::ExtensionProperties>& available) {
	DeviceFeatures result {};
	if (!Loaded()) {
		return result;
	}
	const auto add = [&](const char* name) {
		const bool has = std::ranges::any_of(available, [&](const auto& properties) {
			return std::string_view(properties.extensionName.data()) == name;
		});
		if (has && std::ranges::none_of(out, [&](const char* other) {
			    return std::string_view(other) == name;
		    })) {
			out.push_back(name);
		}
		return has;
	};
	const bool cooperative_matrix = add(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
	const bool float8             = add(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
	                   vk::PhysicalDeviceVulkan13Features,
	                   vk::PhysicalDeviceCooperativeMatrixFeaturesKHR,
	                   vk::PhysicalDeviceShaderFloat8FeaturesEXT>
	    query;
	if (!cooperative_matrix) {
		query.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	}
	if (!float8) {
		query.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	}
	physical_device.getFeatures2(&query.get<vk::PhysicalDeviceFeatures2>());
	const auto& features12 = query.get<vk::PhysicalDeviceVulkan12Features>();
	const auto& features13 = query.get<vk::PhysicalDeviceVulkan13Features>();
	result.cooperative_matrix =
	    cooperative_matrix &&
	    query.get<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>().cooperativeMatrix;
	if (float8) {
		const auto& features8            = query.get<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
		result.float8                    = features8.shaderFloat8;
		result.float8_cooperative_matrix = features8.shaderFloat8CooperativeMatrix;
	}
	result.integer_dot_product     = features13.shaderIntegerDotProduct;
	result.float16                 = features12.shaderFloat16;
	result.int16                   = query.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16;
	result.int8                    = features12.shaderInt8;
	result.subgroup_extended_types = features12.shaderSubgroupExtendedTypes;
	return result;
}

void Init(const GraphicContext& graphics, PFN_vkGetDeviceProcAddr get_device_proc_addr) {
	if (!Loaded()) {
		return;
	}
	g_device = {static_cast<VkDevice>(graphics.device),
	            static_cast<VkPhysicalDevice>(graphics.physical_device), get_device_proc_addr};
	const auto versions = Versions();
	if (versions.empty()) {
		Warn("the FSR DLLs offer no upscaler for this GPU");
		Shutdown();
		return;
	}
	std::string names;
	for (const auto& name: versions) {
		names += (names.empty() ? "" : ", ") + name;
	}
	g_available = true;
	Log::WriteToConsoleAndLog(fmt::format("FSR: upscaling PSSR; upscalers offered: {}\n", names));
}

void Shutdown() {
	g_available = false;
	g_device    = {};
	Unload();
}

bool Available() {
	return g_available;
}

Context* CreateContext(const Creation& creation) {
	ffxCreateContextDescUpscale        desc {};
	ffxCreateContextDescUpscaleVersion version {};
	ffxCreateBackendVKDesc             backend {};
	SetCreation(desc, creation);
	version.header.type   = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
	version.version       = FFX_UPSCALER_VERSION;
	backend.header.type   = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backend.deviceContext = g_device;
	desc.header.pNext     = &version.header;
	version.header.pNext  = &backend.header;
	ffxContext context    = nullptr;
	if (const auto result = g_api.create(&context, &desc.header, nullptr);
	    result != FFX_API_RETURN_OK) {
		Warn(fmt::format("creating a context up to {}x{} failed ({})", creation.output_width,
		                 creation.output_height, result));
		return nullptr;
	}
	ffxQueryGetProviderVersion provider {};
	provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
	const bool known     = g_api.query(&context, &provider.header) == FFX_API_RETURN_OK &&
	                   provider.versionName != nullptr;
	Log::WriteToConsoleAndLog(fmt::format("FSR: {}x{} context uses {}\n", creation.output_width,
	                                      creation.output_height,
	                                      known ? provider.versionName : "an unknown upscaler"));
	return static_cast<Context*>(context);
}

bool Dispatch(vk::CommandBuffer command, Context* context, const Pssr::Request& request,
              const Inputs& inputs, bool reset, float frame_ms) {
	ffxDispatchDescUpscale desc {};
	SetDispatch(desc, static_cast<VkCommandBuffer>(command), request, inputs, reset, frame_ms);
	ffxContext handle = context;
	if (const auto result = g_api.dispatch(&handle, &desc.header); result != FFX_API_RETURN_OK) {
		Warn(fmt::format("dispatch failed ({})", result));
		return false;
	}
	return true;
}

void ReleaseContext(Context* context) {
	ffxContext handle = context;
	g_api.destroy(&handle, nullptr);
}

#else

uint32_t CreationFlags(const Pssr::Request&, bool) {
	return 0;
}

Creation Choose(const Pssr::Request&, const Inputs& inputs) {
	return {inputs.output.width, inputs.output.height, 0};
}

bool Takes(const Creation&, const Pssr::Request&, const Inputs&) {
	return false;
}

void Load() {
	if (Config::Ps5ProEnabled() && !Config::GetFsrFolder().empty()) {
		Log::WriteToConsoleAndLog("WARNING: FSR: the AMD FSR SDK runs on Windows only; PSSR scales "
		                          "linearly.\n");
	}
}

bool Loaded() {
	return false;
}

DeviceFeatures AppendDeviceExtensions(vk::PhysicalDevice, std::vector<const char*>&,
                                      const std::vector<vk::ExtensionProperties>&) {
	return {};
}

void Init(const GraphicContext&, PFN_vkGetDeviceProcAddr) {}

void Shutdown() {}

bool Available() {
	return false;
}

Context* CreateContext(const Creation&) {
	return nullptr;
}

bool Dispatch(vk::CommandBuffer, Context*, const Pssr::Request&, const Inputs&, bool, float) {
	return false;
}

void ReleaseContext(Context*) {}

#endif

} // namespace Libs::Graphics::Fsr
