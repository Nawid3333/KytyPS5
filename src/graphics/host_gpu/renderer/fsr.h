#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ffxCreateContextDescUpscale;
struct ffxDispatchDescUpscale;
struct FfxApiResource;

namespace Libs::Graphics {
struct GraphicContext;
namespace Pssr {
struct Request;
}
}

namespace Libs::Graphics::Fsr {

struct Context;

struct Resource {
	VkImage            image;
	VkFormat           format;
	uint32_t           width;
	uint32_t           height;
	uint32_t           mip_levels;
	uint32_t           layers;
	VkImageUsageFlags  usage;
	bool               depth;
	bool               output;
};

struct Inputs {
	Resource        color;
	Resource        depth;
	Resource        motion;
	const Resource* exposure;
	Resource        output;
};

struct Creation {
	uint32_t output_width;
	uint32_t output_height;
	uint32_t flags;

	bool operator==(const Creation&) const = default;
};

struct DeviceFeatures {
	bool cooperative_matrix;
	bool float8;
	bool float8_cooperative_matrix;
	bool integer_dot_product;
	bool float16;
	bool int16;
	bool int8;
	bool subgroup_extended_types;
};

[[nodiscard]] bool     Accepts(const Pssr::Request& request);
[[nodiscard]] uint32_t CreationFlags(const Pssr::Request& request, bool exposure);
[[nodiscard]] Creation Choose(const Pssr::Request& request, const Inputs& inputs);
[[nodiscard]] bool     Takes(const Creation& creation, const Pssr::Request& request,
                             const Inputs& inputs);
[[nodiscard]] bool     Viewable(VkFormat image_format, VkImageCreateFlags image_flags,
                                VkFormat view_format, uint32_t samples);
[[nodiscard]] FfxApiResource MakeResource(const Resource* resource);
void SetCreation(ffxCreateContextDescUpscale& desc, const Creation& creation);
void SetDispatch(ffxDispatchDescUpscale& desc, VkCommandBuffer command,
                 const Pssr::Request& request, const Inputs& inputs, bool reset, float frame_ms);

void Load();
[[nodiscard]] bool Loaded();
[[nodiscard]] DeviceFeatures AppendDeviceExtensions(
    vk::PhysicalDevice physical_device, std::vector<const char*>& out,
    const std::vector<vk::ExtensionProperties>& available);
void Init(const GraphicContext& graphics, PFN_vkGetDeviceProcAddr get_device_proc_addr);
void               Shutdown();
[[nodiscard]] bool Available();

[[nodiscard]] Context* CreateContext(const Creation& creation);
[[nodiscard]] bool     Dispatch(vk::CommandBuffer command, Context* context,
                                const Pssr::Request& request, const Inputs& inputs, bool reset,
                                float frame_ms);
void                   ReleaseContext(Context* context);

}

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR_H_
