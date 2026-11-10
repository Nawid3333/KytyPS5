#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/renderer/fsr.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include "api/include/vk/ffx_api_vk.h"
#include "upscalers/include/ffx_upscale.h"
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using namespace Libs::Graphics;

Pssr::Request TemporalRequest() {
	Pssr::Request request {};
	request.depth[0]  = 0x100;
	request.motion[0] = 0x200;
	request.width     = 1280;
	request.height    = 720;
	return request;
}

Fsr::Resource Image(VkFormat format, uint32_t width, uint32_t height, bool output = false) {
	return {.image      = reinterpret_cast<VkImage>(uintptr_t {0x1000} + width),
	        .format     = format,
	        .width      = width,
	        .height     = height,
	        .mip_levels = 1,
	        .layers     = 1,
	        .usage      = output ? VkImageUsageFlags {VK_IMAGE_USAGE_STORAGE_BIT}
	                             : VkImageUsageFlags {VK_IMAGE_USAGE_SAMPLED_BIT},
	        .depth      = format == VK_FORMAT_D32_SFLOAT,
	        .output     = output};
}

Fsr::Inputs MakeInputs(const Fsr::Resource* exposure) {
	return {Image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 1280, 720),
	        Image(VK_FORMAT_D32_SFLOAT, 1280, 720), Image(VK_FORMAT_R16G16_SFLOAT, 1280, 720),
	        exposure, Image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 2560, 1440, true)};
}

TEST(FsrTest, TemporalRequestsNeedDepthMotionAndNoOffset) {
	auto request = TemporalRequest();
	EXPECT_TRUE(Fsr::Accepts(request));
	request.flags = Pssr::kSpatialOnly;
	EXPECT_FALSE(Fsr::Accepts(request));
	request          = TemporalRequest();
	request.depth[0] = 0;
	EXPECT_FALSE(Fsr::Accepts(request));
	request           = TemporalRequest();
	request.motion[0] = 0;
	EXPECT_FALSE(Fsr::Accepts(request));
	request   = TemporalRequest();
	request.x = 8;
	EXPECT_FALSE(Fsr::Accepts(request));
	request   = TemporalRequest();
	request.y = 8;
	EXPECT_FALSE(Fsr::Accepts(request));
}

TEST(FsrTest, ViewableNeedsOneSampleAndItsOwnOrAMutableFormat) {
	EXPECT_TRUE(Fsr::Viewable(VK_FORMAT_R16G16_SFLOAT, 0, VK_FORMAT_R16G16_SFLOAT, 1));
	EXPECT_FALSE(Fsr::Viewable(VK_FORMAT_R16G16_SFLOAT, 0, VK_FORMAT_R16G16_SFLOAT, 4));
	EXPECT_FALSE(Fsr::Viewable(VK_FORMAT_R32_UINT, 0, VK_FORMAT_R32_SFLOAT, 1));
	EXPECT_TRUE(Fsr::Viewable(VK_FORMAT_R32_UINT, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
	                          VK_FORMAT_R32_SFLOAT, 1));
}

TEST(FsrTest, ChannelsMustNotBeSwizzled) {
	const VkComponentMapping identity {};
	const VkComponentMapping own {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
	                              VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
	const VkComponentMapping bgra {VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_G,
	                               VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A};
	const VkComponentMapping one_alpha {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
	                                    VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_ONE};
	EXPECT_TRUE(Fsr::IdentityChannels(identity, 3));
	EXPECT_TRUE(Fsr::IdentityChannels(own, 4));
	EXPECT_FALSE(Fsr::IdentityChannels(bgra, 3));
	EXPECT_TRUE(Fsr::IdentityChannels(bgra, 0));
	EXPECT_TRUE(Fsr::IdentityChannels(one_alpha, 3));
	EXPECT_FALSE(Fsr::IdentityChannels(one_alpha, 4));
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

TEST(FsrTest, CreationFlagsFollowTheRequest) {
	auto            request = TemporalRequest();
	const uint32_t  always  = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
	                         FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
	EXPECT_EQ(Fsr::CreationFlags(request, true), always);
	EXPECT_EQ(Fsr::CreationFlags(request, false), always | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE);
	request.flags = Pssr::kReverseDepth | Pssr::kAutoExposure;
	EXPECT_EQ(Fsr::CreationFlags(request, true),
	          always | FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE);
	request.flags = Pssr::kReset;
	EXPECT_EQ(Fsr::CreationFlags(request, true), always);
}

TEST(FsrTest, ContextsTakeRenderSizesUpToTheOutput) {
	auto       request  = TemporalRequest();
	const auto inputs   = MakeInputs(nullptr);
	const auto creation = Fsr::Choose(request, inputs);
	EXPECT_EQ(creation.output_width, 2560u);
	EXPECT_EQ(creation.output_height, 1440u);
	EXPECT_TRUE(Fsr::Takes(creation, request, inputs));
	request.width  = 2560;
	request.height = 1440;
	EXPECT_TRUE(Fsr::Takes(creation, request, inputs));
	request.width = 2561;
	EXPECT_FALSE(Fsr::Takes(creation, request, inputs));
	request       = TemporalRequest();
	request.flags = Pssr::kReverseDepth;
	EXPECT_FALSE(Fsr::Takes(creation, request, inputs));
	request          = TemporalRequest();
	auto larger      = inputs;
	larger.output    = Image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 3840, 2160, true);
	EXPECT_FALSE(Fsr::Takes(creation, request, larger));
	const auto exposure = Image(VK_FORMAT_R32_SFLOAT, 1, 1);
	EXPECT_FALSE(Fsr::Takes(creation, request, MakeInputs(&exposure)));
}

TEST(FsrTest, CreationSetsTheMaximumSizesAndFlags) {
	ffxCreateContextDescUpscale desc {};
	desc.flags = 0xffffffffu;
	Fsr::SetCreation(desc, {2560, 1440, FFX_UPSCALE_ENABLE_DEPTH_INVERTED});
	EXPECT_EQ(desc.header.type, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE);
	EXPECT_EQ(desc.header.pNext, nullptr);
	EXPECT_EQ(desc.flags, uint32_t {FFX_UPSCALE_ENABLE_DEPTH_INVERTED});
	EXPECT_EQ(desc.maxRenderSize.width, 2560u);
	EXPECT_EQ(desc.maxRenderSize.height, 1440u);
	EXPECT_EQ(desc.maxUpscaleSize.width, 2560u);
	EXPECT_EQ(desc.maxUpscaleSize.height, 1440u);
	EXPECT_NE(desc.fpMessage, nullptr);
}

TEST(FsrTest, ResourcesDescribeWholeImages) {
	const auto inputs = MakeInputs(nullptr);
	const auto color  = Fsr::MakeResource(&inputs.color);
	EXPECT_EQ(color.description.type, uint32_t {FFX_API_RESOURCE_TYPE_TEXTURE2D});
	EXPECT_EQ(color.description.format, uint32_t {FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT});
	EXPECT_EQ(color.description.width, 1280u);
	EXPECT_EQ(color.description.height, 720u);
	EXPECT_EQ(color.description.mipCount, 1u);
	EXPECT_EQ(color.description.usage, uint32_t {FFX_API_RESOURCE_USAGE_READ_ONLY});
	EXPECT_EQ(color.state, uint32_t {FFX_API_RESOURCE_STATE_COMPUTE_READ});
	const auto depth_image = Image(VK_FORMAT_D32_SFLOAT, 1280, 720);
	const auto depth       = Fsr::MakeResource(&depth_image);
	EXPECT_EQ(depth.description.format, uint32_t {FFX_API_SURFACE_FORMAT_R32_FLOAT});
	EXPECT_EQ(depth.description.usage, uint32_t {FFX_API_RESOURCE_USAGE_DEPTHTARGET});
	auto stencil_image   = depth_image;
	stencil_image.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
	const auto stencil   = Fsr::MakeResource(&stencil_image);
	EXPECT_EQ(stencil.description.format, uint32_t {FFX_API_SURFACE_FORMAT_R32_FLOAT});
	EXPECT_EQ(stencil.description.usage,
	          uint32_t {FFX_API_RESOURCE_USAGE_DEPTHTARGET | FFX_API_RESOURCE_USAGE_STENCILTARGET});
	const auto output_image = Image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 2560, 1440, true);
	const auto output       = Fsr::MakeResource(&output_image);
	EXPECT_EQ(output.description.usage, uint32_t {FFX_API_RESOURCE_USAGE_UAV});
	EXPECT_EQ(output.state, uint32_t {FFX_API_RESOURCE_STATE_UNORDERED_ACCESS});
	EXPECT_EQ(output.resource, reinterpret_cast<void*>(output_image.image));
	const auto absent = Fsr::MakeResource(nullptr);
	EXPECT_EQ(absent.resource, nullptr);
}

TEST(FsrTest, DispatchNegatesTheJitterAndPassesTheRequest) {
	auto request            = TemporalRequest();
	request.jitter[0]       = 0.25f;
	request.jitter[1]       = -0.375f;
	request.motion_scale[0] = 1280.0f;
	request.motion_scale[1] = -720.0f;
	request.pre_exposure    = 2.0f;
	const auto             exposure = Image(VK_FORMAT_R32_SFLOAT, 1, 1);
	const auto             inputs   = MakeInputs(&exposure);
	const auto             command  = reinterpret_cast<VkCommandBuffer>(uintptr_t {0x42});
	ffxDispatchDescUpscale desc {};
	Fsr::SetDispatch(desc, command, request, inputs, true, 8.0f);
	EXPECT_EQ(desc.header.type, FFX_API_DISPATCH_DESC_TYPE_UPSCALE);
	EXPECT_EQ(desc.commandList, static_cast<void*>(command));
	EXPECT_EQ(desc.jitterOffset.x, -0.25f);
	EXPECT_EQ(desc.jitterOffset.y, 0.375f);
	EXPECT_EQ(desc.motionVectorScale.x, 1280.0f);
	EXPECT_EQ(desc.motionVectorScale.y, -720.0f);
	EXPECT_EQ(desc.renderSize.width, 1280u);
	EXPECT_EQ(desc.renderSize.height, 720u);
	EXPECT_EQ(desc.upscaleSize.width, 2560u);
	EXPECT_EQ(desc.upscaleSize.height, 1440u);
	EXPECT_EQ(desc.preExposure, 2.0f);
	EXPECT_EQ(desc.frameTimeDelta, 8.0f);
	EXPECT_TRUE(desc.reset);
	EXPECT_FALSE(desc.enableSharpening);
	EXPECT_EQ(desc.exposure.resource, reinterpret_cast<void*>(exposure.image));
	request.flags = Pssr::kAutoExposure;
	Fsr::SetDispatch(desc, command, request, inputs, true, 8.0f);
	EXPECT_EQ(desc.exposure.resource, nullptr);
	EXPECT_EQ(desc.reactive.resource, nullptr);
	EXPECT_EQ(desc.transparencyAndComposition.resource, nullptr);
	EXPECT_GT(desc.cameraFar, desc.cameraNear);
	EXPECT_GT(desc.cameraFovAngleVertical, 0.0f);
}

TEST(FsrTest, DispatchReplacesInvalidExposureAndFrameTimes) {
	auto request = TemporalRequest();
	for (const float pre_exposure: {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
	                                std::numeric_limits<float>::infinity()}) {
		request.pre_exposure = pre_exposure;
		ffxDispatchDescUpscale desc {};
		Fsr::SetDispatch(desc, nullptr, request, MakeInputs(nullptr), false, pre_exposure);
		EXPECT_EQ(desc.preExposure, 1.0f) << pre_exposure;
		EXPECT_TRUE(std::isfinite(desc.frameTimeDelta) && desc.frameTimeDelta > 0.0f);
		EXPECT_EQ(desc.exposure.resource, nullptr);
		EXPECT_FALSE(desc.reset);
	}
}

TEST(FsrTest, DispatchTakesTheCameraOfTheRequest) {
	struct Camera {
		float near_plane;
		float far_plane;
		float y_scale;
		float vertical_fov;
	};
	constexpr float tolerance = 1e-5f;
	const Camera    ghost_of_yotei {10.0f, 1e7f, 1.830488f, 1.0f};
	const Camera    demons_souls {0.1f, 3000.0f, -2.53865f, 0.750491f};
	for (const auto& camera: {ghost_of_yotei, demons_souls}) {
		auto request                = TemporalRequest();
		request.camera_near         = camera.near_plane;
		request.camera_far          = camera.far_plane;
		request.projection_scale[1] = camera.y_scale;
		ffxDispatchDescUpscale desc {};
		Fsr::SetDispatch(desc, nullptr, request, MakeInputs(nullptr), false, 8.0f);
		EXPECT_EQ(desc.cameraNear, camera.near_plane);
		EXPECT_EQ(desc.cameraFar, camera.far_plane);
		EXPECT_NEAR(desc.cameraFovAngleVertical, camera.vertical_fov, tolerance);
		EXPECT_EQ(desc.viewSpaceToMetersFactor, 1.0f);
	}

	ffxDispatchDescUpscale fallback {};
	Fsr::SetDispatch(fallback, nullptr, TemporalRequest(), MakeInputs(nullptr), false, 8.0f);
	EXPECT_EQ(fallback.cameraNear, 0.1f);
	EXPECT_EQ(fallback.cameraFar, 10000.0f);
	EXPECT_NEAR(fallback.cameraFovAngleVertical, std::numbers::pi_v<float> / 3.0f, tolerance);

	const float nan      = std::numeric_limits<float>::quiet_NaN();
	const float infinity = std::numeric_limits<float>::infinity();
	for (const auto& camera: {Camera {nan, 3000.0f, nan}, Camera {0.1f, infinity, infinity},
	                          Camera {-0.1f, 3000.0f, 0.0f}, Camera {5.0f, 5.0f, -0.0f}}) {
		auto request                = TemporalRequest();
		request.camera_near         = camera.near_plane;
		request.camera_far          = camera.far_plane;
		request.projection_scale[1] = camera.y_scale;
		ffxDispatchDescUpscale desc {};
		Fsr::SetDispatch(desc, nullptr, request, MakeInputs(nullptr), false, 8.0f);
		EXPECT_EQ(desc.cameraNear, fallback.cameraNear) << camera.near_plane;
		EXPECT_EQ(desc.cameraFar, fallback.cameraFar) << camera.far_plane;
		EXPECT_EQ(desc.cameraFovAngleVertical, fallback.cameraFovAngleVertical) << camera.y_scale;
	}
}

#endif

TEST(FsrTest, LoadsOnlyInProModeWithTheDlls) {
	Config::Initialize();
	for (const auto& [pro, folder]: {std::pair {false, "fsr"}, {true, ""}, {true, "no-such-fsr"}}) {
		Config::ConfigOptions options;
		options.pro_enabled = pro;
		options.fsr_folder  = folder;
		Config::Load(options);
		Fsr::Load();
		EXPECT_FALSE(Fsr::Loaded()) << pro << folder;
		EXPECT_FALSE(Fsr::Available());
		Fsr::Shutdown();
	}
	Config::Shutdown();
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
