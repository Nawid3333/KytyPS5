#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <optional>

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

// DLSS reads and writes with stages and accesses of its own.
void MemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

} // namespace

RenderExecutor::~RenderExecutor() {
	for (const auto& context: m_dlss_contexts) {
		if (context.feature != nullptr) {
			Dlss::ReleaseFeature(context.feature);
		}
	}
	for (const auto& context: m_fsr_contexts) {
		if (context.handle != nullptr) {
			Fsr::ReleaseContext(context.handle);
		}
	}
	const auto device = m_context.GetGraphics().device;
	device.destroyPipeline(m_pssr_pipeline, nullptr);
	device.destroyPipelineLayout(m_pssr_pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(m_pssr_descriptor_layout, nullptr);
}

bool RenderExecutor::Upscale(vk::CommandBuffer command, const Pssr::Request& request,
                             Dlss::Inputs& inputs) {
	const auto release = [&](Dlss::Feature* feature) {
		if (feature != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [feature] { Dlss::ReleaseFeature(feature); });
		}
	};
	auto context = std::ranges::find(m_dlss_contexts, request.context, &DlssContext::context);
	bool reset   = (request.flags & Pssr::kReset) != 0;
	if (context == m_dlss_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
		// A render size no quality mode takes scales linearly.
		const auto creation = Dlss::Choose(*Dlss::Capabilities(), request, inputs);
		if (!creation) {
			return false;
		}
		if (context != m_dlss_contexts.end()) {
			release(context->feature);
		} else {
			// A library has at most four contexts.
			if (m_dlss_contexts.size() == 4) {
				release(m_dlss_contexts.front().feature);
				m_dlss_contexts.erase(m_dlss_contexts.begin());
			}
			context = m_dlss_contexts.insert(m_dlss_contexts.end(), {request.context, {}, nullptr});
		}
		context->creation = *creation;
		context->feature  = Dlss::CreateFeature(command, *creation);
		reset             = true;
	}
	if (context->feature == nullptr) {
		return false;
	}
	MemoryBarrier(command);
	if (!Dlss::Evaluate(command, context->feature, request, inputs, reset)) {
		release(context->feature);
		context->feature = nullptr;
		return false;
	}
	MemoryBarrier(command);
	return true;
}

bool RenderExecutor::UpscaleFsr(vk::CommandBuffer command, const Pssr::Request& request,
                                const Fsr::Inputs& inputs) {
	const auto release = [&](Fsr::Context* handle) {
		if (handle != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [handle] { Fsr::ReleaseContext(handle); });
		}
	};
	const auto now     = std::chrono::steady_clock::now();
	auto       context = std::ranges::find(m_fsr_contexts, request.context, &FsrContext::context);
	bool       reset   = (request.flags & Pssr::kReset) != 0;
	if (context != m_fsr_contexts.end()) {
		std::rotate(context, std::next(context), m_fsr_contexts.end());
		context = std::prev(m_fsr_contexts.end());
	}
	if (context == m_fsr_contexts.end() || !Fsr::Takes(context->creation, request, inputs)) {
		const auto creation = Fsr::Choose(request, inputs);
		if (!Fsr::Takes(creation, request, inputs)) {
			return false;
		}
		if (context != m_fsr_contexts.end()) {
			release(context->handle);
		} else {
			if (m_fsr_contexts.size() == 4) {
				release(m_fsr_contexts.front().handle);
				m_fsr_contexts.erase(m_fsr_contexts.begin());
			}
			context = m_fsr_contexts.insert(m_fsr_contexts.end(),
			                                 {request.context, {}, nullptr, now, false});
		}
		context->creation = creation;
		context->handle   = Fsr::CreateContext(creation);
		reset             = true;
	}
	if (context->handle == nullptr) {
		return false;
	}
	reset          = reset || context->stale;
	context->stale = false;
	const float frame_ms =
	    reset ? 0.0f : std::chrono::duration<float, std::milli>(now - context->last).count();
	context->last = now;
	MemoryBarrier(command);
	if (!Fsr::Dispatch(command, context->handle, request, inputs, reset, frame_ms)) {
		release(context->handle);
		context->handle = nullptr;
		return false;
	}
	MemoryBarrier(command);
	m_fsr_ticks[m_fsr_tick_slot] = m_context.GetCommandScheduler().CurrentTick();
	m_fsr_tick_slot              = (m_fsr_tick_slot + 1) % m_fsr_ticks.size();
	return true;
}

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	ShaderTextureResource input_descriptor {}, output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	const auto is_2d = [](const ShaderTextureResource& descriptor) {
		return !descriptor.IsNull() &&
		       (descriptor.Type() == Prospero::ImageType::kColor2D ||
		        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
		       descriptor.TileMode() != Prospero::TileMode::kDepth;
	};
	EXIT_IF(!is_2d(input_descriptor) || !is_2d(output_descriptor) ||
	        !Pssr::IsColorFormat(input_descriptor.Format()) ||
	        output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	        output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	        output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	        output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	        output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0);
	const auto input_width = std::max(1u, (input_descriptor.Width5() + 1u) >>
	                                         input_descriptor.BaseLevel());
	const auto input_height = std::max(1u, (input_descriptor.Height5() + 1u) >>
	                                          input_descriptor.BaseLevel());
	EXIT_IF(request.width > input_width || request.height > input_height);
	PssrParameters parameters {{request.width, request.height},
	                           {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	                           {4, 4, 4, 4}};
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class = Prospero::TextureNumericClass::Float;
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read = !output;
		resource.written = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod = 0;
		return binding;
	};
	auto& cache = m_context.GetTextureCache();
	auto input = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	EXIT_IF(input.desc.info.data.address < output.desc.info.data.End() &&
	        output.desc.info.data.address < input.desc.info.data.End());
	// Depth, motion vectors and exposure for DLSS, which writes the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	const bool dlss = Dlss::Capabilities() != nullptr && Dlss::Accepts(request);
	const bool fsr  = !dlss && Fsr::Available() && Fsr::Accepts(request);
	if ((dlss || fsr) &&
	    parameters.output_channels[0] == 0 && parameters.output_channels[1] == 1 &&
	    parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto& source = cache.GetImage(input.image_id);
	auto& destination = cache.GetImage(output.image_id);
	const auto& source_view = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	EXIT_IF(source.info.IsDepth() || destination.info.IsDepth() ||
	        source.backing.samples != 1 || destination.backing.samples != 1 ||
	        (input.image_id == output.image_id &&
	         source_view.base_level == destination_view.base_level &&
	         source_view.base_layer == destination_view.base_layer));

	{
		static std::atomic_uint64_t lab_requests = 0;
		const auto n = lab_requests.fetch_add(1, std::memory_order_relaxed) + 1;
		if (n <= 12 || n % 600 == 0) {
			ShaderTextureResource lab_depth {}, lab_motion {}, lab_exposure {};
			std::copy_n(request.depth, 8, lab_depth.fields);
			std::copy_n(request.motion, 8, lab_motion.fields);
			std::copy_n(request.exposure, 8, lab_exposure.fields);
			Log::WriteToConsoleAndLog(fmt::format(
			    "lab-pssr: request={} context=0x{:x} render={}x{} at {},{} input_desc={}x{} fmt={} "
			    "input_image={}x{} mips={} layers={} view_level={} view_layer={} vkfmt={} "
			    "mutable={} output={}x{} vkfmt={} depth={} motion={} exposure={} flags=0x{:x} "
			    "jitter={:.3f},{:.3f} mv_scale={:.1f},{:.1f} pre_exposure={:.3f} channels={}{}{} "
			    "dlss={} fsr_available={} fsr_accepts={}\n",
			    n, request.context, request.width, request.height, request.x, request.y,
			    input_descriptor.Width5() + 1u, input_descriptor.Height5() + 1u,
			    static_cast<uint32_t>(input_descriptor.Format()), source.backing.extent.width,
			    source.backing.extent.height, source.backing.mip_levels, source.backing.layers,
			    source_view.base_level, source_view.base_layer,
			    static_cast<int>(source_view.format),
			    (source.backing.flags & vk::ImageCreateFlagBits::eMutableFormat) ? 1 : 0,
			    destination.backing.extent.width, destination.backing.extent.height,
			    static_cast<int>(destination_view.format), lab_depth.IsNull() ? 0 : 1,
			    lab_motion.IsNull() ? 0 : 1, lab_exposure.IsNull() ? 0 : 1, request.flags,
			    request.jitter[0], request.jitter[1], request.motion_scale[0],
			    request.motion_scale[1], request.pre_exposure, parameters.output_channels[0],
			    parameters.output_channels[1], parameters.output_channels[2], dlss ? 1 : 0,
			    Fsr::Available() ? 1 : 0, Fsr::Accepts(request) ? 1 : 0));
		}
	}
	if (fsr) {
		auto&      scheduler = m_context.GetCommandScheduler();
		const auto oldest    = m_fsr_ticks[m_fsr_tick_slot];
		if (oldest != 0 && !scheduler.IsFree(oldest)) {
			scheduler.Wait(oldest);
		}
	}
	buffer.EndRendering();
	const auto command = buffer.Handle();
	const auto range   = [](const ImageViewInfo& view) {
		return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	const auto whole_range = [&](ImageId id) {
		const auto& image = cache.GetImage(id);
		return ImageSubresourceRange {0, image.backing.mip_levels, 0, image.backing.layers};
	};
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, range(source_view),
	               command);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    range(destination_view), command);
	if (fsr && dlss_bindings[0] && dlss_bindings[1]) {
		const auto whole = [&](const TextureBinding& binding, bool output, uint32_t channels) {
			const auto& image = cache.GetImage(binding.image_id);
			const auto& view  = binding.desc.view_info;
			const bool  depth = image.info.IsDepth();
			const auto  format = static_cast<VkFormat>(depth ? image.backing.format : view.format);
			const bool  sized  = !output || (image.backing.extent.width == parameters.output_size[0] &&
			                                 image.backing.extent.height == parameters.output_size[1]);
			return std::optional<Fsr::Resource> {
			    view.base_level == 0 && view.base_layer == 0 && image.backing.layers == 1 && sized &&
			            Fsr::IdentityChannels(static_cast<VkComponentMapping>(view.mapping), channels) &&
			            Fsr::Viewable(static_cast<VkFormat>(image.backing.format),
			                          static_cast<VkImageCreateFlags>(image.backing.flags), format,
			                          image.backing.samples)
			        ? std::optional<Fsr::Resource> {Fsr::Resource {
			              .image      = static_cast<VkImage>(image.backing.image),
			              .format     = format,
			              .width      = image.backing.extent.width,
			              .height     = image.backing.extent.height,
			              .mip_levels = image.backing.mip_levels,
			              .layers     = image.backing.layers,
			              .usage      = static_cast<VkImageUsageFlags>(image.backing.usage),
			              .depth      = depth,
			              .output     = output}}
			        : std::nullopt};
		};
		const auto color  = whole(input, false, 3);
		const auto depth  = whole(*dlss_bindings[0], false, 0);
		const auto motion = whole(*dlss_bindings[1], false, 2);
		const auto target = whole(output, true, 0);
		const auto exposure =
		    dlss_bindings[2] ? whole(*dlss_bindings[2], false, 1) : std::optional<Fsr::Resource> {};
		if (color && depth && motion && target && (!dlss_bindings[2] || exposure)) {
			for (const auto* binding: {&input, &*dlss_bindings[0], &*dlss_bindings[1]}) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
				             whole_range(binding->image_id), command);
			}
			if (dlss_bindings[2]) {
				cache.GetImage(dlss_bindings[2]->image_id)
				    .Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
				             whole_range(dlss_bindings[2]->image_id), command);
			}
			destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
			                    whole_range(output.image_id), command);
			const Fsr::Inputs inputs {*color, *depth, *motion, exposure ? &*exposure : nullptr,
			                         *target};
			if (UpscaleFsr(command, request, inputs)) {
				ResetBindings();
				return;
			}
			source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
			               range(source_view), command);
		}
	}
	if (dlss && dlss_bindings[0]) {
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto  extent = binding.desc.info.MipExtent(view.base_level);
			// DLSS takes a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
				               view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = extent.width,
			    .height     = extent.height,
			    .read_write = write};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					         range(binding->desc.view_info), command);
			}
		}
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		if (Upscale(command, request, inputs)) {
			ResetBindings();
			return;
		}
	}
	if (auto stale = std::ranges::find(m_fsr_contexts, request.context, &FsrContext::context);
	    stale != m_fsr_contexts.end()) {
		stale->stale = true;
	}
	auto& graphics = m_context.GetGraphics();
	if (m_pssr_pipeline == nullptr) {
		const std::array bindings {
		    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = bindings.size();
		descriptor_info.pBindings = bindings.data();
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(
		                         &descriptor_info, nullptr, &m_pssr_descriptor_layout),
		                     "create upscaling descriptor layout");
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
		                                  sizeof(PssrParameters)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount = 1;
		layout_info.pSetLayouts = &m_pssr_descriptor_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(
		                         &layout_info, nullptr, &m_pssr_pipeline_layout),
		                     "create upscaling pipeline layout");
		const auto shader = CompileSPV(GPU_PSSR_LINEAR_SPV, graphics.device);
		vk::ComputePipelineCreateInfo create {};
		create.stage.stage = vk::ShaderStageFlagBits::eCompute;
		create.stage.module = shader;
		create.stage.pName = "main";
		create.layout = m_pssr_pipeline_layout;
		const auto result = graphics.device.createComputePipelines(
		    nullptr, 1, &create, nullptr, &m_pssr_pipeline);
		graphics.device.destroyShaderModule(shader, nullptr);
		RequireVulkanSuccess(result, "create upscaling pipeline");
	}
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, input.image_view, vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, output.image_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampledImage
		                                 : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo = &images[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pssr_pipeline_layout,
	                             0, writes.size(), writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pssr_pipeline);
	command.pushConstants(m_pssr_pipeline_layout, vk::ShaderStageFlagBits::eCompute,
	                      0, sizeof(parameters), &parameters);
	command.dispatch((parameters.output_size[0] + 7) / 8,
	                 (parameters.output_size[1] + 7) / 8, 1);
	ShaderAccessBarrier(command, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
