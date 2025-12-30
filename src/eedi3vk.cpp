#include <VSHelper4.h>
#include <VapourSynth4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vulkan/EEDI3Pipelines.hpp"
#include "vulkan/Params.hpp"
#include "vulkan/VulkanComputePipeline.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/VulkanMemory.hpp"
#include "vulkan/VulkanResourcePool.hpp"

using namespace std::literals;

namespace {

constexpr int MARGIN_H = 12;
constexpr int MARGIN_V = 4;

struct EEDI3Data {
    VSNode *node, *sclip, *mclip;
    VSVideoInfo vi;
    VSVideoFormat padFormat;
    int field, nrad, mdis, vcheck;
    std::array<bool, 3> process;
    bool dh, ucubic, cost3;
    float alpha, beta, gamma, vthresh2;
    float remainingWeight, rcpVthresh0, rcpVthresh1, rcpVthresh2;
    int tpitch;
};

struct EEDI3VKData {
    EEDI3Data d;

    int device_id;
    std::shared_ptr<eedi3vk::VulkanContext> context;
    std::unique_ptr<eedi3vk::VulkanMemory> memory;
    std::unique_ptr<eedi3vk::EEDI3Pipelines> pipelines;
    std::unique_ptr<eedi3vk::VulkanResourcePool> resource_pool;
    std::unique_ptr<eedi3vk::DescriptorPool> descriptor_pool;

    size_t vk_stride;
    int vk_stride_pixels;
};

inline void copyPad(const VSFrame* src, VSFrame* dst, const int plane,
                    const bool dh, const int off, const VSAPI* vsapi) noexcept {
    const int srcWidth = vsapi->getFrameWidth(src, plane);
    const int dstWidth = vsapi->getFrameWidth(dst, 0);
    const int srcHeight = vsapi->getFrameHeight(src, plane);
    const int dstHeight = vsapi->getFrameHeight(dst, 0);
    const ptrdiff_t srcStride =
        vsapi->getStride(src, plane) / static_cast<ptrdiff_t>(sizeof(float));
    const ptrdiff_t dstStride =
        vsapi->getStride(dst, 0) / static_cast<ptrdiff_t>(sizeof(float));
    const auto* srcp =
        reinterpret_cast<const float*>(vsapi->getReadPtr(src, plane));
    auto* VS_RESTRICT dstp =
        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));

    if (!dh) {
        vsh::bitblt(dstp + (dstStride * (MARGIN_V + off)) + MARGIN_H,
                    vsapi->getStride(dst, 0) * 2, srcp + (srcStride * off),
                    vsapi->getStride(src, plane) * 2, srcWidth * sizeof(float),
                    srcHeight / 2);
    } else {
        vsh::bitblt(dstp + (dstStride * (MARGIN_V + off)) + MARGIN_H,
                    vsapi->getStride(dst, 0) * 2, srcp,
                    vsapi->getStride(src, plane), srcWidth * sizeof(float),
                    srcHeight);
    }

    dstp += dstStride * (MARGIN_V + off);

    for (int y = MARGIN_V + off; y < dstHeight - MARGIN_V; y += 2) {
        for (int x = 0; x < MARGIN_H; x++) {
            dstp[x] = dstp[(MARGIN_H * 2) - x];
        }

        for (int x = dstWidth - MARGIN_H, c = 2; x < dstWidth; x++, c += 2) {
            dstp[x] = dstp[x - c];
        }

        dstp += dstStride * 2;
    }

    dstp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));

    for (int y = off; y < MARGIN_V; y += 2) {
        memcpy(dstp + (dstStride * y), dstp + (dstStride * (MARGIN_V * 2 - y)),
               dstWidth * sizeof(float));
    }

    for (int y = dstHeight - MARGIN_V + off, c = 2 + (2 * off); y < dstHeight;
         y += 2, c += 4) {
        memcpy(dstp + (dstStride * y), dstp + (dstStride * (y - c)),
               dstWidth * sizeof(float));
    }
}

inline void vCheck(const float* srcp, const float* scpp,
                   float* VS_RESTRICT dstp, const int* dmap,
                   float* VS_RESTRICT tline, const int field_n, const int width,
                   const int height, const ptrdiff_t srcStride,
                   const ptrdiff_t dstStride,
                   const EEDI3Data* VS_RESTRICT d) noexcept {
    for (int y = MARGIN_V + field_n; y < height - MARGIN_V; y += 2) {
        if (y >= 6 && y < height - 6) {
            const auto* dst3p = srcp - (srcStride * 3) + MARGIN_H;
            auto* dst2p = dstp - (dstStride * 2);
            auto* dst1p = dstp - dstStride;
            auto* dst1n = dstp + dstStride;
            auto* dst2n = dstp + (dstStride * 2);
            const auto* dst3n = srcp + (srcStride * 3) + MARGIN_H;

            for (int x = 0; x < width; x++) {
                const int dirc = dmap[x];
                float cint = 0.0F;
                if (scpp != nullptr) {
                    cint = scpp[x];
                } else {
                    cint = (0.5625F * (dst1p[x] + dst1n[x])) -
                           (0.0625F * (dst3p[x] + dst3n[x]));
                }

                if (dirc == 0) {
                    tline[x] = cint;
                    continue;
                }

                const int dirt = dmap[x - width];
                const int dirb = dmap[x + width];

                if (std::max(dirc * dirt, dirc * dirb) < 0 ||
                    (dirt == dirb && dirt == 0)) {
                    tline[x] = cint;
                    continue;
                }

                const float it = (dst2p[x + dirc] + dstp[x - dirc]) / 2.0F;
                const float ib = (dstp[x + dirc] + dst2n[x - dirc]) / 2.0F;
                const float vt = std::abs(dst2p[x + dirc] - dst1p[x + dirc]) +
                                 std::abs(dstp[x + dirc] - dst1p[x + dirc]);
                const float vb = std::abs(dst2n[x - dirc] - dst1n[x - dirc]) +
                                 std::abs(dstp[x - dirc] - dst1n[x - dirc]);
                const float vc =
                    std::abs(dstp[x] - dst1p[x]) + std::abs(dstp[x] - dst1n[x]);

                const float d0 = std::abs(it - dst1p[x]);
                const float d1 = std::abs(ib - dst1n[x]);
                const float d2 = std::abs(vt - vc);
                const float d3 = std::abs(vb - vc);

                float mdiff0 = 0.0F;
                if (d->vcheck == 1) {
                    mdiff0 = std::min(d0, d1);
                } else if (d->vcheck == 2) {
                    mdiff0 = (d0 + d1) / 2.0F;
                } else {
                    mdiff0 = std::max(d0, d1);
                }

                float mdiff1 = 0.0F;
                if (d->vcheck == 1) {
                    mdiff1 = std::min(d2, d3);
                } else if (d->vcheck == 2) {
                    mdiff1 = (d2 + d3) / 2.0F;
                } else {
                    mdiff1 = std::max(d2, d3);
                }

                const float a0 = mdiff0 * d->rcpVthresh0;
                const float a1 = mdiff1 * d->rcpVthresh1;
                const float a2 = std::max(
                    (d->vthresh2 - static_cast<float>(std::abs(dirc))) *
                        d->rcpVthresh2,
                    0.0F);
                const float a = std::min(std::max({a0, a1, a2}), 1.0F);

                tline[x] = ((1.0F - a) * dstp[x]) + (a * cint);
            }

            memcpy(dstp, tline, width * sizeof(float));
        }

        srcp += srcStride * 2;
        if (scpp != nullptr) {
            scpp += dstStride * 2;
        }
        dstp += dstStride * 2;
        dmap += width;
    }
}

inline uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

void updateDescriptorSet(vk::Device device, vk::DescriptorSet set,
                         std::span<const vk::Buffer> buffers) {
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    buffer_infos.reserve(buffers.size());
    for (auto buffer : buffers) {
        buffer_infos.emplace_back(buffer, 0, VK_WHOLE_SIZE);
    }
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        writes.emplace_back(set, static_cast<uint32_t>(i), 0, 1,
                            vk::DescriptorType::eStorageBuffer, nullptr,
                            &buffer_infos[i]);
    }
    device.updateDescriptorSets(writes, nullptr);
}

const VSFrame* VS_CC eedi3GetFrame(int n, int activationReason,
                                   void* instanceData,
                                   [[maybe_unused]] void** frameData,
                                   VSFrameContext* frameCtx, VSCore* core,
                                   const VSAPI* vsapi) {

    auto* vk_d = static_cast<EEDI3VKData*>(instanceData);
    auto* d = &vk_d->d;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);

        if (d->vcheck > 0 && d->sclip != nullptr) {
            vsapi->requestFrameFilter(n, d->sclip, frameCtx);
        }

        if (d->mclip != nullptr) {
            vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->mclip,
                                      frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        eedi3vk::ResourceHolder rh(*vk_d->resource_pool);
        eedi3vk::FrameResources& res = rh.get();

        const VSFrame* src =
            vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
        const VSFrame* mclip = nullptr;
        if (d->mclip != nullptr) {
            mclip = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->mclip,
                                          frameCtx);
        }

        VSFrame* dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width,
                                            d->vi.height, src, core);

        int field = d->field;
        if (field > 1) {
            field -= 2;
        }

        int err = 0;
        const int fieldBased = vsapi->mapGetIntSaturated(
            vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err);
        if (fieldBased == 1) {
            field = 0;
        } else if (fieldBased == 2) {
            field = 1;
        }

        int field_n = 0;
        if (d->field > 1) {
            if ((n & 1) != 0) {
                field_n = static_cast<int>(field == 0);
            } else {
                field_n = static_cast<int>(field == 1);
            }
        } else {
            field_n = field;
        }

        const bool use_dedicated_transfer =
            vk_d->context->hasDedicatedTransferQueue();

        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            if (!d->process.at(plane)) {
                const uint8_t* srcp = vsapi->getReadPtr(src, plane);
                uint8_t* dstp = vsapi->getWritePtr(dst, plane);
                int src_stride = static_cast<int>(vsapi->getStride(src, plane));
                int dst_stride = static_cast<int>(vsapi->getStride(dst, plane));
                int height = vsapi->getFrameHeight(src, plane);
                int row_size = vsapi->getFrameWidth(src, plane) *
                               static_cast<int>(sizeof(float));

                for (int y = 0; y < height; y++) {
                    std::memcpy(dstp + (y * dst_stride),
                                srcp + (y * src_stride), row_size);
                }
                continue;
            }

            const int plane_width = vsapi->getFrameWidth(src, plane);
            const int plane_height = vsapi->getFrameHeight(src, plane);
            const int dst_width = vsapi->getFrameWidth(dst, plane);
            const int dst_height = vsapi->getFrameHeight(dst, plane);
            const int field_height = (dst_height - field_n + 1) / 2;

            size_t plane_size_bytes = vk_d->vk_stride * plane_height;
            size_t dst_size_bytes = vk_d->vk_stride * dst_height;
            const bool need_dmap_readback = (d->vcheck > 0);
            VkDeviceSize dmap_size_bytes = 0;

            if (res.descriptor_sets.empty()) {
                auto& device = vk_d->context->getDevice();

                auto add_set = [&](vk::raii::DescriptorSet&& set,
                                   std::span<const vk::Buffer> buffers) {
                    updateDescriptorSet(*device, *set, buffers);
                    res.descriptor_sets.push_back(std::move(set));
                };

                add_set(
                    vk_d->pipelines->allocateCopyFieldSet(*res.descriptor_pool),
                    {{res.src_buffer.buffer, res.dst_buffer.buffer}});

                if (mclip != nullptr) {
                    add_set(
                        vk_d->pipelines->allocateDilateMaskSet(
                            *res.descriptor_pool),
                        {{res.mclip_buffer.buffer, res.bmask_buffer.buffer}});
                }

                add_set(
                    vk_d->pipelines->allocateCalcCostsSet(*res.descriptor_pool),
                    {{res.src_buffer.buffer, res.cost_buffer.buffer,
                      res.bmask_buffer.buffer}});

                add_set(vk_d->pipelines->allocateViterbiScanSet(
                            *res.descriptor_pool),
                        {{res.cost_buffer.buffer, res.dmap_buffer.buffer,
                          res.pbackt_buffer.buffer, res.bmask_buffer.buffer}});

                add_set(vk_d->pipelines->allocateInterpolateSet(
                            *res.descriptor_pool),
                        {{res.src_buffer.buffer, res.dst_buffer.buffer,
                          res.dmap_buffer.buffer, res.bmask_buffer.buffer}});
            }

            vk::MemoryBarrier transfer_to_compute_barrier{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead |
                    vk::AccessFlagBits::eShaderWrite};
            vk::MemoryBarrier compute_to_compute_barrier{
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead |
                    vk::AccessFlagBits::eShaderWrite};
            vk::MemoryBarrier readback_barrier{
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eTransferRead};

            auto record_compute = [&](vk::raii::CommandBuffer& cmd) {
                int set_idx = 0;
                auto dispatch = [&](eedi3vk::VulkanComputePipeline& pipeline,
                                    uint32_t gx, uint32_t gy, uint32_t gz,
                                    const void* params, size_t params_size) {
                    pipeline.dispatch(cmd, *res.descriptor_sets[set_idx++], gx,
                                      gy, gz, params,
                                      static_cast<uint32_t>(params_size));
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {},
                        compute_to_compute_barrier, nullptr, nullptr);
                };

                {
                    eedi3vk::CopyFieldParams params{};
                    params.width = dst_width;
                    params.height = dst_height;
                    params.field = field_n;
                    params.stride = vk_d->vk_stride_pixels;
                    params.dh = d->dh ? 1 : 0;

                    int copy_field_height =
                        (dst_height - (1 - field_n) + 1) / 2;
                    dispatch(vk_d->pipelines->getCopyField(),
                             divUp(dst_width, 16), divUp(copy_field_height, 16),
                             1, &params, sizeof(params));
                }

                if (mclip != nullptr) {
                    eedi3vk::DilateMaskParams params{};
                    params.width = dst_width;
                    params.height = dst_height;
                    params.mdis = d->mdis;
                    params.field = field_n;
                    params.stride = vk_d->vk_stride_pixels;
                    params.dh = d->dh ? 1 : 0;
                    params.mclip_offset =
                        (d->vi.format.colorFamily == cfYUV && plane > 0) ? 0.5F
                                                                         : 0.0F;

                    dispatch(vk_d->pipelines->getDilateMask(),
                             divUp(dst_width, 32), divUp(field_height, 16), 1,
                             &params, sizeof(params));
                }

                {
                    eedi3vk::CalcCostsParams params{};
                    params.width = dst_width;
                    params.height = dst_height;
                    params.mdis = d->mdis;
                    params.tpitch = d->tpitch;
                    params.alpha = d->alpha;
                    params.beta = d->beta;
                    params.gamma = d->gamma;
                    params.remainingWeight = d->remainingWeight;
                    params.cost3 = d->cost3 ? 1 : 0;
                    params.ucubic = d->ucubic ? 1 : 0;
                    params.has_mclip = (mclip != nullptr) ? 1 : 0;
                    params.nrad = d->nrad;
                    params.field = field_n;
                    params.stride = vk_d->vk_stride_pixels;
                    params.dh = d->dh ? 1 : 0;

                    dispatch(vk_d->pipelines->getCalcCosts(),
                             divUp(dst_width, 32), divUp(field_height, 4), 1,
                             &params, sizeof(params));
                }

                {
                    eedi3vk::ViterbiScanParams params{};
                    params.width = dst_width;
                    params.height = dst_height;
                    params.mdis = d->mdis;
                    params.tpitch = d->tpitch;
                    params.gamma = d->gamma;
                    params.has_mclip = (mclip != nullptr) ? 1 : 0;

                    dispatch(vk_d->pipelines->getViterbiScan(), 1, field_height,
                             1, &params, sizeof(params));
                }

                {
                    eedi3vk::InterpolateParams params{};
                    params.width = dst_width;
                    params.height = dst_height;
                    params.mdis = d->mdis;
                    params.ucubic = d->ucubic ? 1 : 0;
                    params.has_mclip = (mclip != nullptr) ? 1 : 0;
                    params.field = field_n;
                    params.stride = vk_d->vk_stride_pixels;
                    params.dh = d->dh ? 1 : 0;

                    dispatch(vk_d->pipelines->getInterpolate(),
                             divUp(dst_width, 16), divUp(field_height, 16), 1,
                             &params, sizeof(params));
                }
            };

            auto upload_plane = [&](vk::raii::CommandBuffer& cmd) {
                {
                    auto* staging_ptr =
                        static_cast<uint8_t*>(res.src_staging.getMappedData());
                    const uint8_t* src_ptr = vsapi->getReadPtr(src, plane);
                    int src_stride =
                        static_cast<int>(vsapi->getStride(src, plane));

                    for (int y = 0; y < plane_height; y++) {
                        std::memcpy(staging_ptr + (y * vk_d->vk_stride),
                                    src_ptr + (y * src_stride),
                                    plane_width * sizeof(float));
                    }

                    vmaFlushAllocation(vk_d->memory->getAllocator(),
                                       res.src_staging.allocation, 0,
                                       plane_size_bytes);

                    vk::BufferCopy copy_region{0, 0, plane_size_bytes};
                    cmd.copyBuffer(res.src_staging.buffer,
                                   res.src_buffer.buffer, copy_region);
                }

                if (mclip != nullptr) {
                    auto* staging_ptr = static_cast<uint8_t*>(
                        res.mclip_staging.getMappedData());
                    const uint8_t* mclip_ptr = vsapi->getReadPtr(mclip, plane);
                    int mclip_stride =
                        static_cast<int>(vsapi->getStride(mclip, plane));

                    for (int y = 0; y < plane_height; y++) {
                        std::memcpy(staging_ptr + (y * vk_d->vk_stride),
                                    mclip_ptr + (y * mclip_stride),
                                    plane_width * sizeof(float));
                    }

                    vmaFlushAllocation(vk_d->memory->getAllocator(),
                                       res.mclip_staging.allocation, 0,
                                       plane_size_bytes);

                    vk::BufferCopy copy_region{0, 0, plane_size_bytes};
                    cmd.copyBuffer(res.mclip_staging.buffer,
                                   res.mclip_buffer.buffer, copy_region);
                }
            };

            auto download_plane = [&](vk::raii::CommandBuffer& cmd) {
                if (need_dmap_readback) {
                    dmap_size_bytes = static_cast<VkDeviceSize>(
                        static_cast<size_t>(dst_width) * field_height *
                        sizeof(int32_t));
                    vk::BufferCopy dmap_copy{0, 0, dmap_size_bytes};
                    cmd.copyBuffer(res.dmap_buffer.buffer,
                                   res.dmap_staging.buffer, dmap_copy);
                }

                vk::BufferCopy dst_copy{0, 0, dst_size_bytes};
                cmd.copyBuffer(res.dst_buffer.buffer, res.dst_staging.buffer,
                               dst_copy);
            };

            if (use_dedicated_transfer) {
                res.transfer_upload_command_buffer.reset({});
                res.transfer_upload_command_buffer.begin(
                    {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                upload_plane(res.transfer_upload_command_buffer);
                res.transfer_upload_command_buffer.end();

                vk::SubmitInfo upload_submit_info{};
                upload_submit_info.commandBufferCount = 1;
                upload_submit_info.pCommandBuffers =
                    &*res.transfer_upload_command_buffer;
                upload_submit_info.signalSemaphoreCount = 1;
                upload_submit_info.pSignalSemaphores = &*res.upload_semaphore;
                vk_d->context->submitTransfer(upload_submit_info, vk::Fence{});

                res.compute_command_buffer.reset({});
                res.compute_command_buffer.begin(
                    {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                record_compute(res.compute_command_buffer);
                res.compute_command_buffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eTransfer, {}, readback_barrier,
                    nullptr, nullptr);
                res.compute_command_buffer.end();

                vk::PipelineStageFlags wait_stage =
                    vk::PipelineStageFlagBits::eComputeShader;
                vk::SubmitInfo compute_submit_info{};
                compute_submit_info.waitSemaphoreCount = 1;
                compute_submit_info.pWaitSemaphores = &*res.upload_semaphore;
                compute_submit_info.pWaitDstStageMask = &wait_stage;
                compute_submit_info.commandBufferCount = 1;
                compute_submit_info.pCommandBuffers =
                    &*res.compute_command_buffer;
                compute_submit_info.signalSemaphoreCount = 1;
                compute_submit_info.pSignalSemaphores = &*res.compute_semaphore;
                vk_d->context->submitCompute(compute_submit_info, vk::Fence{});

                res.transfer_download_command_buffer.reset({});
                res.transfer_download_command_buffer.begin(
                    {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                download_plane(res.transfer_download_command_buffer);
                res.transfer_download_command_buffer.end();

                vk::PipelineStageFlags transfer_wait_stage =
                    vk::PipelineStageFlagBits::eTransfer;
                vk::SubmitInfo download_submit_info{};
                download_submit_info.waitSemaphoreCount = 1;
                download_submit_info.pWaitSemaphores = &*res.compute_semaphore;
                download_submit_info.pWaitDstStageMask = &transfer_wait_stage;
                download_submit_info.commandBufferCount = 1;
                download_submit_info.pCommandBuffers =
                    &*res.transfer_download_command_buffer;
                vk_d->context->submitTransfer(download_submit_info, *res.fence);
            } else {
                res.compute_command_buffer.reset({});
                res.compute_command_buffer.begin(
                    {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                upload_plane(res.compute_command_buffer);

                res.compute_command_buffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eComputeShader, {},
                    transfer_to_compute_barrier, nullptr, nullptr);

                record_compute(res.compute_command_buffer);

                res.compute_command_buffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eTransfer, {}, readback_barrier,
                    nullptr, nullptr);

                download_plane(res.compute_command_buffer);

                res.compute_command_buffer.end();

                vk::SubmitInfo submit_info{};
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &*res.compute_command_buffer;
                vk_d->context->submitCompute(submit_info, *res.fence);
            }

            auto result = vk_d->context->getDevice().waitForFences(
                *res.fence, VK_TRUE, UINT64_MAX);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed waiting for GPU fence");
            }
            vk_d->context->getDevice().resetFences(*res.fence);

            vmaInvalidateAllocation(vk_d->memory->getAllocator(),
                                    res.dst_staging.allocation, 0,
                                    dst_size_bytes);
            if (need_dmap_readback) {
                vmaInvalidateAllocation(vk_d->memory->getAllocator(),
                                        res.dmap_staging.allocation, 0,
                                        dmap_size_bytes);
            }

            const auto* staging_ptr =
                static_cast<const uint8_t*>(res.dst_staging.getMappedData());
            uint8_t* dst_ptr = vsapi->getWritePtr(dst, plane);
            auto dst_stride = vsapi->getStride(dst, plane);

            for (int y = 0; y < dst_height; y++) {
                std::memcpy(dst_ptr + (y * dst_stride),
                            staging_ptr + (y * vk_d->vk_stride),
                            dst_width * sizeof(float));
            }

            if (d->vcheck > 0) {
                VSFrame* pad = vsapi->newVideoFrame(
                    &d->padFormat, dst_width + (MARGIN_H * 2),
                    dst_height + (MARGIN_V * 2), nullptr, core);

                copyPad(src, pad, plane, d->dh, 1 - field_n, vsapi);

                const auto* src_ptr_base =
                    reinterpret_cast<const float*>(vsapi->getReadPtr(pad, 0));
                const int pad_stride_pixels =
                    static_cast<int>(vsapi->getStride(pad, 0) / sizeof(float));

                const int padded_height = vsapi->getFrameHeight(pad, 0);

                const float* aligned_src_ptr =
                    src_ptr_base + (pad_stride_pixels * (MARGIN_V + field_n));

                auto* aligned_dst_ptr =
                    reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane)) +
                    (static_cast<ptrdiff_t>(dst_stride / sizeof(float)) *
                     field_n);

                const int* dmap_ptr =
                    static_cast<const int*>(res.dmap_staging.getMappedData());

                const VSFrame* scp = nullptr;
                if (d->sclip != nullptr) {
                    scp = vsapi->getFrameFilter(n, d->sclip, frameCtx);
                }

                const float* aligned_scpp = nullptr;
                if (scp != nullptr) {
                    aligned_scpp =
                        reinterpret_cast<const float*>(
                            vsapi->getReadPtr(scp, plane)) +
                        (static_cast<ptrdiff_t>(dst_stride / sizeof(float)) *
                         field_n);
                }

                std::vector<float> tline(dst_width);

                vCheck(aligned_src_ptr, aligned_scpp, aligned_dst_ptr, dmap_ptr,
                       tline.data(), field_n, dst_width, padded_height,
                       pad_stride_pixels,
                       static_cast<ptrdiff_t>(dst_stride / sizeof(float)), d);

                vsapi->freeFrame(pad);
                if (scp != nullptr) {
                    vsapi->freeFrame(scp);
                }
            }
        }

        VSMap* props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapSetInt(props, "_FieldBased", 0, maReplace);

        if (d->field > 1) {
            int errNum = 0;
            int errDen = 0;
            int64_t durationNum =
                vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
            int64_t durationDen =
                vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
            if ((errNum == 0) && (errDen == 0)) {
                vsh::muldivRational(&durationNum, &durationDen, 1, 2);
                vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
                vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
            }
        }

        vsapi->freeFrame(src);
        if (mclip != nullptr) {
            vsapi->freeFrame(mclip);
        }

        return dst;
    }

    return nullptr;
}

void VS_CC eedi3Free(void* instanceData, [[maybe_unused]] VSCore* core,
                     const VSAPI* vsapi) {
    auto vk_d =
        std::unique_ptr<EEDI3VKData>(static_cast<EEDI3VKData*>(instanceData));
    auto* d = &vk_d->d;

    vk_d->context->waitIdle();

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sclip);
    vsapi->freeNode(d->mclip);
}

void VS_CC eedi3Create(const VSMap* in, VSMap* out,
                       [[maybe_unused]] void* userData, VSCore* core,
                       const VSAPI* vsapi) {
    auto vk_d = std::make_unique<EEDI3VKData>();
    auto* d = &vk_d->d;
    int err = 0;

    try {
        vk_d->device_id = vsapi->mapGetIntSaturated(in, "device_id", 0, &err);
        if (err != 0) {
            vk_d->device_id = -1;
        }

        vk_d->context = std::shared_ptr<eedi3vk::VulkanContext>(
            &eedi3vk::VulkanContext::getInstance(vk_d->device_id),
            [](eedi3vk::VulkanContext*) {});
        vk_d->memory = std::make_unique<eedi3vk::VulkanMemory>(*vk_d->context);

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->sclip = vsapi->mapGetNode(in, "sclip", 0, &err);
        d->mclip = vsapi->mapGetNode(in, "mclip", 0, &err);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            d->vi.format.sampleType != stFloat ||
            d->vi.format.bitsPerSample != 32) {
            throw "only constant format 32 bit float input supported"s;
        }

        vsapi->queryVideoFormat(&d->padFormat, cfGray, d->vi.format.sampleType,
                                d->vi.format.bitsPerSample, 0, 0, core);

        d->field = vsapi->mapGetIntSaturated(in, "field", 0, nullptr);
        d->dh = (vsapi->mapGetInt(in, "dh", 0, &err) != 0);

        const int m = vsapi->mapNumElements(in, "planes");
        std::ranges::fill(d->process, (m <= 0));

        for (int i = 0; i < m; i++) {
            const int n = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);
            if (n < 0 || n >= d->vi.format.numPlanes) {
                throw "plane index out of range"s;
            }
            if (d->process.at(n)) {
                throw "plane specified twice"s;
            }
            d->process.at(n) = true;
        }

        d->alpha = vsapi->mapGetFloatSaturated(in, "alpha", 0, &err);
        if (err != 0) {
            d->alpha = 0.2F;
        }

        d->beta = vsapi->mapGetFloatSaturated(in, "beta", 0, &err);
        if (err != 0) {
            d->beta = 0.25F;
        }

        d->gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
        if (err != 0) {
            d->gamma = 20.0F;
        }

        d->nrad = vsapi->mapGetIntSaturated(in, "nrad", 0, &err);
        if (err != 0) {
            d->nrad = 2;
        }

        d->mdis = vsapi->mapGetIntSaturated(in, "mdis", 0, &err);
        if (err != 0) {
            d->mdis = 20;
        }

        d->ucubic = (vsapi->mapGetInt(in, "ucubic", 0, &err) != 0);
        if (err != 0) {
            d->ucubic = true;
        }

        d->cost3 = (vsapi->mapGetInt(in, "cost3", 0, &err) != 0);
        if (err != 0) {
            d->cost3 = true;
        }

        d->vcheck = vsapi->mapGetIntSaturated(in, "vcheck", 0, &err);
        if (err != 0) {
            d->vcheck = 2;
        }

        float vthresh0 = vsapi->mapGetFloatSaturated(in, "vthresh0", 0, &err);
        if (err != 0) {
            vthresh0 = 32.0F;
        }

        float vthresh1 = vsapi->mapGetFloatSaturated(in, "vthresh1", 0, &err);
        if (err != 0) {
            vthresh1 = 64.0F;
        }

        d->vthresh2 = vsapi->mapGetFloatSaturated(in, "vthresh2", 0, &err);
        if (err != 0) {
            d->vthresh2 = 4.0F;
        }

        if (d->field < 0 || d->field > 3) {
            throw "field must be 0, 1, 2, or 3"s;
        }
        if (!d->dh) {
            const auto* frame = vsapi->getFrame(0, d->node, nullptr, 0);

            for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
                if (d->process.at(plane) &&
                    ((vsapi->getFrameHeight(frame, plane) & 1) != 0)) {
                    vsapi->freeFrame(frame);
                    throw "plane's height must be mod 2 when dh=False"s;
                }
            }

            vsapi->freeFrame(frame);
        }
        if (d->dh && d->field > 1) {
            throw "field must be 0 or 1 when dh=True"s;
        }
        if (d->alpha < 0.0F || d->alpha > 1.0F) {
            throw "alpha must be between 0.0 and 1.0"s;
        }
        if (d->beta < 0.0F || d->beta > 1.0F) {
            throw "beta must be between 0.0 and 1.0"s;
        }
        if (d->alpha + d->beta > 1.0F) {
            throw "alpha+beta must be between 0.0 and 1.0"s;
        }
        if (d->gamma < 0.0F) {
            throw "gamma must be >= 0.0"s;
        }
        if (d->nrad < 0 || d->nrad > 3) {
            throw "nrad must be between 0 and 3"s;
        }
        if (d->mdis < 1 || d->mdis > 40) {
            throw "mdis must be between 1 and 40"s;
        }
        if (d->vcheck < 0 || d->vcheck > 3) {
            throw "vcheck must be 0, 1, 2, or 3"s;
        }

        if (d->vcheck > 0 &&
            (vthresh0 <= 0.0F || vthresh1 <= 0.0F || d->vthresh2 <= 0.0F)) {
            throw "vthresh0, vthresh1 and vthresh2 must be greater than 0.0"s;
        }

        if (d->mclip != nullptr) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->mclip), &d->vi)) {
                throw "mclip's format and dimensions don't match"s;
            }

            if (vsapi->getVideoInfo(d->mclip)->numFrames != d->vi.numFrames) {
                throw "mclip's number of frames doesn't match"s;
            }
        }

        if (d->vcheck > 0 && (d->sclip != nullptr)) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->sclip), &d->vi)) {
                throw "sclip's format and dimensions don't match"s;
            }

            if (vsapi->getVideoInfo(d->sclip)->numFrames != d->vi.numFrames) {
                throw "sclip's number of frames doesn't match"s;
            }
        }

        d->remainingWeight = 1.0F - d->alpha - d->beta;
        if (d->cost3) {
            d->alpha /= 3.0F;
        }
        d->beta /= 255.0F;
        d->gamma /= 255.0F;
        vthresh0 /= 255.0F;
        vthresh1 /= 255.0F;

        d->tpitch = (d->mdis * 2) + 1;
        vk_d->pipelines = std::make_unique<eedi3vk::EEDI3Pipelines>(
            *vk_d->context, d->tpitch);

        d->rcpVthresh0 = 1.0F / vthresh0;
        d->rcpVthresh1 = 1.0F / vthresh1;
        d->rcpVthresh2 = 1.0F / d->vthresh2;

        if (d->field > 1) {
            d->vi.numFrames *= 2;
            vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);
        }
        if (d->dh) {
            d->vi.height *= 2;
        }

        vk_d->vk_stride = ((size_t)d->vi.width * sizeof(float) + 63) & ~63;
        vk_d->vk_stride_pixels =
            static_cast<int>(vk_d->vk_stride / sizeof(float));

        int num_streams = vsapi->mapGetIntSaturated(in, "num_streams", 0, &err);
        if (err != 0) {
            num_streams = 1;
        }

        vk_d->resource_pool = std::make_unique<eedi3vk::VulkanResourcePool>(
            *vk_d->context, *vk_d->memory, num_streams, vk_d->vk_stride,
            d->vi.width, d->vi.height, d->tpitch, d->mclip != nullptr);

    } catch (const std::string& error) {
        vsapi->mapSetError(out, ("EEDI3VK: " + error).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->sclip);
        vsapi->freeNode(d->mclip);
        return;
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, ("EEDI3VK: " + std::string(e.what())).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->sclip);
        vsapi->freeNode(d->mclip);
        return;
    }

    std::vector<VSFilterDependency> deps = {
        {d->node, d->field > 1 ? rpGeneral : rpStrictSpatial}};
    if (d->sclip != nullptr) {
        deps.push_back({d->sclip, rpStrictSpatial});
    }
    if (d->mclip != nullptr) {
        deps.push_back({d->mclip, d->field > 1 ? rpGeneral : rpStrictSpatial});
    }

    vsapi->createVideoFilter(
        out, "EEDI3VK", &d->vi, eedi3GetFrame, eedi3Free, fmParallel,
        deps.data(), static_cast<int>(deps.size()), vk_d.release(), core);
}

} // anonymous namespace

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.yuygfgg.eedi3vk", "eedi3vk", "EEDI3 Vulkan Port",
                         VS_MAKE_VERSION(1, 3), VAPOURSYNTH_API_VERSION, 0,
                         plugin);
    vspapi->registerFunction("EEDI3",
                             "clip:vnode;"
                             "field:int;"
                             "dh:int:opt;"
                             "planes:int[]:opt;"
                             "alpha:float:opt;"
                             "beta:float:opt;"
                             "gamma:float:opt;"
                             "nrad:int:opt;"
                             "mdis:int:opt;"
                             "hp:int:opt;"
                             "ucubic:int:opt;"
                             "cost3:int:opt;"
                             "vcheck:int:opt;"
                             "vthresh0:float:opt;"
                             "vthresh1:float:opt;"
                             "vthresh2:float:opt;"
                             "sclip:vnode:opt;"
                             "mclip:vnode:opt;"
                             "device_id:int:opt;"
                             "num_streams:int:opt;",
                             "clip:vnode;", eedi3Create, nullptr, plugin);
}
