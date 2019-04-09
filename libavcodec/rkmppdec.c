/*
 * RockChip MPP Video Decoder
 * Copyright (c) 2017 Lionel CHAZALLON
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config.h"
#if HAVE_NEON
#include <arm_neon.h>
#endif
#include <drm_fourcc.h>
#include <pthread.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_mpi.h>
#include <time.h>
#include <unistd.h>

#include "avcodec.h"
#include "decode.h"
#include "hwaccel.h"
#include "internal.h"
#include "libavutil/buffer.h"
#include "libavutil/common.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_drm.h"
#include "libavutil/imgutils.h"
#include "libavutil/log.h"

#define RECEIVE_FRAME_TIMEOUT   5
#define FRAMEGROUP_MAX_FRAMES   16
#define INPUT_MAX_PACKETS       2

typedef struct {
    MppCtx ctx;
    MppApi *mpi;
    MppBufferGroup frame_group;

    char first_packet;
    char eos_reached;

    AVBufferRef *frames_ref;
    AVBufferRef *device_ref;
} RKMPPDecoder;

typedef struct {
    AVClass *av_class;
    AVBufferRef *decoder_ref;
} RKMPPDecodeContext;

typedef struct {
    MppFrame frame;
    AVBufferRef *decoder_ref;
} RKMPPFrameContext;

static MppCodingType rkmpp_get_codingtype(AVCodecContext *avctx)
{
    switch (avctx->codec_id) {
    case AV_CODEC_ID_H264:          return MPP_VIDEO_CodingAVC;
    case AV_CODEC_ID_HEVC:          return MPP_VIDEO_CodingHEVC;
    case AV_CODEC_ID_VP8:           return MPP_VIDEO_CodingVP8;
    case AV_CODEC_ID_VP9:           return MPP_VIDEO_CodingVP9;
    default:                        return MPP_VIDEO_CodingUnused;
    }
}

static uint32_t rkmpp_get_frameformat(MppFrameFormat mppformat)
{
    switch (mppformat) {
    case MPP_FMT_YUV420SP:          return DRM_FORMAT_NV12;
#ifdef DRM_FORMAT_NV12_10
    case MPP_FMT_YUV420SP_10BIT:    return DRM_FORMAT_NV12_10;
#endif
    default:                        return 0;
    }
}

static int rkmpp_write_data(AVCodecContext *avctx, uint8_t *buffer, int size, int64_t pts)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    int ret;
    MppPacket packet;

    // create the MPP packet
    ret = mpp_packet_init(&packet, buffer, size);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to init MPP packet (code = %d)\n", ret);
        return AVERROR_UNKNOWN;
    }

    mpp_packet_set_pts(packet, pts);

    if (!buffer)
        mpp_packet_set_eos(packet);

    ret = decoder->mpi->decode_put_packet(decoder->ctx, packet);
    if (ret != MPP_OK) {
        if (ret == MPP_ERR_BUFFER_FULL) {
            av_log(avctx, AV_LOG_DEBUG, "Buffer full writing %d bytes to decoder\n", size);
            ret = AVERROR(EAGAIN);
        } else
            ret = AVERROR_UNKNOWN;
    }
    else
        av_log(avctx, AV_LOG_DEBUG, "Wrote %d bytes to decoder\n", size);

    mpp_packet_deinit(&packet);

    return ret;
}

static int rkmpp_close_decoder(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    av_buffer_unref(&rk_context->decoder_ref);
    return 0;
}

static void rkmpp_release_decoder(void *opaque, uint8_t *data)
{
    RKMPPDecoder *decoder = (RKMPPDecoder *)data;

    if (decoder->mpi) {
        decoder->mpi->reset(decoder->ctx);
        mpp_destroy(decoder->ctx);
        decoder->ctx = NULL;
    }

    if (decoder->frame_group) {
        mpp_buffer_group_put(decoder->frame_group);
        decoder->frame_group = NULL;
    }

    av_buffer_unref(&decoder->frames_ref);
    av_buffer_unref(&decoder->device_ref);

    av_free(decoder);
}

static int rkmpp_init_decoder(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = NULL;
    MppCodingType codectype = MPP_VIDEO_CodingUnused;
    int ret;
    RK_S64 paramS64;
    RK_S32 paramS32;

    if (avctx->pix_fmt != AV_PIX_FMT_YUV420P && avctx->pix_fmt != AV_PIX_FMT_NV12)
        avctx->pix_fmt = AV_PIX_FMT_DRM_PRIME;

    // create a decoder and a ref to it
    decoder = av_mallocz(sizeof(RKMPPDecoder));
    if (!decoder) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    rk_context->decoder_ref = av_buffer_create((uint8_t *)decoder, sizeof(*decoder), rkmpp_release_decoder,
                                               NULL, AV_BUFFER_FLAG_READONLY);
    if (!rk_context->decoder_ref) {
        av_free(decoder);
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    av_log(avctx, AV_LOG_DEBUG, "Initializing RKMPP decoder.\n");

    codectype = rkmpp_get_codingtype(avctx);
    if (codectype == MPP_VIDEO_CodingUnused) {
        av_log(avctx, AV_LOG_ERROR, "Unknown codec type (%d).\n", avctx->codec_id);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = mpp_check_support_format(MPP_CTX_DEC, codectype);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Codec type (%d) unsupported by MPP\n", avctx->codec_id);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    // Create the MPP context
    ret = mpp_create(&decoder->ctx, &decoder->mpi);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create MPP context (code = %d).\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    // initialize mpp
    ret = mpp_init(decoder->ctx, MPP_CTX_DEC, codectype);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to initialize MPP context (code = %d).\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    // make decode calls blocking with a timeout
    paramS32 = MPP_POLL_BLOCK;
    ret = decoder->mpi->control(decoder->ctx, MPP_SET_OUTPUT_BLOCK, &paramS32);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set blocking mode on MPI (code = %d).\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    paramS64 = RECEIVE_FRAME_TIMEOUT;
    ret = decoder->mpi->control(decoder->ctx, MPP_SET_OUTPUT_BLOCK_TIMEOUT, &paramS64);
    if (ret != MPP_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set block timeout on MPI (code = %d).\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = mpp_buffer_group_get_internal(&decoder->frame_group, MPP_BUFFER_TYPE_ION);
    if (ret) {
       av_log(avctx, AV_LOG_ERROR, "Failed to retrieve buffer group (code = %d)\n", ret);
       ret = AVERROR_UNKNOWN;
       goto fail;
    }

    ret = decoder->mpi->control(decoder->ctx, MPP_DEC_SET_EXT_BUF_GROUP, decoder->frame_group);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to assign buffer group (code = %d)\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = mpp_buffer_group_limit_config(decoder->frame_group, 0, FRAMEGROUP_MAX_FRAMES);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set buffer group limit (code = %d)\n", ret);
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    decoder->first_packet = 1;

    av_log(avctx, AV_LOG_DEBUG, "RKMPP decoder initialized successfully.\n");

    decoder->device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_DRM);
    if (!decoder->device_ref) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ret = av_hwdevice_ctx_init(decoder->device_ref);
    if (ret < 0)
        goto fail;

    return 0;

fail:
    av_log(avctx, AV_LOG_ERROR, "Failed to initialize RKMPP decoder.\n");
    rkmpp_close_decoder(avctx);
    return ret;
}

static int rkmpp_send_packet(AVCodecContext *avctx, const AVPacket *avpkt)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    int ret;

    // handle EOF
    if (!avpkt->size) {
        av_log(avctx, AV_LOG_DEBUG, "End of stream.\n");
        decoder->eos_reached = 1;
        ret = rkmpp_write_data(avctx, NULL, 0, 0);
        if (ret)
            av_log(avctx, AV_LOG_ERROR, "Failed to send EOS to decoder (code = %d)\n", ret);
        return ret;
    }

    // on first packet, send extradata
    if (decoder->first_packet) {
        if (avctx->extradata_size) {
            ret = rkmpp_write_data(avctx, avctx->extradata,
                                            avctx->extradata_size,
                                            avpkt->pts);
            if (ret) {
                av_log(avctx, AV_LOG_ERROR, "Failed to write extradata to decoder (code = %d)\n", ret);
                return ret;
            }
        }
        decoder->first_packet = 0;
    }

    // now send packet
    ret = rkmpp_write_data(avctx, avpkt->data, avpkt->size, avpkt->pts);
    if (ret && ret!=AVERROR(EAGAIN))
        av_log(avctx, AV_LOG_ERROR, "Failed to write data to decoder (code = %d)\n", ret);

    return ret;
}

static void rkmpp_release_frame(void *opaque, uint8_t *data)
{
    AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor *)data;
    AVBufferRef *framecontextref = (AVBufferRef *)opaque;
    RKMPPFrameContext *framecontext = (RKMPPFrameContext *)framecontextref->data;

    mpp_frame_deinit(&framecontext->frame);
    av_buffer_unref(&framecontext->decoder_ref);
    av_buffer_unref(&framecontextref);

    av_free(desc);
}


static int rkmpp_mem_buffer_to_frame(AVCodecContext *avctx, MppFrame mppframe,
                                     MppBuffer buffer, AVFrame *frame)
{
    unsigned h_stride = mpp_frame_get_hor_stride(mppframe);
    unsigned v_stride = mpp_frame_get_ver_stride(mppframe);
    unsigned linesize, i, width = frame->width;
    uint8_t *buf_ptr = mpp_buffer_get_ptr(buffer);
    int ret;

    if (!buf_ptr)
        return AVERROR_UNKNOWN; // no buffer pointer

    if (mpp_frame_get_fmt(mppframe) != MPP_FMT_YUV420SP)
        return AVERROR_UNKNOWN; // unexpected format

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
        return ret;

    //frame->key_frame = 1;
    //frame->pict_type = AV_PICTURE_TYPE_I;

    av_log(avctx, AV_LOG_DEBUG, "Frame copy: %d/%d, h_stride=%d, v_stride=%d\n",
        frame->linesize[0], frame->linesize[1], h_stride, v_stride);

    linesize = frame->linesize[0];
    if (h_stride == linesize) {
        memcpy(frame->data[0], buf_ptr, linesize * frame->height);
    } else {
        uint8_t *src = buf_ptr;
        uint8_t *end = src + h_stride * frame->height;
        uint8_t *y = frame->data[0];

        for (; src < end; src += h_stride) {
            memcpy(y, src, width);
            y += linesize;
        }
    }

    // NV12    - YYYY... UV UV...
    // YUV420P - YYYY... UU... VV...
    buf_ptr += h_stride * v_stride;
    if (frame->format == AV_PIX_FMT_YUV420P) {
        for (i = 0; i < frame->height / 2; ++i) {
            uint32_t *c = (void*) &buf_ptr[i * h_stride];
            uint32_t *e = c + (width - 1) / 4;
            uint16_t *u = (void*) &frame->data[1][i * frame->linesize[1]];
            uint16_t *v = (void*) &frame->data[2][i * frame->linesize[2]];

#if HAVE_NEON && 0
            int vectors = width / 32;

            while (--vectors >= 0) {
                const uint8x16_t c0 = vld1q_u8((uint8_t*) c);
                const uint8x16_t c1 = vld1q_u8((uint8_t*) c + 16);
                const uint8x16x2_t dst = vuzpq_u8(c0, c1);

                vst1q_u8((uint8_t*) u, dst.val[0]);
                vst1q_u8((uint8_t*) v, dst.val[1]);
                c += 8;
                u += 8;
                v += 8;
            }
#endif
            while (c <= e) {
                uint32_t uv = *(c++);
                *(u++) = (uv & 0xff) | ((uv >> 8) & 0xff00);
                *(v++) = ((uv >> 8) & 0xff) | ((uv >> 16) & 0xff00);
            }
        }
    } else if (h_stride == frame->linesize[1]) {
        memcpy(frame->data[1], buf_ptr, h_stride * frame->height / 2);
    } else {
        linesize = frame->linesize[1];
        for (i = 0; i < frame->height / 2; ++i)
            memcpy(frame->data[1] + i * linesize, buf_ptr + i * h_stride, width);
    }
    mpp_frame_deinit(&mppframe);
    return 0;
}

static int rkmpp_drm_buffer_to_frame(RKMPPDecodeContext *rk_context, MppFrame mppframe,
                                     MppBuffer buffer, AVFrame *frame)
{
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    RKMPPFrameContext *framecontext = NULL;
    AVBufferRef *framecontextref = NULL;
    AVDRMFrameDescriptor *desc = NULL;
    AVDRMLayerDescriptor *layer = NULL;

    desc = av_mallocz(sizeof(AVDRMFrameDescriptor));
    if (!desc)
        goto fail;

    desc->nb_objects = 1;
    desc->objects[0].fd = mpp_buffer_get_fd(buffer);
    desc->objects[0].size = mpp_buffer_get_size(buffer);

    desc->nb_layers = 1;
    layer = &desc->layers[0];
    layer->format = rkmpp_get_frameformat(mpp_frame_get_fmt(mppframe));

    layer->nb_planes = 2;

    layer->planes[0].object_index = 0;
    layer->planes[0].offset = 0;
    layer->planes[0].pitch = mpp_frame_get_hor_stride(mppframe);

    layer->planes[1].object_index = 0;
    layer->planes[1].offset = layer->planes[0].pitch * mpp_frame_get_ver_stride(mppframe);
    layer->planes[1].pitch = layer->planes[0].pitch;

    // we also allocate a struct in buf[0] that will allow to hold additionnal information
    // for releasing properly MPP frames and decoder
    framecontextref = av_buffer_allocz(sizeof(*framecontext));
    if (!framecontextref)
        goto fail;

    // MPP decoder needs to be closed only when all frames have been released.
    framecontext = (RKMPPFrameContext *)framecontextref->data;
    framecontext->decoder_ref = av_buffer_ref(rk_context->decoder_ref);
    framecontext->frame = mppframe;

    frame->data[0]  = (uint8_t *)desc;
    frame->buf[0]   = av_buffer_create((uint8_t *)desc, sizeof(*desc), rkmpp_release_frame,
                                       framecontextref, AV_BUFFER_FLAG_READONLY);

    if (!frame->buf[0])
        goto fail;

    frame->hw_frames_ctx = av_buffer_ref(decoder->frames_ref);
    if (!frame->hw_frames_ctx)
        goto fail;

    return 0;

fail:
    if (framecontext)
        av_buffer_unref(&framecontext->decoder_ref);

    if (framecontextref)
        av_buffer_unref(&framecontextref);

    if (desc)
        av_free(desc);

    return AVERROR(ENOMEM);
}

static int rkmpp_configure_decoder(AVCodecContext *avctx, MppFrame mppframe, AVFrame *frame)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    AVHWFramesContext *hwframes;
    int ret;
    MppFrameFormat mppformat;
    uint32_t drmformat;

    av_log(avctx, AV_LOG_INFO, "Decoder noticed an info change (%dx%d), format=%d\n",
                                (int)mpp_frame_get_width(mppframe), (int)mpp_frame_get_height(mppframe),
                                (int)mpp_frame_get_fmt(mppframe));

    avctx->width = mpp_frame_get_width(mppframe);
    avctx->height = mpp_frame_get_height(mppframe);

    ret = mpp_buffer_group_limit_config(decoder->frame_group, 0, FRAMEGROUP_MAX_FRAMES);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to set buffer group limit (code = %d)\n", ret);
        return AVERROR_UNKNOWN;
    }

    decoder->mpi->control(decoder->ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);

    av_buffer_unref(&decoder->frames_ref);

    decoder->frames_ref = av_hwframe_ctx_alloc(decoder->device_ref);
    if (!decoder->frames_ref) {
        av_log(avctx, AV_LOG_DEBUG, "No frames ref!");
        return AVERROR(ENOMEM);
    }

    mppformat = mpp_frame_get_fmt(mppframe);
    drmformat = rkmpp_get_frameformat(mppformat);

    hwframes = (AVHWFramesContext*)decoder->frames_ref->data;
    hwframes->format    = AV_PIX_FMT_DRM_PRIME;
    hwframes->sw_format = drmformat == DRM_FORMAT_NV12 ? AV_PIX_FMT_NV12 : AV_PIX_FMT_NONE;
    hwframes->width     = avctx->width;
    hwframes->height    = avctx->height;
    ret = av_hwframe_ctx_init(decoder->frames_ref);
    if (ret < 0) {
        av_log(avctx, AV_LOG_DEBUG, "Init fail!");
        return ret;
    }

    // here decoder is fully initialized, we need to feed it again with data
#if 0
    frame->flags = AV_FRAME_FLAG_CORRUPT;
    mpp_frame_deinit(&mppframe);
    return 0;
#else
    //frame->flags = AV_FRAME_FLAG_CORRUPT;
    av_log(avctx, AV_LOG_DEBUG, "AGAIN!");
    //mpp_frame_deinit(&mppframe);
    return AVERROR(EAGAIN);
#endif
}

static int rkmpp_decode_mppframe(AVCodecContext *avctx, MppFrame mppframe, AVFrame *frame)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    int ret;
    MppBuffer buffer = NULL;
    int mode;

    // here we should have a valid frame
    av_log(avctx, AV_LOG_DEBUG, "Received a frame.\n");

    // setup general frame fields
    frame->width            = mpp_frame_get_width(mppframe);
    frame->height           = mpp_frame_get_height(mppframe);
    frame->pts              = mpp_frame_get_pts(mppframe);
    frame->color_range      = mpp_frame_get_color_range(mppframe);
    frame->color_primaries  = mpp_frame_get_color_primaries(mppframe);
    frame->color_trc        = mpp_frame_get_color_trc(mppframe);
    frame->colorspace       = mpp_frame_get_colorspace(mppframe);

    mode = mpp_frame_get_mode(mppframe);
    frame->interlaced_frame = ((mode & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_DEINTERLACED);
    frame->top_field_first  = ((mode & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_TOP_FIRST);
    av_log(avctx, AV_LOG_DEBUG, "interlaced_frame=%d, top_field_first=%d.\n",
        frame->interlaced_frame, frame->top_field_first);

    // now setup the frame buffer info
    buffer = mpp_frame_get_buffer(mppframe);
    if (buffer) {
        if (avctx->pix_fmt == AV_PIX_FMT_YUV420P || avctx->pix_fmt == AV_PIX_FMT_NV12) {
            av_log(avctx, AV_LOG_DEBUG, "Convert frame.\n");
            frame->format = avctx->pix_fmt;
            ret = rkmpp_mem_buffer_to_frame(avctx, mppframe, buffer, frame);
        } else {
            av_log(avctx, AV_LOG_DEBUG, "DRM frame.\n");
            frame->format = AV_PIX_FMT_DRM_PRIME;
            ret = rkmpp_drm_buffer_to_frame(rk_context, mppframe, buffer, frame);
        }
    } else {
        av_log(avctx, AV_LOG_ERROR, "Failed to retrieve the frame buffer, frame is dropped (code = %d)\n", ret);
        //mpp_frame_deinit(&mppframe);
        //frame->flags = AV_FRAME_FLAG_CORRUPT;
        return AVERROR(EAGAIN);
    }

    return ret;
}

static int rkmpp_retrieve_frame(AVCodecContext *avctx, AVFrame *frame)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    int ret;
    MppFrame mppframe = NULL;

    ret = decoder->mpi->decode_get_frame(decoder->ctx, &mppframe);
    if (ret != MPP_OK && ret != MPP_ERR_TIMEOUT) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get a frame from MPP (code = %d)\n", ret);
        goto fail;
    }

    if (mppframe) {
        // Check whether we have a special frame or not
        if (mpp_frame_get_info_change(mppframe)) {
            ret = rkmpp_configure_decoder(avctx, mppframe, frame);
        } else if (mpp_frame_get_eos(mppframe)) {
            av_log(avctx, AV_LOG_DEBUG, "Received a EOS frame.\n");
            decoder->eos_reached = 1;
            ret = AVERROR_EOF;
        } else if (mpp_frame_get_discard(mppframe)) {
            av_log(avctx, AV_LOG_DEBUG, "Received a discard frame.\n");
            frame->flags = AV_FRAME_FLAG_DISCARD;
        } else if (mpp_frame_get_errinfo(mppframe)) {
            av_log(avctx, AV_LOG_ERROR, "Received a errinfo frame.\n");
            ret = AVERROR_UNKNOWN;
        } else {
            ret = rkmpp_decode_mppframe(avctx, mppframe, frame);
        }

        if (ret) {
            goto fail;
        }
        return ret;
    } else if (decoder->eos_reached) {
        return AVERROR_EOF;
    } else if (ret == MPP_ERR_TIMEOUT) {
        av_log(avctx, AV_LOG_DEBUG, "Timeout when trying to get a frame from MPP\n");
    }

    return AVERROR(EAGAIN);

fail:
    if (mppframe)
        mpp_frame_deinit(&mppframe);

    return ret;
}

static int rkmpp_decoder_put(AVCodecContext *avctx, AVFrame *frame)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    int ret = MPP_NOK;
    AVPacket pkt = {0};
    RK_S32 usedslots, freeslots;

    if (!decoder->eos_reached) {
        // we get the available slots in decoder
        ret = decoder->mpi->control(decoder->ctx, MPP_DEC_GET_STREAM_COUNT, &usedslots);
        if (ret != MPP_OK) {
            av_log(avctx, AV_LOG_ERROR, "Failed to get decoder used slots (code = %d).\n", ret);
            return ret;
        }

        freeslots = INPUT_MAX_PACKETS - usedslots;
        if (freeslots > 0) {
            ret = ff_decode_get_packet(avctx, &pkt);
            if (ret == AVERROR(EAGAIN)) {
               // return 0;
            }
            if (ret < 0 && ret != AVERROR_EOF) {
                av_log(avctx, AV_LOG_DEBUG, "ff_decode_get_packet = %d\n", ret);
                return ret;
            }

            ret = rkmpp_send_packet(avctx, &pkt);
            av_packet_unref(&pkt);

            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "Failed to send packet to decoder (code = %d)\n", ret);
                return ret;
            }

            return 0;
        }
        
        av_log(avctx, AV_LOG_DEBUG, "Free slots: %d\n", freeslots);
    }

    return AVERROR(EAGAIN);
}

static int rkmpp_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    int ret, ret2;

    ret = rkmpp_decoder_put(avctx, frame);
    av_log(avctx, AV_LOG_DEBUG, "rkmpp_decoder_put = %d\n", ret2);
    if (ret == 0)
    {
        ret = rkmpp_retrieve_frame(avctx, frame);
        av_log(avctx, AV_LOG_DEBUG, "rkmpp_retrieve_frame = %d\n", ret);
    }
    return ret;
}

static int rkmpp_decode_frame(AVCodecContext *avctx, void *data,
                             int *got_frame, AVPacket *avpkt)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    int ret = MPP_NOK;
    AVFrame *frame = data;

    if (!decoder->eos_reached) {
        ret = rkmpp_send_packet(avctx, avpkt);
        av_packet_unref(avpkt);

        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to send packet to decoder (code = %d)\n", ret);
            return ret;
        }

        return AVERROR(EAGAIN);
    }

    ret = rkmpp_retrieve_frame(avctx, frame);
    av_log(avctx, AV_LOG_DEBUG, "rkmpp_retrieve_frame = %d\n", ret);
    if (ret == 0) {
        *got_frame = 1;
        return 0;
    }

    return ret;
}

static void rkmpp_flush(AVCodecContext *avctx)
{
    RKMPPDecodeContext *rk_context = avctx->priv_data;
    RKMPPDecoder *decoder = (RKMPPDecoder *)rk_context->decoder_ref->data;
    int ret = MPP_NOK;

    av_log(avctx, AV_LOG_DEBUG, "Flush.\n");

    ret = decoder->mpi->reset(decoder->ctx);
    if (ret == MPP_OK) {
        decoder->first_packet = 1;
    } else
        av_log(avctx, AV_LOG_ERROR, "Failed to reset MPI (code = %d)\n", ret);
}

static const AVCodecHWConfigInternal *rkmpp_hw_configs[] = {
    HW_CONFIG_INTERNAL(DRM_PRIME),
    NULL
};

#define RKMPP_DEC_CLASS(NAME) \
    static const AVClass rkmpp_##NAME##_dec_class = { \
        .class_name = "rkmpp_" #NAME "_dec", \
        .version    = LIBAVUTIL_VERSION_INT, \
    };

#define RKMPP_DEC(NAME, ID, BSFS) \
    RKMPP_DEC_CLASS(NAME) \
    AVCodec ff_##NAME##_rkmpp_decoder = { \
        .name           = #NAME "_rkmpp", \
        .long_name      = NULL_IF_CONFIG_SMALL(#NAME " (rkmpp)"), \
        .type           = AVMEDIA_TYPE_VIDEO, \
        .id             = ID, \
        .priv_data_size = sizeof(RKMPPDecodeContext), \
        .init           = rkmpp_init_decoder, \
        .close          = rkmpp_close_decoder, \
        .receive_frame  = rkmpp_receive_frame, \
        .flush          = rkmpp_flush, \
        .priv_class     = &rkmpp_##NAME##_dec_class, \
        .capabilities   = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_HARDWARE, \
        .pix_fmts       = (const enum AVPixelFormat[]) { AV_PIX_FMT_DRM_PRIME, \
                                                         AV_PIX_FMT_NV12, \
                                                         /*AV_PIX_FMT_YUV420P, */ \
                                                         AV_PIX_FMT_NONE}, \
        .hw_configs     = rkmpp_hw_configs, \
        .bsfs           = BSFS, \
        .wrapper_name   = "rkmpp", \
    };

RKMPP_DEC(h264,  AV_CODEC_ID_H264,          "h264_mp4toannexb")
RKMPP_DEC(hevc,  AV_CODEC_ID_HEVC,          "hevc_mp4toannexb")
RKMPP_DEC(vp8,   AV_CODEC_ID_VP8,           NULL)
RKMPP_DEC(vp9,   AV_CODEC_ID_VP9,           NULL)
