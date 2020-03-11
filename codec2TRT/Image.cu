/*
* Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <stdint.h>
#include <cuda_runtime.h>
#include "../Utils/NvCodecUtils.h"
//#include "../Utils/ColorSpace.h"
#define SLEEP_TIME 0

__constant__ float matYuv2Rgb[3][3];
__constant__ float matRgb2Yuv[3][3];
union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};
typedef enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 0,
    ColorSpaceStandard_BT601 = 2,
    ColorSpaceStandard_BT2020 = 4
} ColorSpaceStandard;
template void Nv12ToColor32<RGBA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v);

inline __device__ double sleep(int n) {
    double d = 1.0;
    for (int i = 0; i < n; i++) {
        d += sin(d);
    }
    return d;
}

static __global__ void Ripple(uint8_t *pImage, int nWidth, int nHeight, int xCenter, int yCenter, int iTime) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nWidth || iy >= nHeight) {
        return;
    }
    float dx = ix - xCenter, dy = iy - yCenter, d = sqrtf(dx * dx + dy * dy), dmax = sqrtf(nWidth * nWidth + nHeight * nHeight) / 2.0f;
    pImage[iy * nWidth + ix] = (uint8_t)(127.0f * (1.0f - d / dmax) * sinf((d - iTime * 10)* 0.1) + 128.0f);
    sleep(SLEEP_TIME);
}

void LaunchRipple(cudaStream_t stream, uint8_t *dpImage, int nWidth, int nHeight, int xCenter, int yCenter, int iTime) {
    Ripple<<<dim3((nWidth + 15) / 16, (nHeight + 15) / 16), dim3(16, 16), 0, stream>>>(dpImage, nWidth, nHeight, xCenter, yCenter, iTime);
}

inline __device__ uint8_t clamp(int i) {
    return (uint8_t)min(max(i, 0), 255);
}

static __global__ void OverlayRipple(uint8_t *pNv12, uint8_t *pRipple, int nWidth, int nHeight) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nWidth || iy >= nHeight) {
        return;
    }
    pNv12[iy * nWidth + ix] = clamp(pNv12[iy * nWidth + ix] + (pRipple[iy * nWidth + ix] - 127.0f) * 0.8f);
    sleep(SLEEP_TIME);
}

void LaunchOverlayRipple(cudaStream_t stream, uint8_t *dpNv12, uint8_t *dpRipple, int nWidth, int nHeight) {
    OverlayRipple<<<dim3((nWidth + 15) / 16, (nHeight + 15) / 16), dim3(16, 16), 0, stream>>>(dpNv12, dpRipple, nWidth, nHeight);
}

static __global__ void Merge(uint8_t *pNv12Merged, uint8_t **apNv12, int nImage, int nWidth, int nHeight) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nWidth / 2 || iy >= nHeight / 2) {
        return;
    }
    uint2 y01 = {}, y23 = {}, uv = {};
    for (int i = 0; i < nImage; i++) {
        uchar2 c2;
        c2 = *(uchar2 *)(apNv12[i] + nWidth * iy * 2 + ix * 2);
        y01.x += c2.x; y01.y += c2.y;
        c2 = *(uchar2 *)(apNv12[i] + nWidth * (iy * 2 + 1) + ix * 2);
        y23.x += c2.x; y23.y += c2.y;
        c2 = *(uchar2 *)(apNv12[i] + nWidth * (nHeight + iy) + ix * 2);
        uv.x += c2.x; uv.y += c2.y;
    }
    *(uchar2 *)(pNv12Merged + nWidth * iy * 2 + ix * 2) = uchar2 {(uint8_t)(y01.x / nImage), (uint8_t)(y01.y / nImage)};
    *(uchar2 *)(pNv12Merged + nWidth * (iy * 2 + 1) + ix * 2) = uchar2 {(uint8_t)(y23.x / nImage), (uint8_t)(y23.y / nImage)};
    *(uchar2 *)(pNv12Merged + nWidth * (nHeight + iy) + ix * 2) = uchar2 {(uint8_t)(uv.x / nImage), (uint8_t)(uv.y / nImage)};
    sleep(SLEEP_TIME);
}

void LaunchMerge(cudaStream_t stream, uint8_t *dpNv12Merged, uint8_t **pdpNv12, int nImage, int nWidth, int nHeight) {
    uint8_t **dadpNv12;
    ck(cudaMalloc(&dadpNv12, sizeof(uint8_t *) * nImage));
    ck(cudaMemcpy(dadpNv12, pdpNv12, sizeof(uint8_t *) * nImage, cudaMemcpyHostToDevice));
    Merge<<<dim3((nWidth + 15) / 16, (nHeight + 15) / 16), dim3(8, 8), 0, stream>>>(dpNv12Merged, dadpNv12, nImage, nWidth, nHeight);
    ck(cudaFree(dadpNv12));
}

void inline GetConstants(int iMatrix, float &wr, float &wb, int &black, int &white, int &max) {
    // Default is BT709
    wr = 0.2126f; wb = 0.0722f;
    black = 16; white = 235;
    max = 255;
    if (iMatrix == ColorSpaceStandard_BT601) {
        wr = 0.2990f; wb = 0.1140f;
    } else if (iMatrix == ColorSpaceStandard_BT2020) {
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
    }
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

void SetMatYuv2Rgb(int iMatrix) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbol(matYuv2Rgb, mat, sizeof(mat));
}

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int 
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit 
        r = (YuvUnit)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);
    
    Rgb rgb{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(rgb.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(rgb.c.r)) {
        rgb.c.r = r >> nShift;
        rgb.c.g = g >> nShift;
        rgb.c.b = b >> nShift;
    } else {
        rgb.c.r = r << nShift;
        rgb.c.g = g << nShift;
        rgb.c.b = b << nShift;
    }
    return rgb;
}

template <class COLOR32>
void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

static __global__ void Resize(float *dstRGB, uint8_t *srcRGBA, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    int srcX = ix*(srcWidth-1)/(dstWidth-1);
    int srcY = ix*(srcHeight-1)/(dstHeight-1);
    int srcIdx = srcY*srcWidth + srcX;
    int dstIdx = iy*dstWidth + ix;

    int sliceSz = dstWidth * dstHeight;
    dstRGB[0*sliceSz + dstIdx] = srcRGBA[srcIdx*4+0]*(2.0f/255.0f)-1.0f;
    dstRGB[1*sliceSz + dstIdx] = srcRGBA[srcIdx*4+1]*(2.0f/255.0f)-1.0f;
    dstRGB[2*sliceSz + dstIdx] = srcRGBA[srcIdx*4+2]*(2.0f/255.0f)-1.0f;
}

void LaunchResize(cudaStream_t stream, float *dstRGB, uint8_t *srcNV12, int srcWidth, int srcHeight, 
		int dstWidth, int dstHeight, int dstChannel, uint8_t* pTmpImage)
{
    assert(dstChannel==3); // only support NV12 to RGB conversion
    Nv12ToColor32<RGBA32>((uint8_t *)srcNV12, srcWidth, (uint8_t*)pTmpImage, 4*srcWidth, srcWidth, srcHeight);

    //if( srcWidth==dstWidth && srcHeight==dstHeight )
    //{
    //    DirectCopy<<<dim3((dstWidth + 15) / 16, (dstHeight + 15) / 16), dim3(8, 8), 0, stream>>>(dstRGB, pTmpImage, nWidth, nHeight);
    //}
    //else
    {
        Resize<<<dim3((dstWidth + 15) / 16, (dstHeight + 15) / 16), dim3(8, 8), 0, stream>>>(dstRGB, pTmpImage, srcWidth, srcHeight, dstWidth, dstHeight);
    }
}
