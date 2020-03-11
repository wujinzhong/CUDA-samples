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

//---------------------------------------------------------------------------
//! \file AppDecMultiInput.cpp
//! \brief Source file for AppDecMultiInput sample
//!
//! This sample application demonstrates how to decode multiple raw video files and
//! post-process them with CUDA kernels on different CUDA streams.
//! This sample applies Ripple effect as a part of post processing.
//! The effect consists of ripples expanding across the surface of decoded frames
//---------------------------------------------------------------------------

#include <iostream>
#include <algorithm>
#include <thread>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "NvDecoder/NvDecoder.h"
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"
#include "../Common/AppDecUtils.h"
#include "./sampleINT8API.cpp"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void LaunchRipple(cudaStream_t stream, uint8_t *dpImage, int nWidth, int nHeight, int xCenter, int yCenter, int iTime);
void LaunchOverlayRipple(cudaStream_t stream, uint8_t *dpNv12, uint8_t *dpRipple, int nWidth, int nHeight);
void LaunchMerge(cudaStream_t stream, uint8_t *dpNv12Merged, uint8_t **pdpNv12, int nImage, int nWidth, int nHeight);
void LaunchResize(cudaStream_t stream, float *dstRGB, uint8_t *srcNV12, int srcWidth, int srcHeight, 
		int dstWidth, int dstHeight, int dstChannel, uint8_t* pTmpImage);
int initTRT( SampleINT8API ** sample );
/**
*   @brief  Function to decode frame from media file and post process it using CUDA kernels. 
*   @param  pDec          - Pointer to NvDecoder object which is already initialized
*   @param  szInFilePath  - Path to file to be decoded
*   @param  nWidth        - Width of the decoded video
*   @param  nHeight       - Height of the decoded video
*   @param  apFrameBuffer - Pointer to decoded frame
*   @param  nFrameBuffer  - Capacity of decoder's own circular queue
*   @param  piEnd         - Pointer to hold value of queue's end
*   @param  piHead        - Pointer to hold value of queue's start
*   @param  pbStop        - Boolean to mark the end of post processing by this function
*   @param  stream        - Pointer to CUDA stream
*   @param  xCenter       - X co-ordinate of ripple center
*   @param  yCenter       - Y co-ordinate of ripple center
*   @param  ex            - Stores exception value in case exception is raised
*
*/
void DecProc_Codec_Inf(NvDecoder *pDec, const char *szInFilePath, int nWidth, int nHeight, uint8_t **apFrameBuffer,
    int nFrameBuffer, int *piHead, bool *pbStop, std::exception_ptr &ex, SampleINT8API * sample, int ecIdx, int nThreads)
{
	try
    
    {
	printf("go in DecProc_Codec_Inf for thread%d\n", ecIdx);
	FFmpegDemuxer demuxer(szInFilePath);
	ck(cuCtxSetCurrent(pDec->GetContext()));
	//SampleINT8API * sample = NULL;    
	//initTRT(&sample);
	sampleINT8API_createExecutionContexts(sample, nThreads, ecIdx);

	float *normalizedRGBBuf = NULL;
        int inWidth=224, inHeight=224, inChannel=3;
        ck(cudaMalloc(&normalizedRGBBuf, sizeof(float) * inWidth*inHeight*inChannel*sample->mParams.maxBatchSize));
	uint8_t *pRGBBuf;
    	// decoder->NV12->RGB->RGB[-1,1]->network, pRGBImage stores the transformed RGB, and normalizedRGBBuf stores RGB[-1,1]
	ck(cudaMalloc(&pRGBBuf, sizeof(uint8_t) * nWidth*nHeight*4*sample->mParams.maxBatchSize));

	cudaStream_t stream;
	ck(cudaStreamCreate(&stream));
        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
        uint8_t *pVideo = NULL, **ppFrame;
	*piHead = 0;
	do
        {
            demuxer.Demux(&pVideo, &nVideoBytes);
            pDec->DecodeLockFrame(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
            for (int i = 0; i < nFrameReturned; i++) {
		printf("performing inference for frame%d at thread%d\n", *piHead, ecIdx);
                // Frame buffer is locked, so no data copy is needed here
                apFrameBuffer[*piHead % nFrameBuffer] = ppFrame[i];
		//TRT inference
		if(sample && ((*piHead&(sample->mParams.maxBatchSize-1))==(sample->mParams.maxBatchSize-1)))
		{
			int bs = sample->mParams.maxBatchSize;
			//only one frame is transformed and resized, for simplification.
			LaunchResize( stream, normalizedRGBBuf, apFrameBuffer[*piHead % nFrameBuffer], nWidth, nHeight, inWidth, inHeight, inChannel, pRGBBuf );
			sampleINT8API_infer( sample, ecIdx, bs, normalizedRGBBuf, inWidth, inHeight, inChannel, stream );
		}
		ck(cudaStreamSynchronize(stream));
		pDec->UnlockFrame(&apFrameBuffer[*piHead % nFrameBuffer], 1);
		++*piHead;
            }
        } while (nVideoBytes);
	ck(cudaFree(normalizedRGBBuf));
	ck(cudaFree(pRGBBuf));
	ck(cudaStreamDestroy(stream));
        *pbStop = true;
	printf("maxBatchSize %d\n", sample->mParams.maxBatchSize);
    }
    catch (std::exception&)
    {
        ex = std::current_exception();
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    std::ostringstream oss;
    bool bThrowError = false;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i           Input file path" << std::endl
	<< "-o           Output file path" << std::endl
        ;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName) 
{
    for (int i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
	if (!_stricmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
	ShowHelpAndExit(argv[i]);
    }
}

int initTRT( SampleINT8API ** sample, int nThreads, int maxBatchSize )
{
	std::stringstream ss; ss<<"--maxBatchSize="<<maxBatchSize;
	char str[32]={}; memcpy( str, (char*)(ss.str().c_str()), ss.str().size() );
        int argc = 7;
        char** argv = new char*[argc];
        argv[0] = "./AppDecMultiInput";
        argv[1] = "--model=./data/model.onnx";
        argv[2] = "--image=./data/airliner.ppm";
        argv[3] = "--ranges=./data/resnet50_per_tensor_dynamic_range.txt";
        argv[4] = "--reference=./data/reference_labels.txt";
        argv[5] = "--int8";
	argv[6] = &str[0];
	printf("%s\n", argv[6]);

        sampleINT8API_buildEngine( argc, argv, sample );
        (*sample)->m_execCtxList.resize(nThreads);
	(*sample)->mBuffersList.resize(nThreads);
	//sampleINT8API_createExecutionContexts(*sample, nThreads);
        delete []argv;
        return 0;
}

int ParseCodecArgs( char* szInFilePath, char* szOutFilePath )
{
	int argc0 = 5;
        char** argv0 = new char*[argc0];
        argv0[0] = "./AppDecMultiInput";
        argv0[1] = "-i";
        argv0[2] = "./data/sample_720p.mp4";
        argv0[3] = "-o";
        argv0[4] = "./output";
        ParseCommandLine(argc0, argv0, szInFilePath, szOutFilePath); delete[] argv0;
	CheckInputFile(szInFilePath);
}

void ParseCommandLine2(int argc, char *argv[], int &nThreads, int &maxBatchSize)
{
    for (int i = 1; i < argc; i++) {
    	if (!_stricmp(argv[i], "-thread")) {
            if (++i == argc) {
                ShowHelpAndExit("-thread");
            }
            nThreads = atoi(argv[i]);
            continue;
	}
	else if (!_stricmp(argv[i], "-maxBatchSize")) {
            if (++i == argc) {
                ShowHelpAndExit("-maxBatchSize");
            }
            maxBatchSize = atoi(argv[i]);
            continue;
        }
	else
	{
	    	std::ostringstream oss;
    		oss << "Please follow Options:" << std::endl
        	<< "-maxBatchSize         max batch size used to build CUDA engine" << std::endl
        	<< "-thread               Number of decoding thread" << std::endl
        	;
        	std::cout << oss.str();
        	exit(0);
	}
    }
    assert( nThreads>0 && (nThreads&(nThreads-1))==0 );
    assert( maxBatchSize>0 && (maxBatchSize&(maxBatchSize-1))==0 );
}

int main(int argc, char *argv[])
{
    try
    {
    	ck(cuInit(0));
	int iGpu = 0, nThreads=8, maxBatchSize=2;
	
	ParseCommandLine2( argc, argv, nThreads, maxBatchSize );
	SampleINT8API * sample=NULL;
	int nWidth=0, nHeight=0;
    	std::vector<std::exception_ptr> vExceptionPtrs;
	char szInFilePath[256] = "", szOutFilePath[256] = "out.nv12";
	ParseCodecArgs( szInFilePath, szOutFilePath );
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu)
        {
            std::ostringstream err;
            err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            throw std::invalid_argument(err.str());

        }
        CUcontext cuContext = NULL;
        createCudaContext(&cuContext, iGpu, 0);

	ck(cuCtxSetCurrent(cuContext));
        initTRT(&sample, nThreads, maxBatchSize);

        FFmpegDemuxer demuxer(szInFilePath);
        nWidth = demuxer.GetWidth(); nHeight = demuxer.GetHeight(); int nByte = nWidth * nHeight * 3 / 2;
        // Number of decoders
        const int n = nThreads;
        // Every decoder has its own round queue
        uint8_t *aapFrameBuffer[n][8];
        // Queue capacity
        const int nFrameBuffer = sizeof(aapFrameBuffer[0]) / sizeof(aapFrameBuffer[0][0]);
        bool abStop[n] = {};
        int aiHead[n] = {};
        std::vector <NvThread> vThreads;
        std::vector <std::unique_ptr<NvDecoder>> vDecoders;
        cudaStream_t aStream[n];
        vExceptionPtrs.resize(n);
        
	// Init decoders
	for(int i=0; i<n; i++)
	{
		std::unique_ptr<NvDecoder> dec(new NvDecoder(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec())));
		vDecoders.push_back(std::move(dec));
	}
	StopWatch watch;
	watch.Start();
	int mode = 0;
	for (int i = 0; i < n; i++)
        {
		if( mode==0 )
            		vThreads.push_back(NvThread(std::thread(DecProc_Codec_Inf, vDecoders[i].get(), szInFilePath, nWidth, nHeight, aapFrameBuffer[i],
                				nFrameBuffer, aiHead + i, abStop + i, std::ref(vExceptionPtrs[i]), sample, i, n)));
        }
	for (int i = 0; i < n; i++)
        {
            vThreads[i].join();
        }

        for (int i = 0; i < n; i++)
        {
            if (vExceptionPtrs[i])
            {
                std::rethrow_exception(vExceptionPtrs[i]);
            }
        }
	double sec = watch.Stop();
	int nFrame=0;
	for( int i=0; i<n; i++ )
		nFrame += aiHead[i];

	if(nFrame)
	{
		std::cout<<"concurrency streams: "<<n<<", each decodes mean "<<nFrame/n<<" frames, total fps: "<<nFrame/sec<<std::endl;
	}

        ck(cudaProfilerStop());
        if (nFrame)
        {
            std::cout << "Merged video saved in " << szOutFilePath << ". A total of " << nFrame << " frames were decoded." << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Warning: no video frame decoded. Please don't use container formats (such as mp4/avi/webm) as the input, but use raw elementary stream file instead." << std::endl;
            return 1;
        }

	if(sample)
	{
		sampleINT8API_destroy(sample);
                if( sample!=NULL ) delete sample;
	}
	for(int i=0; i<n; i++)
        {
                ck(cudaStreamDestroy(aStream[i]));
        }

    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what();
        exit(1);
    }
    return 0;
}
