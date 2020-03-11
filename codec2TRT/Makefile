################################################################################
#
# Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

include ../../common.mk

NVCCFLAGS := $(CCFLAGS)

LDFLAGS += -pthread
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lnvcuvid -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser
LDFLAGS += $(shell pkg-config --libs libavcodec libavutil libavformat)

INCLUDES += $(shell pkg-config --cflags libavcodec libavutil libavformat)

#EXTRA_DIRECTORIES = ./common
#MAKEFILE ?= ./Makefile.config
#include $(MAKEFILE)

# Target rules
all: build

build: AppDecMultiInput

NvDecoder.o: ../../NvCodec/NvDecoder/NvDecoder.cpp ../../NvCodec/NvDecoder/NvDecoder.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

TRTlogger.o: ./common/logger.cpp ./common/logger.h 
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

Image.o: Image.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

AppDecMultiInput.o: AppDecMultiInput.cpp ../../NvCodec/NvDecoder/NvDecoder.h \
                    ../../Utils/NvCodecUtils.h ../Common/AppDecUtils.h \
                    ../../Utils/Logger.h\
		    sampleINT8API.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

AppDecMultiInput: AppDecMultiInput.o Image.o NvDecoder.o TRTlogger.o
	$(GCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf AppDecMultiInput AppDecMultiInput.o Image.o NvDecoder.o TRTlogger.o
