This sample shows the conversion of each GPU decoded frames NV12-->RGB. 

Steps:
1．	install Video_Codec_SDK_9.0.20;
2．	replace these files (AppDecImageProvider.cpp,ColorSpace.cu,ColorSpace.h,NvCodecUtils.h) in Samples/ dir;
3．	cd ./Samples/AppDec/AppDecImageProvider/;
4．	make
5．	./AppDecImageProvider -h
6．	./AppDecImageProvider -i ./test.mp4 -o output.rgb24 -of rgb24 -gpu 0
