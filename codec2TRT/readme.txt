command line:
./AppDecMultiInput -thread 4 -maxBatchSize 8


modify parameters if necessary:
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