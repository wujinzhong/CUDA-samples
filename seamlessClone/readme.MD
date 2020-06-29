SEAMLESSCLONE PROJECT OVERVIEW

1.	PROJECT OVERVIEW
1.1.	SEAMLESS CLONE
2.	Guide

1.	Project Overview
Seamless clone is an advanced image editing feature published in this paper, “Poisson Image Editing”, OpenCV has this feature too. This project re-implements seamless clone feature for NVIDIA GPU platform, using cuda programming. 

1.1.	Seamless Clone
Seamless clone is an advanced image editing feature, not like direct cloning image patch from source image to destination image, which has a drawback of inconsistent boundary colors. Seamless clone uses a guidance of vector in differential field and solve a Poisson equation with boundary pixels from destination image. This tech beautifully solves the problem of inconsistent drawback between source image and destination image. Details please refer to the original paper.
 
This project is verified and tested with V100/T4 platform:
GPU driver	440.64
CUDA	10.2
OpenCV	3.4.5
OS	Ubuntu 18.04.4 LTS \n \l
GPU	V100/T4

2.	Guide
Steps to run the sample code:
1.	Install CUDA toolkit and copy “/usr/local/cuda/samples/” to “/home/usr_name/”.
2.	We use this command line:
“nvidia-docker run -it -v /home/usr_name/:/usr_name --net host nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 bash”
for testing and install OpenCV3.4.5 from source code to dir “/usr_name/OpenCV-3.4.5/”.
3.	Copy “seamless-clone.7z” to “/usr_name/” and unzip it;
4.	Copy “seamlessClone” dir to “/usr_name/samples-10.1/7_CUDALibraries/”. Goto “./seamlessClone” and run “./make2.sh” and the output image is here “./seamlessClone/output/ucRGB_Output.bmp”, this is the seamless cloned image from source patch “./images/src.yml” and destination image “./images/dst.yml”. We use yml format for source/destination images to make sure the input is same for our implementation and OpenCV. The original jpg format is “sky.jpg” of size 1600x898, source patches {“airplane2.jpg” of size 300x194, “airplane154x100.jpg”, “airplane592x592.jpg”} for reference.
Open “./seamlessClone/output/ucRGB_Output.bmp” to see if the output is correct.
5.	Copy “seamlessClone_OpenCV” dir to “/usr_name/OpenCV-3.4.5/opencv-3.4.5/samples/cpp/” and goto “./seamlessClone_OpenCV”. “make” it and run “./seamlessClone_OpenCV”. Check the OpenCV output here “./output/opencv-seamless-cloning-example-rect-all-255.bmp”. 
6.	Copy “compare” dir to “/usr_name” and goto “/usr_name/compare”, open file “./vs.py” find and modify output file path “xxxxx/opencv-seamless-cloning-example-rect-all-255.bmp”, ”xxxx/ucRGB_Output.bmp”, “xxxxx/g*.yml”, “xxxxx/mod_diff*.yml”, to reflect your file path.
7.	Run “python vs.py” and the output look like this:
((898, 1600, 3), dtype('uint8'))
((898, 1600, 3), dtype('uint8'))
152
mod_diff0
g2
('mod_diff0_vs_g2', ': absdiff_sum ', 0.0)
mod_diff1
g1
('mod_diff1_vs_g1', ': absdiff_sum ', 0.0)
mod_diff2
g0
('mod_diff2_vs_g0', ': absdiff_sum ', 0.0)	src and dst images size and format

This number show the absdiff_sum for src/dst images, ourput from our CUDA implementation and OpenCV. In our one testing, absdiff_sum=152 for a mask patch size 300x194x3 is reasonable, with max_diff=1, check “diff.yml” for details.
Compare channel 0 (mod_diff0<-->g2) before sending to Poisson solver.
0.0 means no diff found.
Compare channel 1 (mod_diff1<-->g1) before sending to Poisson solver.
0.0 means no diff found.
Compare channel 2 (mod_diff2<-->g0) before sending to Poisson solver.
0.0 means no diff found.

Turn on/off these macros to check the performance and precision changing. Enabling FFT makes the max_diff down to 1/255, with the cost of a little performance drop.
#define SCDEBUG true
#define SC_FFT_ENABLE true
