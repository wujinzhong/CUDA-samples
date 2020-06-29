#standard imports
import cv2
import numpy as np

def getFileName( yml0 ):
    name0 = yml0.split('.')
    name0 = name0[len(name0)-2].split('/')
    name0 = name0[len(name0)-1]
    print(name0)
    return name0

def compareYaml( yml0, yml1 ):
    
    name0 = getFileName( yml0 )
    name1 = getFileName( yml1 )    

    yml0 = cv2.FileStorage(yml0, cv2.FileStorage_READ)
    yml1 = cv2.FileStorage(yml1, cv2.FileStorage_READ)
    yml0 = yml0.getNode("data").mat()
    yml1 = yml1.getNode("data").mat()
    diff = cv2.absdiff( yml0, yml1 )
    sum = diff.sum()

    name = name0+"_vs_"+name1

    print( name, ": absdiff_sum ", sum )
    
    fs = cv2.FileStorage( name+".yml", cv2.FileStorage_WRITE )
    fs.write("data", diff)

    diff *= 30
    diff = np.clip(diff, 0, 255)
    diff = np.uint8(diff)
    cv2.imwrite(name+".bmp", diff)

mask_type = 1; # 1 for rect255, 2 for polygon

if mask_type==1:
    # Read images
    #src = cv2.imread("/thor/OpenCVNew/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/output/opencv-seamless-cloning-example-rect-all-255.bmp")
    src = cv2.imread("/thor/OpenCVNew/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/output/opencv-seamless-cloning-example-rect-all-255-2400x1552.bmp")

    #src = cv2.imread( "./images/sky.jpg" )
    dst = cv2.imread("/thorRaid/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/ucRGB_Output.bmp")
    print(src.shape, src.dtype)
    print(dst.shape, dst.dtype)

    height = src.shape[0]
    width = src.shape[1]
    channels = src.shape[2]

    diff = cv2.absdiff(src, dst)
    sum = diff.sum()
    print(sum)

    #for row in range(height):
    #    for col in range(width):
    #        for channel in range(channels):
    #            if diff[row][col][channel] != 0:
    #                print( row, col, channel, diff[row][col][channel] )


    fs = cv2.FileStorage('diff.yml', cv2.FileStorage_WRITE)
    fs.write("diff", diff)

    # Save result
    diff *= 30

    diff *= 255
    diff = np.clip(diff, 0, 255)
    diff = np.uint8(diff)
    cv2.imwrite("./images/diff.bmp", diff)

    compareYaml( "/thor/OpenCVNew/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/mod_diff0.yml",
            "/thorRaid/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/g2.yml" )
    compareYaml( "/thor/OpenCVNew/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/mod_diff1.yml",
            "/thorRaid/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/g1.yml" )
    compareYaml( "/thor/OpenCVNew/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/mod_diff2.yml",
            "/thorRaid/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/g0.yml" )
elif mask_type==2:
    # Read images
    src = cv2.imread("/thor/OpenCV3.4.5/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/output/opencv-seamless-cloning-example-polygon.bmp")
    #src = cv2.imread( "./images/sky.jpg" )
    dst = cv2.imread("/thor/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/ucRGB_Output.bmp")
    print(src.shape, src.dtype)
    print(dst.shape, dst.dtype)

    height = src.shape[0]
    width = src.shape[1]
    channels = src.shape[2]

    diff = cv2.absdiff(src, dst)
    sum = diff.sum()
    print(sum)

    #for row in range(height):
    #    for col in range(width):
    #        for channel in range(channels):
    #            if diff[row][col][channel] != 0:
    #                print( row, col, channel, diff[row][col][channel] )


    fs = cv2.FileStorage('diff.yml', cv2.FileStorage_WRITE)
    fs.write("diff", diff)

    # Save result
    diff *= 1

    diff *= 1
    diff = np.clip(diff, 0, 255)
    diff = np.uint8(diff)
    cv2.imwrite("./images/diff.bmp", diff)

    compareYaml( "/thor/OpenCV3.4.5/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/mod_diff0.yml",
            "/thor/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/g2.yml" )
    compareYaml( "/thor/OpenCV3.4.5/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/mod_diff1.yml",
            "/thor/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/g1.yml" )
    compareYaml( "/thor/OpenCV3.4.5/opencv-3.4.5/samples/cpp/seamlessClone_OpenCV/mod_diff2.yml",
            "/thor/cuda/samples-10.1/7_CUDALibraries/seamlessClone/output/g0.yml" )
else:
    print("error mask type!")
    assert(False)
    exit(1)

exit(0)
