# From Python
# It requires OpenCV installed for Python
#coding=utf-8
import sys

import cv2
import os
from sys import platform
import argparse
import shutil
import time
import numpy


# clip video into frames
def clipVideoIntoFrames(path,outPath):
    capVideo = cv2.VideoCapture(path)
    if not capVideo.isOpened():
        print("[ERROR] Could not open the video: " + path)
        sys.exit(-1)
    res, frame = capVideo.read()
    file_name = os.path.basename(path)
    file_name = file_name.split('.')[0]
    time_f = 5
    c = 1
    count = 0
    while res:
        if c % time_f == 0:
            cv2.imwrite(outPath + "/" + file_name + "_f_" + str(count) + ".jpg", frame)
            count += 1
        res, frame = capVideo.read()
        c = c + 1
    capVideo.release()
    return


# output a image with only a person
def getPersonImage(pose_key_point, image_orign):
    x_max, y_max = 0, 0
    sp = image_orign.shape
    y_min = sp[0]
    x_min = sp[1]
    for data in pose_key_point:
        if (data[2] < 0.5):
            continue
        x_max=max(x_max,int(data[0]))
        y_max=max(y_max,int(data[1]))
        x_min=min(x_min,int(data[0]))
        y_min=min(y_min,int(data[1]))
    if x_max <= x_min or y_max <= y_min:
        return None
    extend = 20  # this is for extending clip to get the whole person
    x_max=rangeAdd(x_max,extend,x_max,sp[1])
    x_min=rangeAdd(x_min,-extend,0,x_min)
    # suppose that the distance from nose to neck equal to the distance from nose to hair
    extend_y_up = int(pose_key_point[0][1] - pose_key_point[1][1])
    y_max=rangeAdd(y_max,max(extend_y_up,extend),y_max,sp[0])
    y_min=rangeAdd(y_min,-extend,0,y_min)
    image = image_orign[y_min:y_max, x_min:x_max]
    return image


def processFrames(imagePaths, outPath,opWrapper):
    if os.path.exists(outPath):
        shutil.rmtree(outPath)
    os.makedirs(outPath)
    image_num= len(imagePaths)
    image_count = 1
    for imagePath in imagePaths:
        sys.stdout.write("\r" + "[INFO] Image Processing:" + str(image_count) + "/" + str(image_num))
        sys.stdout.write(" | Image : "+str(os.path.basename(imagePath)))
        datum = op.Datum()
        image_name = os.path.basename(imagePath).split('.')[0]
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        i = 0
        image_count += 1
        if numpy.ndim(datum.poseKeypoints)==0:
            continue
        for pose_key_point in datum.poseKeypoints:
            image_out = getPersonImage(pose_key_point, imageToProcess)
            if image_out is None:
                continue
            cv2.imwrite(outPath + "/" + image_name + "_" + str(i) + ".jpg", image_out)
            i += 1
    sys.stdout.write("\n")
    return


def processVideos(dirPaths):
    framePath=dirPaths+"/video_frame_output"
    if os.path.exists(framePath):
        shutil.rmtree(framePath)
    files=os.listdir(dirPaths)
    for file in files:
        if ('.mp4' in file) or ('.avi' in file):
            if not os.path.exists(framePath):
                os.mkdir(framePath)
            sys.stdout.write("\r[INFO] Get Frames from Video:" + file)
            clipVideoIntoFrames(dirPaths+"/"+file,framePath)
    sys.stdout.write("\r[INFO] Get Frames from Video----Finished\n")
    return

#add function, can control the result in a range
def rangeAdd(x,y,min,max):
    s=x+y
    if s<min:
        return min
    elif s>max:
        return max
    return s

def clipVideoToPersonImg(video_dir):
    print("[INFO] Process Video in "+video_dir)
    # for detect joints in pictures
    processVideos(video_dir)

    # Config OpenPose
    params = dict()
    params["model_folder"] = "../../../models/"
    params["image_dir"] = video_dir + "/video_frame_output/"
    # print("[DEBUG] frame's path" + frame_path)
    # params["write_json"] = frame_path
    # params["write_images"] = frame_path

    if not os.path.exists(video_dir + "/video_frame_output/"):
        return

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    # Read frames on directory
    imagePaths = op.get_images_on_directory(video_dir + "/video_frame_output/");
    print("[INFO] Process Images in "+video_dir+"/body_clip_output")
    # Process and display images
    processFrames(imagePaths, video_dir + "/body_clip_output",opWrapper)
    shutil.rmtree(video_dir + "/video_frame_output/")
    return

def deepProcess(rootdir,deep_flag):
    # process in sub dir
    if deep_flag:
        files = os.listdir(rootdir)
        for file in files:
            if os.path.isdir(rootdir + "/" + file):
                deepProcess(rootdir + "/" + file,True)
    # process in root dir
    clipVideoToPersonImg(rootdir)
    print("****************************")
    return




try:

    #sys.setrecursionlimit(10000)
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # get param
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default="../../../examples/media/", help="Process a directory of videos")
    parser.add_argument('--deep_search',action="store_true",default=False,help="If process dirs in this dir")
    args = parser.parse_args()

    start = time.time()
    # process in root dir
    deepProcess(args.video_dir,args.deep_search)

    end = time.time()
    print("clip video to person images successfully finished. Total time: " + str(end - start) + " seconds")


except Exception as e:
    print(e)
    sys.exit(-1)
