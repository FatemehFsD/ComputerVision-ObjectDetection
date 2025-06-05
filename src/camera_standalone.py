#!/usr/bin/python
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from io import BytesIO
import imageio
import time
import queue
import threading

# intrinsic camera matrix - RGB camera
K_RGB = np.array([[456.5170009681582, 0.0, 332.8146709049914],[0.0, 458.1027893118239, 243.1950147690160],[0.0, 0.0, 1.0]])

def point_cloud(imgDepth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    # intrinsic camera matrix - DepthCamera
    cx = 320.2263488769531
    cy = 193.44775390625
    fx = 476.65203857421875
    fy = 476.65203857421875

    depth = imgDepth[:, :, 1]

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 20000)
    z = np.where(valid, depth / 10000, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))

def get_coord_mouse(event, x, y, flags, param):
    global px1, py1
    if event == cv2.EVENT_LBUTTONUP:
        px1, py1 = x, y
        # print("pixel coord: ",x,y)

def get_coord_mouse_RGB(event, x, y, flags, param):
    global pxRGB, pyRGB
    if event == cv2.EVENT_LBUTTONUP:
        pxRGB, pyRGB = x, y
        print("pixel coord: ", x, y)

if __name__ == '__main__':
    # Initialize the depth device
    openni2.initialize("/home/controller/OpenNI_2.3.0.66/Linux/OpenNI-Linux-x64-2.3.0.66/Redist")
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    # disable LDP (Security distance sensor)
    dev.set_property(0x1080FFBE, 0x00, 4)

    # Start the depth stream
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640,
                           resolutionY=480, fps=30))
    depth_stream.set_mirroring_enabled(False);

    # dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    print(dev.get_image_registration_mode())

    # start RGB video capture
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if (cap.isOpened()):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, 200.0)
    else:
        print("Could not open RGB camera")
        
    print("autoExposure: ",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print("Exposure: ",cap.get(cv2.CAP_PROP_EXPOSURE))

    newFrameTime = 0
    prevFrameTime = 0

    numFrames = 0
    fps = 0

    px1 = 0
    py1 = 0
    x1 = 0
    y1 = 0
    z1 = 1

    pz = 0

    pxRGB = 0
    pyRGB = 0

    cv2.namedWindow("Depth Image")
    cv2.setMouseCallback("Depth Image", get_coord_mouse)

    cv2.namedWindow("RGB Image")
    cv2.setMouseCallback("RGB Image", get_coord_mouse_RGB)

    # Loop
    while True:
        numFrames = numFrames + 1

        # Capture RGBframe-by-frame
        ret, imgRGB = cap.read()

        # Display the resulting frame

        # Grab a new depth frame
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        # Put the depth frame into a numpy array and reshape it
        imgDepth = np.frombuffer(frame_data, dtype=np.uint16)
        imgDepth.shape = (1, 480, 640)
        imgDepth = np.concatenate((imgDepth, imgDepth, imgDepth), axis=0)
        imgDepth = np.swapaxes(imgDepth, 0, 2)
        imgDepth = np.swapaxes(imgDepth, 0, 1)

        if numFrames == 20:
            # measure frame rate
            newFrameTime = time.time()
            fps = 20 / (newFrameTime - prevFrameTime)
            prevFrameTime = newFrameTime
            numFrames = 0

        # put fps on RGB image
        cv2.putText(imgRGB, str(fps), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
        cv2.putText(imgRGB, str(cap.get(cv2.CAP_PROP_EXPOSURE)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))

        # print("Distance image center: ",imgDepth[200][320][2]/10000,"[m]")

        # create pointcloud
        pcd = point_cloud(imgDepth)

        # normalise depthImage
        normedDepth = cv2.normalize(imgDepth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colorDepth = cv2.applyColorMap(normedDepth, cv2.COLORMAP_JET)

        # mouse pixel to coord
        x1, y1, z1 = pcd[py1, px1, :]
        # draw last mouse clic on image
        cv2.drawMarker(colorDepth, (px1, py1), (100, 100, 100), thickness=2)
        # write coordinates to image
        text = "coords: " + str(px1) + " " + str(py1) + " [px], " + "{:.3f}".format(x1) + " " + "{:.3f}".format(
            y1) + " " + "{:.3f}".format(z1) + " [m]"
        cv2.putText(colorDepth, text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

        # correct pointcloud with extrinsic parameters (0.02m shifting in x)
        imgPCDcorr = np.zeros((imgRGB.shape[0], imgRGB.shape[1], 1))
        T1 = np.array([[1, 0, 0, -0.02],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # transform point on depth image to rgb
        point = np.array([x1, y1, z1, 1])
        pointCorr = np.matmul(T1, point)
        px, py, lambda1 = np.matmul(K_RGB, pointCorr[0:3])
        if ~np.isnan(px) and ~np.isnan(px):
            px = int(px / lambda1)
            py = int(py / lambda1)
            cv2.drawMarker(imgRGB, (px, py-10), (255, 0, 0), thickness=2) #y-direction 10px difference

        
        # Display images
        cv2.imshow('RGB Image', imgRGB)
        cv2.imshow("Depth Image", colorDepth)

        key = cv2.waitKey(1) & 0xFF

        # If the 'c' key is pressed, break the while loop
        if key == ord("c"):
            break

    # Close all windows and unload the depth device
    depth_stream.close()
    dev.close()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()
