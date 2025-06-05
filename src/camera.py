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
# we keep a queue, the video capture devices, and the thread
q = None
cap = None
t = None

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


def start_camera_and_wait():
    global cap
    global q
    global t
    # Initialize the depth device
    openni2.initialize("/home/controller/OpenNI_2.3.0.66/Linux/OpenNI-Linux-x64-2.3.0.66/Redist")
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    # disable LDP (Security distance sensor)
    dev.set_property(0x1080FFBE, 0x00, 4)

    # Start the depth stream
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 400, fps = 30))
    depth_stream.set_mirroring_enabled(False);

    #dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    print(dev.get_image_registration_mode())

    # start RGB video capture
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        print('opened with index 1')

    if (cap.isOpened()):
        cap.set(cv2.CAP_PROP_FOURCC, 1196444237)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        print("Could not open RGB camera")

    # we use a thread to keep reading from the RGB camera buffer, and keep the last frame
    q = queue.Queue()
    t = threading.Thread(target=_reader)
    t.daemon = True
    t.start()

    return dev, cap, depth_stream

# read frames as soon as they are available, keeping only most recent one
def _reader():
    global q
    global cap
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      if not q.empty():
        try:
          q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      q.put(frame)

def read():
    global q
    return q.get()

def capture_frame(cap, depth_stream, detection_callback, save_images=False):

    # Capture RGBframe-by-frame
    # ret, imgBGR = cap.read()
    # We read from the queue here
    imgBGR = read()
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    # Grab a new depth frame
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    # Put the depth frame into a numpy array and reshape it
    imgDepth = np.frombuffer(frame_data, dtype=np.uint16)
    imgDepth.shape = (1, 400, 640)
    imgDepth = np.concatenate((imgDepth, imgDepth, imgDepth), axis=0)
    imgDepth = np.swapaxes(imgDepth, 0, 2)
    imgDepth = np.swapaxes(imgDepth, 0, 1)

    # create pointcloud
    pcd = point_cloud(imgDepth)

    # normalise depthImage
    normedDepth = cv2.normalize(imgDepth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colorDepth = cv2.applyColorMap(normedDepth, cv2.COLORMAP_JET)

    # correct pointcloud with extrinsic parameters (0.02m shifting in x)
    imgPCDcorr = np.zeros((imgRGB.shape[0], imgRGB.shape[1], 1))
    T1 = np.array([[1, 0, 0, -0.01],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # invoke socket detection
    maskedImg, socket_poses, masks = detection_callback(imgRGB, pcd)
    if (len(socket_poses) >= 1):
        if (save_images):
            # save RGB image to to memory file
            # skimage.io.imsave("static/rgb_image.png", imgRGB)
            rgb_buf = BytesIO()
            imageio.imwrite(rgb_buf, imgRGB, format='png')

            # add marker in the center of the socket
            # transform point on depth image to rgb
            if(socket_poses[0] != None):
                for center in socket_poses[0]:
                    print(center)
                    point = np.array([center[0], center[1], center[2], 1])
                    pointCorr = np.matmul(T1, point)
                    px, py, lambda1 = np.matmul(K_RGB, pointCorr[0:3])
                    if ~np.isnan(px) and ~np.isnan(px):
                        px = int(px / lambda1)
                        py = int(py / lambda1)
                        cv2.drawMarker(imgRGB, (px, py), (255, 0, 0), thickness=2)

                if (len(socket_poses[0]) == 2):
                    pose = socket_poses[0][0]
                    point = np.array([pose[0], pose[1], pose[2], 1])
                    point_normal = np.array([pose[0]+socket_poses[1][0], pose[1]+socket_poses[1][1], pose[2]+socket_poses[1][2], 1])
                    point_horizontal = np.array([pose[0]+socket_poses[2][0], pose[1]+socket_poses[2][1], pose[2]+socket_poses[2][2], 1])
                    point_vertical = np.array([pose[0]+socket_poses[3][0], pose[1]+socket_poses[3][1], pose[2]+socket_poses[3][2], 1])

                    pointCorr = np.matmul(T1, point)
                    point_normal_corr = np.matmul(T1, point_normal)
                    point_horizontal_corr = np.matmul(T1, point_horizontal)
                    point_vertical_corr = np.matmul(T1, point_vertical)

                    px, py, lambda1 = np.matmul(K_RGB, pointCorr[0:3])
                    px_normal, py_normal, lambda1_normal = np.matmul(K_RGB, point_normal_corr[0:3])
                    px_horizontal, py_horizontal, lambda1_horizontal = np.matmul(K_RGB, point_horizontal_corr[0:3])
                    px_vertical, py_vertical, lambda1_vertical = np.matmul(K_RGB, point_vertical_corr[0:3])

                    px = int(px / lambda1)
                    py = int(py / lambda1)
                    px_normal = int(px_normal / lambda1_normal)
                    py_normal = int(py_normal / lambda1_normal)
                    px_horizontal = int(px_horizontal / lambda1_horizontal)
                    py_horizontal = int(py_horizontal / lambda1_horizontal)
                    px_vertical = int(px_vertical / lambda1_vertical)
                    py_vertical = int(py_vertical / lambda1_vertical)

                    cv2.arrowedLine(imgRGB , (px, py), (px_normal, py_normal), (0, 0, 255), thickness=2)
                    cv2.arrowedLine(imgRGB , (px, py), (px_horizontal, py_horizontal), (255, 0, 0), thickness=2)
                    cv2.arrowedLine(imgRGB , (px, py), (px_vertical, py_vertical), (0, 255, 0), thickness=2)

            masked_buf = BytesIO()
            imageio.imwrite(masked_buf, maskedImg, format='png')
            marked_buf = BytesIO()
            imageio.imwrite(marked_buf, imgRGB, format='png')
            depth_buf = BytesIO()
            imageio.imwrite(depth_buf, colorDepth, format='png')
            return socket_poses, rgb_buf, masked_buf, marked_buf, depth_buf
        return socket_poses
