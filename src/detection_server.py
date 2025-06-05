from flask import Flask, Response
from flask import render_template
import os
# DETECTRON
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

# camera script
import camera
# pose estimation
# pose estimation
import socket_pose_estimation
from dict2xml import dict2xml
from datetime import datetime
from flask_caching import Cache
from functools import wraps
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("")
CLASS_LABEL = "socket"
TRAIN_ERROR = "Sorry, train is not supporeted yet. Please come back later"

app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': 'simple', "CACHE_DEFAULT_TIMEOUT": 60})
dev = None
cap = None
depth_stream = None

@app.route('/')
def hello_world():
    return 'Hello, this is the Socket-Detection Web Application Server'

@app.route('/detect')
def capture_and_detect():
    timestamp = datetime.timestamp(datetime.now())
    socket_poses = camera.capture_frame(cap, depth_stream, process_image)
    return Response(dict2xml(poses_to_dict(socket_poses[4], timestamp)), mimetype='text/xml')

@app.route('/detect_with_image')
def capture_detect_show():
    timestamp = datetime.timestamp(datetime.now())
    socket_poses, rgb_buf, masked_buf, marked_buf, depth_buf = camera.capture_frame(cap, depth_stream, process_image, save_images=True)
    # save image buffer values in cache
    cache.set("rgb_image_" + str(timestamp), rgb_buf.getvalue())
    cache.set("masked_image_" + str(timestamp), masked_buf.getvalue())
    cache.set("marked_image_" + str(timestamp), marked_buf.getvalue())
    cache.set("depth_image_" + str(timestamp), depth_buf.getvalue())
    return render_template('detect.html', poses=socket_poses[4], timestamp=timestamp)

@app.route('/test_xml')
def test_xml():
    socket_poses = [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1]],
                  [[1.0, 1.0, 1.0],[0.0, 0.0, 0.0, 1]]
                 ]   
    return Response(dict2xml(poses_to_dict(socket_poses)), mimetype='text/xml')

@app.route('/cache/<timestamp>/rgb_image.png')
def get_rgb_image_from_cache(timestamp):
    return Response(cache.get("rgb_image_" + timestamp), mimetype='image/png')

@app.route('/cache/<timestamp>/masked_image.png')
def get_masked_image_from_cache(timestamp):
    return Response(cache.get("masked_image_" + timestamp), mimetype='image/png')

@app.route('/cache/<timestamp>/marked_image.png')
def get_marked_image_from_cache(timestamp):
    return Response(cache.get("marked_image_" + timestamp), mimetype='image/png')

@app.route('/cache/<timestamp>/depth_image.png')
def get_depth_image_from_cache(timestamp):
    return Response(cache.get("depth_image_" + timestamp), mimetype='image/png')

"""
#def shutdown(signum, frame):
def poses_to_dict(pose, timestamp):
    result = {}
    td = {}
    result["socketDetection"] = td
    td["time"] = timestamp
    socket = {}
    socket['x'] = {"float": pose[0][0]}
    socket['y'] =  {"float": pose[0][1]}
    socket['z'] =  {"float": pose[0][2]}
    socket['qx'] =  {"float": pose[1][0]}
    socket['qy'] =  {"float": pose[1][1]}
    socket['qz'] =  {"float": pose[1][2]}
    socket['qw'] =  {"float": pose[1][3]}
    td["socket"] = socket
    return result
"""

def poses_to_dict(pose, timestamp):
    result = {}
    td = {}
    result["socketDetection"] = td
    td["time"] = timestamp
    socket = {}
    socket['x'] = {"float": pose[0][0]}
    socket['y'] =  {"float": pose[0][1]}
    socket['z'] =  {"float": pose[0][2]}
    socket['aa'] =  {"float": pose[1][0][0]}
    socket['ab'] =  {"float": pose[1][1][0]}
    socket['ac'] =  {"float": pose[1][2][0]}
    socket['ba'] =  {"float": pose[1][0][1]}
    socket['bb'] =  {"float": pose[1][1][1]}
    socket['bc'] =  {"float": pose[1][2][1]}
    socket['ca'] =  {"float": pose[1][0][2]}
    socket['cb'] =  {"float": pose[1][1][2]}
    socket['cc'] =  {"float": pose[1][2][2]}
    td["socket"] = socket
    return result


def process_image(image, point_cloud):
    global predictor
    # Run neural network inference
    masked_image, outputs = GetMask(predictor, image)
    # Extract mask from output, shape (4, 480, 640)
    mask = outputs["instances"].to("cpu").pred_masks.numpy()

    # Run pose estimation on athe mask to find socket
    socket_poses = socket_pose_estimation.extract_pose(mask, point_cloud)
    return masked_image, socket_poses, mask

# Timing function decorator
def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed

# get Mask method
@timeit
def GetMask(predictor, im):
    outputs = predictor(im)
    for d in ["train", "val"]:
        MetadataCatalog.get("{}_".format(CLASS_LABEL) + d).set(thing_classes=[CLASS_LABEL])
    socket_metadata = MetadataCatalog.get("socket_train")
    cpu_device = torch.device("cpu")
    v = Visualizer(im[:, :, ::-1],
                   socket_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1], outputs

if __name__ == '__main__':
    print("Welcome to Detection Server")

    # Configurations
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set a custom testing threshold

    print("Config and weights sucessfully loaded.")

    for d in ["train", "val"]:
        MetadataCatalog.get("{}_".format(CLASS_LABEL) + d).set(thing_classes=[CLASS_LABEL])
    socket_metadata = MetadataCatalog.get("socket_train")

    cpu_device = torch.device("cpu")

    predictor = DefaultPredictor(cfg)
    print(cfg)
    print(predictor)
    print("Predictor succesfully loaded.\n")

    dev, cap, depth_stream = camera.start_camera_and_wait()
    app.run(host="0.0.0.0", port="8777")
