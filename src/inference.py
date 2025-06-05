import os

# Root directory of the project
ROOT_DIR = os.path.abspath("")
CLASS_LABEL = "pen"
TRAIN_ERROR = "Sorry, train is not supporeted yet. Please come back later"

# import some common libraries
import numpy as np
import os, json, cv2, random, tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from predictor import VisualizationDemo

WINDOW_NAME = "COCO detections"

############################################################
#  Dataset
############################################################

def get_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_export_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = v["regions"]
        objs = []
        for anno in annos:
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("{}_".format(CLASS_LABEL) + d, lambda d=d: get_dicts("../datasets/" +  d))
    MetadataCatalog.get("{}_".format(CLASS_LABEL) + d).set(thing_classes=[CLASS_LABEL])
socket_metadata = MetadataCatalog.get("pen_train")

############################################################
#  Inference
############################################################

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
#cfg.MODEL.DEVICE = "cpu"

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if (cap.isOpened()):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for vis in tqdm.tqdm(VisualizationDemo(cfg).run_on_video(cap)):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    vis_scaled = cv2.resize(vis, (1080, 720))
    cv2.imshow(WINDOW_NAME, vis_scaled)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cap.release()
cv2.destroyAllWindows()