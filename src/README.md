## Python scripts

* camera.py - used for detection-server.py   starts the camera and takes the images for the detection.
* camera_standalone.py - starts the depth-/ and rgb-camera and when clicked, displays the coords of a point.
* detection_server.py - the main script to start the detection. starts a webserver to display the images.
* inference.py - script to start a standalone inference to test the trained model.
* predictor.py - used by inference to live predict the images of a webcam
* socket_pose_estimation.py - used by detection_server. Implements the logic to calculate the pose (TODO)
* train.py - script to train a model with new images.

### Training
1. Use labelme to label new images and save them as datasets/train/*.jpg and datasets/train/*.json
2. Use [this script](../datasets/convert_individual_json_to_detectron_json.py) to generate a single json file from all individual files.
3. If necessary, adjust used model in train.py and the computing DEVICE (cpu/cuda)
4. run train.py
