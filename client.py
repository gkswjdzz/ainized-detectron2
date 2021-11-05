import multiprocessing as mp
import numpy as np
import cv2
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from flask import Flask, request, Response, render_template, jsonify, send_file
import io

app = Flask(__name__, static_url_path='/static')

def convert_PIL_to_numpy(image, format):
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

def read_image(file, format=None):
    image = Image.open(file)
    return convert_PIL_to_numpy(image, format)

@app.route('/health')
def health():
    return "ok"

# @app.route('/')
# def main():
#     return render_template('index.html')

panoptic_cfg = get_cfg()
panoptic_cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
panoptic_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
panopticPredictor = DefaultPredictor(panoptic_cfg)



instance_cfg = get_cfg()
instance_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
instance_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
instance_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
instancePredictor = DefaultPredictor(instance_cfg)


keypoint_cfg = get_cfg()
keypoint_cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
keypoint_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
keypoint_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
keypointPredictor = DefaultPredictor(keypoint_cfg)

@app.route('/<path>', methods=['POST'])
def predict(path):
    try:
        input_file = request.files['file']

        if input_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
            return jsonify({'message': 'Only support jpeg, jpg or png'}), 400

        if input_file.content_type is 'image/png': 
            np = Image.open(input_file).convert('RGB')
        else:
            np = read_image(input_file)
            
        if path == 'keypoint':
            cfg = keypoint_cfg
            predictions = keypointPredictor(np)
        elif path == 'instancesegmentation':
            cfg = instance_cfg
            predictions = instancePredictor(np)
        elif path == 'panopticsegmentation':
            cfg = panoptic_cfg
            predictions = panopticPredictor(np)
        else:
            return jsonify({'message': 'path is not vaild'}), 400
        
        visualizer = Visualizer(np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.6)

        instances = predictions["instances"].to('cpu')
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        cv2.imwrite('abc.jpg',vis_output.get_image()[:, :, ::-1])
        result_image = Image.fromarray(vis_output.get_image()[:, :, ::-1])
        result = io.BytesIO()
        
        result_image.save(result, 'JPEG', quality=95)
        result.seek(0)

        return send_file(result, mimetype='image/jpeg')

    except Exception as e:
        print(e)
        return jsonify({'message': 'Server error'}), 500

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app.run(host="0.0.0.0", port=80)