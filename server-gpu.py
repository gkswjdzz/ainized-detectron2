from flask import Flask, jsonify, request, send_file
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from predictor import VisualizationDemo
import numpy as np
import cv2
import io

import subprocess 
from apply_net import main as apply_net_main

app = Flask(__name__)

def setup_cfg(config_file, confidence_threshold = 0.5, is_gpu = False):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.DEVICE = 'cpu' if is_gpu == False else 'cuda'
    cfg.freeze()
    return cfg

@app.route('/health')
def health():
  return "ok"

@app.route('/<method>', methods=['POST'])
def run_python(method):
  filestr = request.files['file'].read()
  npimg = np.fromstring(filestr, np.uint8)
  input_file_in_memory = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

  if input_file_in_memory is None :
    return jsonify({'message': 'invalid file'}), 400
  if input_file_in_memory.shape[2] == 4 :
    input_file_in_memory = input_file_in_memory[:,:,0:-1]
    
  if method == 'instancesegmentation' or method == 'predictions' :
    config_file = '/workspace/detectron2_repo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml'  
  elif method == 'panopticsegmentation' :
    config_file = '/workspace/detectron2_repo/configs/quick_schedules/panoptic_fpn_R_50_inference_acc_test.yaml'
  elif method == 'keypoint' :
    config_file = '/workspace/detectron2_repo/configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml'
  elif method == 'densepose' :
    io_buf = None
    io_buf = io.BytesIO(apply_net_main(input_file_in_memory))
    return send_file(io_buf, mimetype='image/jpeg')
  else :
    return jsonify({'message': 'invalid parameter'}), 400

  cfg = setup_cfg(config_file=config_file, is_gpu=True)
  debug = False if method == 'predictions' else True
  demo = VisualizationDemo(cfg, debug=debug)
  predictions, visualized_output, obj = demo.run_on_image(input_file_in_memory, debug)
  
  if debug :
    np_img = visualized_output.get_image()
    output_file_in_memory = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    is_success, buffer = cv2.imencode(".jpg", output_file_in_memory)
    io_buf = io.BytesIO(buffer)
    
    return send_file(io_buf, mimetype='image/jpeg')
  else : 
    return jsonify(obj), 200
  
if __name__ == "__main__":
  app.run(debug=False, port=80, host='0.0.0.0', threaded=False)  