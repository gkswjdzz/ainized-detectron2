from flask import Flask, jsonify, request, send_file
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from predictor import VisualizationDemo
import numpy as np
import cv2
import io
import requests
from queue import Empty, Queue
import threading
import time 
import json
import subprocess 
from apply_net import main as apply_net_main

requests_queue = Queue()

app = Flask(__name__)

BATCH_SIZE=1
CHECK_INTERVAL=0.1

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except TypeError as e:
    return False
  return True

def handle_requests_by_batch():
  while True:
    requests_batch = []
    while not (
      len(requests_batch) >= BATCH_SIZE # or
      #(len(requests_batch) > 0 #and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
    ):
      try:
        requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
      except Empty:
        continue
    batch_outputs = []
    for request in requests_batch:
      batch_outputs.append(run(request['input'][0], request['input'][1]))

    for request, output in zip(requests_batch, batch_outputs):
      request['output'] = output

threading.Thread(target=handle_requests_by_batch).start()

def track_event(category, action, label=None, value=0):
  data = {
    'v': '1',  # API Version.
    'tid': 'UA-164242824-8',  # Tracking ID / Property ID.
    # Anonymous Client Identifier. Ideally, this should be a UUID that
    # is associated with particular user, device, or browser instance.
    'cid': '555',
    't': 'event',  # Event hit type.
    'ec': category,  # Event category.
    'ea': action,  # Event action.
    'el': label,  # Event label.
    'ev': value,  # Event value, must be an integer
    'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'
  }

  response = requests.post(
    'https://www.google-analytics.com/collect', data=data)

  # If the request fails, this will raise a RequestException. Depending
  # on your application's needs, this may be a non-error and can be caught
  # by the caller.
  response.raise_for_status()

def setup_cfg(config_file, confidence_threshold = 0.5, is_gpu = False):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.DEVICE = 'cpu' if is_gpu == False else 'cuda'
    print(cfg.MODEL.DEVICE)
    cfg.freeze()
    return cfg

def run(input_file_in_memory, method):
  print(input_file_in_memory.shape)
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
    return io_buf
  else :
    return {'message': 'invalid parameter'}

  cfg = setup_cfg(config_file=config_file, is_gpu=True)
  debug = False if method == 'predictions' else True
  demo = VisualizationDemo(cfg, debug=debug)
  predictions, visualized_output, obj = demo.run_on_image(input_file_in_memory, debug)
  
  if debug :
    np_img = visualized_output.get_image()
    output_file_in_memory = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    is_success, buffer = cv2.imencode(".jpg", output_file_in_memory)
    io_buf = io.BytesIO(buffer)
    
    return io_buf
  else : 
    return obj
  
@app.route('/health')
def health():
  return "ok"

@app.route('/<method>', methods=['POST'])
def run_python(method):
  track_event(category='api_gpu', action=f'/${method}')
  print(requests_queue.qsize())
  if requests_queue.qsize() >= 1:
    return 'Too Many Requests', 429
  filestr = request.files['file'].read()
  preview_mode = request.form.get('preview')
  preview_mode = True if preview_mode is not None and preview_mode == 'true' else False

  npimg = np.fromstring(filestr, np.uint8)
  input_file_in_memory = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

  if input_file_in_memory is None :
    return jsonify({'message': 'invalid file'}), 400

  req = { 
    'input': [input_file_in_memory, method]
  }

  requests_queue.put(req)
  
  while 'output' not in req:
    time.sleep(CHECK_INTERVAL)
  
  ret = req['output']
  if type(ret) is dict:
    if preview_mode:
      dump = json.dumps(ret)
      if len(dump) > 1000 :
        return dump[0:1000], 200
    return jsonify(ret), 200
  if is_json(ret) :
    return jsonify(ret), 400
  else :
    return send_file(ret, mimetype='image/jpeg'), 200
  
if __name__ == "__main__":
  app.run(debug=False, port=80, host='0.0.0.0')  