import streamlit as st
import numpy as np
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

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

# @app.route('/health')
# def health():
    # return "ok"

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

def predict(path, np):
        if path == 'keypoint':
            cfg = keypoint_cfg
            predictions = keypointPredictor(np)["instances"]
            visualizer = Visualizer(np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            instances = predictions.to('cpu')
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
       
        elif path == 'instancesegmentation':
            cfg = instance_cfg
            predictions = instancePredictor(np)["instances"]
            visualizer = Visualizer(np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            instances = predictions.to('cpu')
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
       
        elif path == 'panopticsegmentation':
            cfg = panoptic_cfg
            panoptic_seg, segments_info = panopticPredictor(np)["panoptic_seg"]
            visualizer = Visualizer(np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        else:
            return 'nothing'

        result_image = vis_output.get_image()[:, :, ::-1]
        return result_image

FAVICON_URL = "https://cloud.kt.com/favicon.ico"

st.set_page_config(
    page_title="사진을 넣어 물체를 인식해보세요", page_icon=FAVICON_URL,
)

st.title("사진을 넣어 물체를 인식해보세요!")

st.subheader("사진을 넣고 다양한 모델을 이용하여 사진의 물체들을 인식해보세요.")

model = st.selectbox('모델 선택', list(['instancesegmentation', 'panopticsegmentation', 'keypoint']))

input_file = st.file_uploader("파일을 넣어주세요.")
if input_file is not None:
    input_file = read_image(input_file)
    st.write('입력한 사진')
    st.image(input_file)
    st.write('결과물')
    input_file = predict(model, input_file)
    st.image(input_file)
