import argparse
import os
import json
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from predictor import VisualizationDemo

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/workspace/detectron2_repo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="input images name")
    parser.add_argument(
        "--output",
        default=None,
        help="output images name",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    input = args.input
    output = args.output
    
    debug = output is not None
    demo = VisualizationDemo(cfg, debug)
    # use PIL, to be consistent with evaluation
    img = read_image(input, format="BGR")
    predictions, visualized_output, obj = demo.run_on_image(img, debug)

    if output != None:
        visualized_output.save(output)
        print(output)
    else:
        print(json.dumps(obj))
    