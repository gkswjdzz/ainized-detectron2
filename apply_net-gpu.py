# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.extractor import CompoundExtractor, create_extractor


opts = []
config_fpath='/workspace/detectron2_repo/configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml'
model_fpath='/workspace/detectron2_repo/densepose_rcnn_R_50_FPN_s1x.pkl'

def setup_config():
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()
    return cfg

cfg = setup_config()

class Action(object):
    pass

class InferenceAction(Action):
    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        predictor = DefaultPredictor(cfg)
        file_list = [args.input]
        context = cls.create_context(args)
        for file_name in file_list:
            img = file_name
            # img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                out_binary = cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
        cls.postexecute(context)
        return out_binary

class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        # out_dir = os.path.dirname(out_fname)
        # if len(out_dir) > 0 and not os.path.exists(out_dir):
        #    os.makedirs(out_dir)
        # cv2.imwrite(out_fname, image_vis)
        context["entry_idx"] += 1
        return cv2.imencode('.jpg', image_vis)[1]
    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        #return base + ".{0:04d}".format(entry_idx) + ext
        return base + ext
    @classmethod
    def create_context(cls: type, args: argparse.Namespace) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            vis = cls.VISUALIZERS[vis_spec]()
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context

def main(file_in_memory):
    args = argparse.Namespace()
    args.func=ShowAction.execute
    args.input=file_in_memory
    args.min_score=0.8
    args.nms_thresh=None
    args.output='outputres2.png'
    args.verbosity=None
    args.visualizations='dp_contour,bbox'
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    out_binary_buffer = args.func(args)
    return out_binary_buffer
