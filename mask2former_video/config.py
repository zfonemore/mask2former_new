# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 6
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
    cfg.INPUT.TRAIN_CLIP_LEN = 2
    cfg.INPUT.EVAL_CLIP_LEN = 5
