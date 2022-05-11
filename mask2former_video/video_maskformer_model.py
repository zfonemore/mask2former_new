# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher, VideoTrackHungarianMatcher
from .modeling.qim import build as build_query_interaction_layer
from .utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        hidden_dim,
        qim,
        train_clip_len,
        eval_clip_len,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        # learnable query features
        self.track_query = nn.Embedding(num_queries, hidden_dim*2)
        #self.track_base = RuntimeTrackerBase()
        self.curr_objnum = 0
        self.scores_dict = {}
        self.masks_dict = {}
        self.qim = qim
        self.train_clip_len = train_clip_len
        self.eval_clip_len = eval_clip_len
        # learnable query p.e.
        #self.track_query_pos = nn.Embedding(num_queries, hidden_dim)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoTrackHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        # qim module
        d_model = 256
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        qim = build_query_interaction_layer('QIM', d_model, hidden_dim, d_model*2)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "qim": qim,
            "train_clip_len": cfg.INPUT.TRAIN_CLIP_LEN,
            "eval_clip_len": cfg.INPUT.EVAL_CLIP_LEN,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.track_query.weight.shape  # (100, 512)
        device = self.device
        track_instances.query_pos = self.track_query.weight
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        #track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        #track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        #track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        #track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        #track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.sem_seg_head.num_classes+1), dtype=torch.float, device=device)
        #track_instances.attn_mask = torch.zeros((len(track_instances), ), dtype=torch.float, device=device)

        return track_instances.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        batch_size = len(batched_inputs)
        for video in batched_inputs:
            self.num_frames = len(video["image"])
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        # generate empty track instances
        track_instances_list = []
        for i in range(batch_size):
            track_instances = self._generate_empty_tracks()
            track_instances_list.append(track_instances)

        if self.training:
            clip_len = self.train_clip_len

            # mask classification target
            targets = self.prepare_targets_clip(batched_inputs, images, clip_len)

            losses_all = {}
            is_last = False
            for frame in range(0, self.num_frames, clip_len):
                if frame+clip_len < self.num_frames:
                    indices = slice(frame, frame+clip_len, 1)
                else:
                    is_last = True
                    frame = max(0, self.num_frames-clip_len)
                    indices = slice(frame, self.num_frames, 1)

                features_perframe = {}
                for key in features.keys():
                    features_perframe[key] = features[key][indices]
                targets_perframe = [targets[frame // clip_len]]

                track_instance = track_instances_list[0]
                outputs = self.sem_seg_head(features_perframe, track_query=track_instance.query_pos, attn_mask=attn_mask if frame>0 else None)

                # bipartite matching-based loss
                losses, attn_mask = self.post_process(outputs, track_instances_list, targets_perframe, is_last=is_last)
                #losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        if k in losses_all:
                            losses_all[k] += losses[k] * self.criterion.weight_dict[k]
                        else:
                            losses_all[k] = losses[k] * self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
            for k in list(losses.keys()):
                losses_all[k] /= self.num_frames

            return losses_all
        else:
            query_eval = True
            clip_len = self.eval_clip_len

            if self.num_frames % clip_len == 0:
                self.eval_clip_nums = (self.num_frames // clip_len)
            else:
                self.eval_clip_nums = (self.num_frames // clip_len) + 1
            if clip_len > self.num_frames:
                clip_len = self.num_frames
            is_last = False
            for frame in range(0, self.num_frames, clip_len):
                if frame+clip_len < self.num_frames:
                    indices = slice(frame, frame+clip_len, 1)
                else:
                    is_last = True
                    frame = max(0, self.num_frames-clip_len)
                    indices = slice(frame, self.num_frames, 1)
                features_perframe = {}
                for key in features.keys():
                    features_perframe[key] = features[key][indices]

                input_per_image = batched_inputs[0]
                image_size = images.image_sizes[0]  # image size without padding after data augmentation
                height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
                width = input_per_image.get("width", image_size[1])

                if query_eval:
                    outputs = self.sem_seg_head(features_perframe, track_query=track_instances_list[0].query_pos, attn_mask=attn_mask if frame>0 else None)
                    attn_mask = self.post_process(outputs, track_instances_list, None, is_last=is_last, img_size=image_size, pad_size=images.tensor.shape, frame=frame)
                else:
                    outputs = self.sem_seg_head(features_perframe, track_query=None)#track_instances_list[0].query_pos)

                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]

                    mask_cls_result = mask_cls_results[0]
                    # upsample masks
                    mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                        mask_pred_results[0],
                        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )

                del outputs
            if query_eval:
                outputs = self.combine_to_video(height, width)
                self.scores_dict = {}
                self.masks_dict = {}
                return outputs
            else:
                return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)


    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for frame in range(self.num_frames):
            gt_instances_perframe = []
            for targets_per_video in targets:
                _num_instance = len(targets_per_video["instances"][0])
                #mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
                mask_shape = [_num_instance, 1, h_pad, w_pad]
                gt_masks = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

                targets_per_frame = targets_per_video["instances"][frame].to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids = targets_per_frame.gt_ids
                gt_masks[:, 0, :h, :w] = targets_per_frame.gt_masks.tensor

                valid_idx = (gt_ids!= -1)

                gt_classes = targets_per_frame.gt_classes[valid_idx]          # N,
                gt_ids = gt_ids[valid_idx].long()               # N, num_frames
                gt_masks = gt_masks[valid_idx].float()          # N, num_frames, H, W

                gt_instances_perframe.append({"labels": gt_classes, "ids": gt_ids, "masks": gt_masks})
            gt_instances.append(gt_instances_perframe)


        return gt_instances

    def prepare_targets_clip(self, targets, images, clip_len):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            for frame in range(0, self.num_frames, clip_len):
                mask_shape = [_num_instance, clip_len, h_pad, w_pad]
                gt_masks_per_clip = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

                gt_ids_per_clip = []
                for f_i in range(frame, frame+clip_len, 1):
                    targets_per_frame = targets_per_video["instances"][f_i]
                    targets_per_frame = targets_per_frame.to(self.device)
                    h, w = targets_per_frame.image_size

                    gt_ids_per_clip.append(targets_per_frame.gt_ids[:, None])
                    gt_masks_per_clip[:, f_i-frame, :h, :w] = targets_per_frame.gt_masks.tensor

                gt_ids_per_clip = torch.cat(gt_ids_per_clip, dim=1)
                valid_idx = (gt_ids_per_clip != -1).any(dim=-1)

                gt_classes_per_clip = targets_per_frame.gt_classes[valid_idx]          # N,
                gt_ids_per_clip = gt_ids_per_clip[valid_idx]                          # N, num_frames

                gt_instances.append({"labels": gt_classes_per_clip, "ids": gt_ids_per_clip})
                gt_masks_per_clip = gt_masks_per_clip[valid_idx].float()          # N, num_frames, H, W
                gt_instances[-1].update({"masks": gt_masks_per_clip})

        return gt_instances

    def post_process(self, output, track_instances_list, targets, is_last, img_size=None, pad_size=None, frame=0):
        with torch.no_grad():
            if not self.training:
                scores = F.softmax(output['pred_logits'][0], dim=-1)[:, :-1]
                labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(len(scores), 1).flatten(0, 1)
                scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
                score_thr = 0.05
                topk_indices = topk_indices[scores_per_image > score_thr]
                scores_per_image = scores_per_image[scores_per_image > score_thr]
                labels_per_image = labels[topk_indices]
                topk_indices = topk_indices // self.sem_seg_head.num_classes
                pred_masks = output["pred_masks"][0]
                '''
                print('scores_per_image:', scores_per_image)
                print('labels_per_image:', labels_per_image)
                print('topk:', topk_indices)
                import pdb
                pdb.set_trace()
                '''
                topk_indices = torch.unique(topk_indices)
                pred_masks = pred_masks[topk_indices]

                # upsample masks
                pred_masks = retry_if_cuda_oom(F.interpolate)(
                    pred_masks,
                    size=(pad_size[-2], pad_size[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
                pred_scores = scores[topk_indices]

        if self.training:
            for i, track_instances in enumerate(track_instances_list):
                track_instances.pred_logits = output['pred_logits'][i]
                track_instances.output_embedding = output['hs']
                #track_instances.attn_mask = output['attn_mask'].transpose(0,1)
            # the track id will be assigned by the mather.
            losses = self.criterion(output, targets, track_instances_list)
        else:
            track_instances_list[0].output_embedding = output['hs']
            if frame % self.eval_clip_len == 0:
                curr_clip = frame // self.eval_clip_len
            else:
                curr_clip = frame // self.eval_clip_len + 1
            for i, ind in enumerate(topk_indices):
                if track_instances_list[0][ind.item()].obj_idxes == -1:
                    track_instances_list[0][ind.item()].obj_idxes[0] = self.curr_objnum
                    self.scores_dict[self.curr_objnum] = pred_scores.new_zeros(self.eval_clip_nums, self.sem_seg_head.num_classes) #pred_scores[i]
                    self.masks_dict[self.curr_objnum] = pred_masks.new_zeros(self.num_frames, pred_masks.shape[-2], pred_masks.shape[-1]) #pred_masks[i]
                    self.scores_dict[self.curr_objnum][curr_clip] = pred_scores[i]
                    self.masks_dict[self.curr_objnum][frame:frame+self.eval_clip_len] = pred_masks[i]
                    self.curr_objnum += 1
                else:
                    obj_idx = track_instances_list[0][ind.item()].obj_idxes[0].item()
                    self.scores_dict[obj_idx][curr_clip] = pred_scores[i]
                    self.masks_dict[obj_idx][frame:frame+self.eval_clip_len] = pred_masks[i]
        '''
        print('frame:', frame)
        print('topk:', topk_indices)
        print('index:', track_instances_list[0].obj_idxes)
        print('scores:', scores_per_image)
        print('labels:', labels_per_image)
        '''
        if not is_last:
            for i, track_instances in enumerate(track_instances_list):
                init_track_instances = self._generate_empty_tracks()
                attn_mask = output['attn_mask'].transpose(0,1)
                attn_mask = attn_mask[(track_instances.obj_idxes>=0)]
                active_track_instances = self.qim(track_instances, init_track_instances)
                merged_track_instances = active_track_instances #Instances.cat([init_track_instances, active_track_instances])
                track_instances_list[i] = merged_track_instances
        else:
            for i in range(len(track_instances_list)):
                init_track_instances = self._generate_empty_tracks()
                track_instances_list[i] = init_track_instances
                attn_mask = None
        if self.training:
            return losses, attn_mask
        else:
            return attn_mask

    def combine_to_video(self, out_height, out_width):
        pred_scores = []
        pred_masks = []
        for key in self.scores_dict.keys():
            pred_scores.append(self.scores_dict[key].mean(dim=0).unsqueeze(0))
            pred_masks.append(self.masks_dict[key].unsqueeze(0))
        pred_scores = torch.cat(pred_scores)
        pred_masks = torch.cat(pred_masks)
        scores_per_image, topk_indices = pred_scores.flatten(0, 1).topk(10, sorted=False)
        labels_per_image = topk_indices % self.sem_seg_head.num_classes
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        pred_masks = pred_masks[topk_indices]
        pred_masks = F.interpolate(
            pred_masks, size=(out_height, out_width), mode="bilinear", align_corners=False
        )
        masks = pred_masks > 0.

        out_scores = scores_per_image.tolist()
        out_labels = labels_per_image.tolist()
        out_masks = [m for m in masks.cpu()]

        video_output = {
            "image_size": (out_height, out_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
