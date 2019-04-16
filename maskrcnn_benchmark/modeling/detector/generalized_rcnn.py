# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.num=0

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        # self.view(images,proposals,targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

    def view(self,images,proposals,targets):
        import cv2
        import numpy as np
        image=images.tensors.cpu().numpy()[0]
        image=np.transpose(image,(1,2,0))
        if self.training:
            labels=targets[0].bbox.cpu().numpy()
        predictions=proposals[0].bbox.cpu().numpy()
        print('proposals:',proposals)
        print('predictions:',predictions)


        pic0_name = '/home/mvl/Documents/maskrcnn-benchmark/mid_output/{}_0.png'.format(str(self.num))
        pic1_name = '/home/mvl/Documents/maskrcnn-benchmark/mid_output/{}_1.png'.format(str(self.num))
        pic2_name = '/home/mvl/Documents/maskrcnn-benchmark/mid_output/{}_2.png'.format(str(self.num))
        print('num:',self.num)
        self.num+=1
        cv2.imwrite(pic0_name, image)
        if self.training:
            image=self.label_visible(labels,image,color=1,mode='xyxy')
            cv2.imwrite(pic1_name, image)
        image=self.label_visible(predictions,image,color=0,mode='xyxy')
        cv2.imwrite(pic2_name, image)
        # cv2.imshow('pic', image)
        # cv2.waitKey(0)

    def label_visible(self, labels, image, color=1, mode='xyxy'):
        import cv2
        # color==1 green -> GT
        # color==0  red  -> prediction
        image_copy=image.copy()
        if not len(labels) == 0:
            for label in labels:
                if mode == 'xyxy':
                    if color == 1:
                        image_copy = cv2.rectangle(image_copy, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])),
                                              (0, 225, 0), 1)
                        # cv2.putText(image, 'cat:{}'.format(label[4]), (label[0], label[1]), cv2.FONT_HERSHEY_COMPLEX, 5,
                        #             (0, 255, 0), 2)
                    if color == 0:
                        image_copy = cv2.rectangle(image_copy, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])),
                                              (0, 0, 225), 1)

                elif mode == 'xywh':
                    if color == 1:
                        image_copy = cv2.rectangle(image_copy, (int(label[0]), int(label[1])),
                                              (int(label[2] + label[0]), int(label[3] + label[1])), (0, 255, 0), 1)
                        # cv2.putText(image, 'cat:{}'.format(label[4]),
                        #             (int(label[0])-1, int(label[1])-1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    if color == 0:
                        image_copy = cv2.rectangle(image_copy, (int(label[0]), int(label[1])),
                                              (int(label[2] + label[0]), int(label[3] + label[1])), (0, 0, 225), 1)
                        # cv2.putText(image, 'cat:{} score:{}'.format(label[4],round(label[5],2)),
                        #             (int(label[0]) - 1, int(label[1]) - 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        return image_copy