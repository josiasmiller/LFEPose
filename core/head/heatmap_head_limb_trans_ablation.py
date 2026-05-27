# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from mmpose.models.heads.base_head import BaseHead
from projects.sheep_pose.core.head.get_limb_two_points import get_limb_two_points
from projects.sheep_pose.core.head.tokenbase import TokenPose_TB_base
from projects.sheep_pose.core.head.transformer_limb import LimbToKeypointTransformer

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class HeatmapHeadLimb_ablation(BaseHead):

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 limb_channels: int = 0,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 tokenpose_cfg = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.limb_channels = limb_channels
        self.tokenpose_cfg = tokenpose_cfg
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()


        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')
            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        if limb_channels != 0:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=limb_channels,
                kernel_size=1)
            self.limb_final_layer = build_conv_layer(cfg)
        else:
            self.limb_final_layer = nn.Identity()

        # my function

        # self.tokenpose_body = TokenPose_TB_base(feature_size=tokenpose_cfg.feature_size,
        #                                    patch_size=tokenpose_cfg.patch_size,
        #                                    num_keypoints=1,  # body mask
        #                                    dim=tokenpose_cfg.dim,
        #                                    depth=tokenpose_cfg.depth,
        #                                    heads=tokenpose_cfg.heads,
        #                                    mlp_ratio=tokenpose_cfg.mlp_ratio,
        #                                    heatmap_size=tokenpose_cfg.heatmap_size,
        #                                    channels=in_channels,
        #                                    pos_embedding_type=tokenpose_cfg.pos_embedding_type,
        #                                    apply_init=tokenpose_cfg.apply_init)
        self.tokenpose_point = TokenPose_TB_base(feature_size=tokenpose_cfg.feature_size,
                                           patch_size=tokenpose_cfg.patch_size,
                                           num_keypoints=self.out_channels,
                                           dim=tokenpose_cfg.dim,
                                           depth=tokenpose_cfg.depth,
                                           heads=tokenpose_cfg.heads,
                                           mlp_ratio=tokenpose_cfg.mlp_ratio,
                                           heatmap_size=tokenpose_cfg.heatmap_size,
                                           channels=in_channels,
                                           pos_embedding_type=tokenpose_cfg.pos_embedding_type,
                                           apply_init=tokenpose_cfg.apply_init)
        self.tokenpose_limb = TokenPose_TB_base(feature_size=tokenpose_cfg.feature_size,
                                           patch_size=tokenpose_cfg.patch_size,
                                           num_keypoints=self.out_channels,
                                           dim=tokenpose_cfg.dim,
                                           depth=tokenpose_cfg.depth,
                                           heads=tokenpose_cfg.heads,
                                           mlp_ratio=tokenpose_cfg.mlp_ratio,
                                           heatmap_size=tokenpose_cfg.heatmap_size,
                                           channels=in_channels,
                                           pos_embedding_type=tokenpose_cfg.pos_embedding_type,
                                           apply_init=tokenpose_cfg.apply_init)

        # self.limb_mask_generator = LimbMaskGenerator(self.limb_channels)
        self.CMAB = CBAM(in_planes=in_channels, ratio=4)


        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)




    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = feats[-1] # hrner=[16, 32, 64, 64], vit=[16, 768, 16, 16], cspnext=[16, 512, 8, 8]
        B = x.size(0)


        if x.size(1) == 512 or x.size(1) == 768 or x.size(1) == 1024:
            x = self.deconv_layers(x)

        # print("x.size()", x.size())

        ##
        # point
        ##
        # x_point_dict = self.tokenpose_point(x) # x or x_body

        x_point_dict = self.tokenpose_point(x)
        x_point_final = x_point_dict.pred  # [16, 17, 64, 64]

        # x_point_final =  self.final_layer(x)

        return x_point_final

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:


        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)

            flip = self.forward(_feats_flip)
            _batch_heatmaps_flip = flip_heatmaps(
                flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            _point = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            _point = self.forward(feats)

        preds = self.decode(_point)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in _limb.detach()
            ]

            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:

        pred_fields = self.forward(feats)

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        limb_gt_heatmaps = torch.stack([d.gt_fields.limb_heatmaps for d in batch_data_samples])

        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        limb_weights = torch.cat([d.gt_instance_labels.limb_weights for d in batch_data_samples])
        # print("keypoint_weights",keypoint_weights)


        # calculate losses
        losses = dict()
        # pred_fields [16, 17, 64, 64]
        # gt_heatmaps [16, 17, 64, 64]
        # keypoint_weights [16, 17]
        loss_kpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
        # loss_limb = self.loss_module(pred_fields_limb, limb_gt_heatmaps, limb_weights)

        # losses.update(loss_kpt=loss_kpt, loss_limb=loss_limb)
        losses.update(loss_kpt=loss_kpt)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v


class LimbMaskGenerator(nn.Module):
    def __init__(self, in_channels: int = 8):
        super().__init__()
        # 通道注意力：把 8 维通道压缩再压回 8 维
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),  # 8->8
            nn.Sigmoid()
        )
        # 空间注意力：8 维输入 -> 1 维空间热图
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,8,H,W]
        att = self.ca(x)          # [B,8,1,1] 通道权重
        x = x * att               # 通道加权
        mask = self.sa(x)         # [B,1,H,W] 空间热图
        return mask               # 直接返回 [B,1,H,W]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x  # 应用通道注意力
        x = self.sa(x) * x  # 应用空间注意力
        return x