# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from mmpose.codecs.base import BaseKeypointCodec
from mmpose.codecs.utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
from mmpose.codecs.utils.post_processing import get_heatmap_maximum
from mmpose.codecs.utils.refinement import refine_keypoints, refine_keypoints_dark


@KEYPOINT_CODECS.register_module()
class MSRAHeatmapLimb(BaseKeypointCodec):
    """Represent keypoints as heatmaps via "MSRA" approach. See the paper:
    `Simple Baselines for Human Pose Estimation and Tracking`_ by Xiao et al
    (2018) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The generated heatmap in shape (K, H, W)
            where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        unbiased (bool): Whether use unbiased method (DarkPose) in ``'msra'``
            encoding. See `Dark Pose`_ for details. Defaults to ``False``
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. The kernel size and sigma should follow
            the expirical formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`.
            Defaults to 11

    .. _`Simple Baselines for Human Pose Estimation and Tracking`:
        https://arxiv.org/abs/1804.06208
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.unbiased = unbiased

        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        # keypoints (1, 17, 2)
        # keypoints_visible (1, 17)

        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # self.unbiased = false
        # heatmaps (17, 64, 64)
        # keypoint_weights (1, 17) 基于keypoints_visible,将越界的设置为0
        if self.unbiased:
            heatmaps, keypoint_weights = generate_unbiased_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            heatmaps, keypoint_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

        # print("keypoints",keypoints)
        # print("keypoints_visible",keypoints_visible)

        # limb_indices = np.array([
        #     [0, 2], [1, 2],  # 头部
        #     [2, 3], [3, 4],  # 躯干
        #     [3, 5], [5, 6], [6, 7],  # 左前肢
        #     [3, 8], [8, 9], [9, 10],  # 右前肢
        #     [4, 11], [11, 12], [12, 13],  # 左后肢
        #     [4, 14], [14, 15], [15, 16]  # 右后肢
        # ])

        # limb_heatmaps (8, 64, 64)
        # limb_weights (1, 8)
        limb_heatmaps, limb_weights = generate_gaussian_limb_heatmaps(
            heatmap_size=(64, 64),
            keypoints=keypoints / self.scale_factor,
            keypoints_visible=keypoints_visible,
            limb_indices=np.array([[5, 6], [6, 7], [8, 9], [9, 10],
                                   [11, 12], [12, 13], [14, 15], [15, 16]]),
            # limb_indices=limb_indices,
            sigma=1.0, # 2.0
            limb_width=1.0, # 1
        )

        # 保存查看limb热图的效果
        # sum = np.sum(limb_heatmaps)
        # print("\nsum", sum)
        # if np.any(keypoints < 0) == False:
        #     np.save('limb_heatmaps.npy', limb_heatmaps)
        #     return None

        # np.save("heatmap_limb_origin_0812.npy", limb_heatmaps)  # 保存为 limb_images.npy


        encoded = dict(heatmaps=heatmaps, keypoint_weights=keypoint_weights,
                       limb_heatmaps=limb_heatmaps, limb_weights=limb_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, heatmaps)
        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores


import numpy as np
from typing import Tuple, Union

def generate_gaussian_limb_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    limb_indices: np.ndarray,
    sigma: Union[float, Tuple[float], np.ndarray],
    limb_width: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate gaussian heatmaps for limbs (lines between keypoints).

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, 2)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape (N, K)
        limb_indices (np.ndarray): Limb connections in shape (L, 2) where each row
            contains the indices of two connected keypoints
        sigma (float or List[float]): Sigma values of the Gaussian heatmap.
            If single float, same sigma is used for all limbs.
        limb_width (float): Width factor to scale the sigma along the limb

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape (L, H, W)
        - limb_weights (np.ndarray): The target weights in shape (N, L)
    """
    N, K, _ = keypoints.shape
    L = limb_indices.shape[0]
    W, H = heatmap_size

    heatmaps = np.zeros((L, H, W), dtype=np.float32)
    limb_weights = np.zeros((N, L), dtype=np.float32)

    if isinstance(sigma, (int, float)):
        sigma = [sigma] * L

    # Create coordinate grids
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    xy_grid = np.stack([x_grid, y_grid], axis=-1)  # shape (H, W, 2)

    for n in range(N):
        for l, (k1, k2) in enumerate(limb_indices):
            # Skip if either keypoint is not visible
            # if keypoints_visible[n, k1] < 0.5 or keypoints_visible[n, k2] < 0.5:
            #     continue

            # Get the two keypoints
            p1 = keypoints[n, k1]  # (x1, y1)
            p2 = keypoints[n, k2]  # (x2, y2)

            # Set limb weight to 1
            limb_weights[n, l] = 1.0

            # Vector from p1 to p2
            v = p2 - p1
            length = np.linalg.norm(v)

            # Skip if the points are too close
            if length < 1e-6:
                continue

            # Unit vector
            v = v / length

            # Compute distance from each point to the line segment
            # Projection of (xy - p1) onto v
            xy_p1 = xy_grid - p1
            proj = np.dot(xy_p1, v)

            # Distance along the line (clamped to [0, length])
            dist_along = np.clip(proj, 0, length)

            # Closest points on the line segment
            closest = p1 + dist_along[..., None] * v

            # Perpendicular distance (distance to line)
            dist_perp = np.linalg.norm(xy_grid - closest, axis=-1)

            # Along-line distance (scaled Gaussian)
            sigma_width = sigma[l] * limb_width
            gaussian_along = np.exp(-(dist_along - proj) ** 2 / (2 * sigma[l] ** 2))

            # Perpendicular distance (main Gaussian)
            gaussian_perp = np.exp(-(dist_perp ** 2) / (2 * sigma_width ** 2))

            # Combined Gaussian
            gaussian = gaussian_along * gaussian_perp

            # Update heatmap (element-wise maximum)
            heatmaps[l] = np.maximum(heatmaps[l], gaussian)

    return heatmaps, limb_weights