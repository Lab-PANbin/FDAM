"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomVerticalFlip = register()(T.RandomVerticalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt


@register()
class TargetAwareRandomCrop(T.Transform):
    """
    Random crop for large images with target-aware strategy.
    Only keeps bounding boxes whose center points are within the cropped region.

    Args:
        size: (height, width) tuple for crop size, e.g., (1024, 1024)
        min_targets: minimum number of targets required in the crop (default: 1)
                     If 0, allows empty crops (pure background)
    """
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, size, min_targets=1):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size) #(height, width)
        self.min_targets = min_targets

    def _get_params(self, flat_inputs: List[Any], target_boxes: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate random crop parameters.
        If target_boxes is provided and min_targets > 0, try to crop around targets.
        """
        img = flat_inputs[0]
        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        else:
            h, w = F.get_spatial_size(img)

        th, tw = self.size

        # If image is smaller than crop size, no crop needed
        if h <= th and w <= tw:
            return {'top': 0, 'left': 0, 'height': h, 'width': w, 'needs_crop': False}

        # If we want target-aware cropping and have boxes
        if self.min_targets > 0 and target_boxes is not None and len(target_boxes) > 0:
            # Try to crop around a random target
            # Boxes are in xyxy format
            random_idx = torch.randint(0, len(target_boxes), (1,)).item()
            box = target_boxes[random_idx]

            # Get box center
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            # Try to center the crop around this box
            # But add some randomness
            left = int(cx - tw / 2 + torch.randint(-tw // 4, tw // 4, (1,)).item())
            top = int(cy - th / 2 + torch.randint(-th // 4, th // 4, (1,)).item())

            # Clamp to valid range
            left = max(0, min(left, w - tw))
            top = max(0, min(top, h - th))
        else:
            # Pure random crop
            top = torch.randint(0, h - th + 1, (1,)).item() if h > th else 0
            left = torch.randint(0, w - tw + 1, (1,)).item() if w > tw else 0

        return {'top': top, 'left': left, 'height': th, 'width': tw, 'needs_crop': True}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params['needs_crop']:
            return inpt
        return F.crop(inpt, params['top'], params['left'], params['height'], params['width'])

    def forward(self, *inputs: Any) -> Any:
        """
        Custom forward to handle bounding boxes specially.
        """
        if not inputs:
            return inputs

        # Separate image and target
        img = inputs[0]
        target = inputs[1] if len(inputs) > 1 else None

        # Get boxes if available (for target-aware cropping)
        target_boxes = None
        if target is not None and isinstance(target, dict) and 'boxes' in target:
            boxes = target['boxes']
            if isinstance(boxes, BoundingBoxes):
                target_boxes = boxes.data
            elif isinstance(boxes, torch.Tensor):
                target_boxes = boxes

        # Get crop parameters
        flat_inputs = [img]
        params = self._get_params(flat_inputs, target_boxes)

        if not params['needs_crop']:
            return inputs if len(inputs) > 1 else inputs[0]

        # Crop image
        img_cropped = self._transform(img, params)

        # If no target, return cropped image only
        if target is None or not isinstance(target, dict):
            return (img_cropped,) + inputs[1:] if len(inputs) > 1 else img_cropped

        # Process target (boxes, labels, etc.)
        target_cropped = target.copy()

        if 'boxes' in target:
            boxes = target['boxes']
            if isinstance(boxes, BoundingBoxes):
                boxes_data = boxes.data.clone()
            else:
                boxes_data = boxes.clone()

            # Shift boxes by crop offset
            top, left = params['top'], params['left']
            boxes_data[:, [0, 2]] -= left  # x coordinates
            boxes_data[:, [1, 3]] -= top   # y coordinates

            # Calculate box centers
            cx = (boxes_data[:, 0] + boxes_data[:, 2]) / 2
            cy = (boxes_data[:, 1] + boxes_data[:, 3]) / 2

            # Keep only boxes whose centers are inside the crop
            tw, th = params['width'], params['height']
            keep = (cx >= 0) & (cx < tw) & (cy >= 0) & (cy < th)

            # Clamp boxes to crop boundaries
            boxes_data[:, 0].clamp_(min=0, max=tw)
            boxes_data[:, 1].clamp_(min=0, max=th)
            boxes_data[:, 2].clamp_(min=0, max=tw)
            boxes_data[:, 3].clamp_(min=0, max=th)

            # Filter boxes
            boxes_data = boxes_data[keep]

            # Update target
            if isinstance(boxes, BoundingBoxes):
                target_cropped['boxes'] = convert_to_tv_tensor(
                    boxes_data,
                    key='boxes',
                    spatial_size=(th, tw)
                )
            else:
                target_cropped['boxes'] = boxes_data

            # Filter other fields
            for key in ['labels', 'area', 'iscrowd']:
                if key in target:
                    target_cropped[key] = target[key][keep]

            # Store number of kept targets
            target_cropped['num_targets'] = len(boxes_data)

        # Update orig_size if present
        if 'orig_size' in target_cropped:
            target_cropped['orig_size'] = torch.tensor([params['width'], params['height']])

        return img_cropped, target_cropped
