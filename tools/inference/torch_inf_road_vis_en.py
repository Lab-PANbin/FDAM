"""
---------------------------------------------------------------------------------
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import json
import hashlib
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# English labels
DET_LABEL_MAP = {
    0: 'LC', 1: 'TC', 2: 'LR', 3: 'SR'
}

GRID_LABEL_MAP = {
    0: 'LC', 1: 'TC', 2: 'LR', 3: 'SR'
}


def load_chinese_font(fontsize):
    """Load the optional local font or a sized Pillow fallback."""
    font_path = "font/SimHei.ttf"
    try:
        font = ImageFont.truetype(font_path, fontsize)
        return font
    except (IOError, OSError):
        return ImageFont.load_default(size=fontsize)

# Grid class -> detection classes (1:1 mapping)
GRID2DET_MAP = {
    0: (0,),  # LC grid -> LC det
    1: (1,),  # TC grid -> TC det
    2: (2,),  # LR grid -> LR det
    3: (3,),  # SR grid -> SR det
}

# RGB equivalents of the reference detector's BGR detection palette.
# Both PIL and the NumPy buffers below retain RGB channel order.
DET_COLORS = [
    (255, 0, 0),      # LC: red
    (255, 0, 255),    # TC: magenta
    (255, 97, 3),     # LR: orange
    (0, 0, 255),      # SR: blue
]

# Use the same category colors for detection and grid localization.
GRID_COLORS = list(DET_COLORS)

DET_COLOR_MAP = {i: DET_COLORS[i % len(DET_COLORS)] for i in DET_LABEL_MAP}
GRID_COLOR_MAP = {i: GRID_COLORS[i % len(GRID_COLORS)] for i in GRID_LABEL_MAP}


def configure_class_maps(categories, num_classes=4):
    """Use training category IDs, never the order of the categories list."""
    global DET_LABEL_MAP, GRID_LABEL_MAP, DET_COLOR_MAP, GRID_COLOR_MAP, GRID2DET_MAP
    aliases = {'纵向裂缝': 'LC', '横向裂缝': 'TC', '修补.块状': 'LR', '修补.条状': 'SR'}
    names = {int(c['id']): aliases.get(c['name'].strip(), c['name'].strip().upper())
             for c in categories}
    det = {i: name for i, name in names.items() if i >= 0}
    if set(det) != set(range(num_classes)):
        raise ValueError(f"Annotation detection IDs must be 0..{num_classes - 1}; got {det}. "
                         "Use the annotation file used to train this checkpoint.")
    grid = {-i - 1: name for i, name in names.items() if i < 0}
    if grid and set(grid) != set(range(num_classes)):
        raise ValueError(f"Grid annotation IDs must be -1..-{num_classes}; got {grid}.")
    DET_LABEL_MAP = det
    GRID_LABEL_MAP = grid or dict(det)
    colors = {'LC': (255, 0, 0), 'TC': (255, 0, 255), 'LR': (255, 97, 3), 'SR': (0, 0, 255)}
    DET_COLOR_MAP = {i: colors.get(name, (255, 255, 255)) for i, name in DET_LABEL_MAP.items()}
    GRID_COLOR_MAP = {i: colors.get(name, (255, 255, 255)) for i, name in GRID_LABEL_MAP.items()}
    GRID2DET_MAP = {g: tuple(d for d, name in DET_LABEL_MAP.items() if name == grid_name)
                    for g, grid_name in GRID_LABEL_MAP.items()}
    print(f"Detection category IDs: {DET_LABEL_MAP}; grid channels: {GRID_LABEL_MAP}")


def box_ioa(box1, box2):
    """
    Compute IoA (Intersection over Area) of box1 with respect to box2.
    box1: (N, 4) [x1, y1, x2, y2]
    box2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) IoA matrix
    """
    # Intersection
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Area of box1
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])

    return inter_area / (area1[:, None] + 1e-7)


def grid2box(grid_logits, grid_thrh, orig_size, infer_size, edge_filter=0.0):
    """Convert grid predictions to box format."""
    C, H, W = grid_logits.shape
    device = grid_logits.device

    # Convert to probabilities
    grid_probs = torch.sigmoid(grid_logits)

    # Get max confidence and class for each position
    conf, cls = grid_probs.max(dim=0)  # (H, W)

    # Match the unified Road evaluator, including scores exactly at the threshold.
    mask = conf >= grid_thrh

    # Apply edge filter if specified
    if edge_filter > 0:
        edge_w = int(W * edge_filter)
        edge_h = int(H * edge_filter)
        # Create edge mask: True for non-edge regions
        edge_mask = torch.ones((H, W), dtype=torch.bool, device=device)
        if edge_h > 0:
            edge_mask[:edge_h, :] = False
            edge_mask[-edge_h:, :] = False
        if edge_w > 0:
            edge_mask[:, :edge_w] = False
            edge_mask[:, -edge_w:] = False
        mask = mask & edge_mask

    y_indices, x_indices = mask.nonzero(as_tuple=True)

    if len(y_indices) == 0:
        return torch.zeros((0, 6), device=device)

    orig_w, orig_h = orig_size[0], orig_size[1]
    infer_w, infer_h = infer_size[0], infer_size[1]

    # Grid cell size is stride (infer_size / grid_size)
    stride_w = infer_w / W
    stride_h = infer_h / H

    # Ratio to convert from inference coordinates to original image coordinates
    ratio_w = orig_w / infer_w
    ratio_h = orig_h / infer_h

    # Generate boxes in inference coordinates first
    x1_infer = x_indices.float() * stride_w
    y1_infer = y_indices.float() * stride_h
    x2_infer = (x_indices.float() + 1) * stride_w
    y2_infer = (y_indices.float() + 1) * stride_h

    # Convert to original image coordinates
    x1 = x1_infer * ratio_w
    y1 = y1_infer * ratio_h
    x2 = x2_infer * ratio_w
    y2 = y2_infer * ratio_h

    confs = conf[y_indices, x_indices]
    classes = cls[y_indices, x_indices].float()

    boxes = torch.stack([x1, y1, x2, y2, confs, classes], dim=1)

    return boxes


def expand_grid_boxes(grid_boxes, orig_size, grid_size, infer_size, expand_cells=1):
    """Expand grid boxes for better visualization."""
    if len(grid_boxes) == 0 or expand_cells == 0:
        return grid_boxes

    orig_w, orig_h = orig_size
    H, W = grid_size
    infer_w, infer_h = infer_size

    # Cell size in inference space (square)
    stride_w = infer_w / W
    stride_h = infer_h / H

    # Ratio to convert between original and inference coordinates
    ratio_w = orig_w / infer_w
    ratio_h = orig_h / infer_h

    # Cell size in original image space
    cell_w = stride_w * ratio_w
    cell_h = stride_h * ratio_h

    boxes = grid_boxes.clone()

    # Convert to grid coordinates
    grid_x1 = (boxes[:, 0] / cell_w).floor()
    grid_y1 = (boxes[:, 1] / cell_h).floor()
    grid_x2 = (boxes[:, 2] / cell_w).ceil()
    grid_y2 = (boxes[:, 3] / cell_h).ceil()

    # Expand in grid space
    grid_x1 = (grid_x1 - expand_cells).clamp(min=0)
    grid_y1 = (grid_y1 - expand_cells).clamp(min=0)
    grid_x2 = (grid_x2 + expand_cells).clamp(max=W)
    grid_y2 = (grid_y2 + expand_cells).clamp(max=H)

    # Convert back to pixel coordinates
    boxes[:, 0] = grid_x1 * cell_w
    boxes[:, 1] = grid_y1 * cell_h
    boxes[:, 2] = grid_x2 * cell_w
    boxes[:, 3] = grid_y2 * cell_h

    return boxes


def match_grid_to_det(grid_boxes, det_boxes, det_labels, max_w=4096):
    """Retain original grid candidates with positive-area, same-class overlap.

    max_w is retained for call compatibility; explicit class equality avoids
    coordinate-offset collisions and preserves each grid's score and category.
    """
    if grid_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
        return grid_boxes[:0]
    # Grid channel indices can differ from detection IDs; associate by category.
    same_class = torch.zeros((len(grid_boxes), len(det_labels)), dtype=torch.bool, device=grid_boxes.device)
    for grid_cls in grid_boxes[:, 5].unique().tolist():
        for det_cls in GRID2DET_MAP.get(int(grid_cls), (int(grid_cls),)):
            same_class |= (grid_boxes[:, 5:6] == grid_cls) & (det_labels[None, :] == det_cls)
    positive_overlap = box_ioa(grid_boxes[:, :4], det_boxes) > 0
    return grid_boxes[(same_class & positive_overlap).any(dim=1)]


def draw_boxes(image, labels, boxes, scores, thrh=0.45, show_conf=False, style_scale=1,
               show_label=True):
    """Draw boxes; show_label=False hides class, confidence and text background."""
    draw = ImageDraw.Draw(image)
    w, h = image.size

    max_side = max(w, h)
    linewidth = max(1, round(style_scale * max(int(max_side / 800), 2)))
    base_fontsize = 2 * max(int(max_side / 80), 16)
    fontsize = max(1, round(style_scale * base_fontsize))

    font = load_chinese_font(fontsize) if show_label else None
    text_stroke = max(1, round(style_scale * max(1, round(base_fontsize / 40))))

    mask = scores > thrh
    labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

    for j, box in enumerate(boxes):
        if not torch.isfinite(box).all():
            raise ValueError(f'Cannot draw non-finite detection box: {box.tolist()}')
        # Signed DEIM regression can yield nonpositive extents. These have no
        # drawable area; keep them in the inference record, but do not swap corners.
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        category = labels[j].item()
        color = DET_COLOR_MAP.get(category, (255, 255, 255))
        box = list(map(int, box))

        draw.rectangle(box, outline=color, width=linewidth)
        if not show_label:
            continue

        label = DET_LABEL_MAP.get(category, f"Class {category}")
        if show_conf:
            text = f"{label} {scores[j].item():.2f}"
        else:
            text = label

        if hasattr(font, 'getbbox'):
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=text_stroke)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(text, font)
            bbox = (0, 0, text_width, text_height)

        text_x = box[0]
        text_y = box[1] - text_height - 4 - linewidth // 2

        if text_y < 0:
            text_y = box[1] + linewidth + 2
        if text_x + text_width + 4 > w:
            text_x = w - text_width - 4
        text_x = max(0, text_x)

        pad = max(4, text_stroke + 2)
        text_x = max(pad, min(text_x, w - text_width - pad))
        text_y = max(pad, min(text_y, h - text_height - pad))
        text_background = [text_x - pad, text_y - pad, text_x + text_width + pad, text_y + text_height + pad]
        draw.rectangle(text_background, fill=color)
        draw.text((text_x - bbox[0], text_y - bbox[1]), text, fill="white", font=font,
                  stroke_width=text_stroke, stroke_fill="white")

    return image


def draw_grid_boxes(image, grid_boxes, show_conf=False, use_det_class=False, style_scale=1):
    """
    Draw grid boxes on image. No individual labels on each grid cell.

    Args:
        image: PIL Image
        grid_boxes: (N, 6) [x1, y1, x2, y2, conf, class]
        show_conf: whether to show confidence (not used)
        use_det_class: if True, use detection class colors
    """
    w, h = image.size
    linewidth = max(1, round(style_scale * max(int(max(w, h) / 800), 2)))

    if len(grid_boxes) == 0:
        return image

    # Convert PIL to numpy for OpenCV drawing
    img_array = np.array(image)

    for box in grid_boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)

        if use_det_class:
            color = DET_COLOR_MAP.get(cls, (255, 255, 255))
        else:
            color = GRID_COLOR_MAP.get(cls, (255, 255, 255))

        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)

        # cv2 writes directly into this RGB buffer; keep RGB channel order.
        cv2.rectangle(img_array, (x1_int, y1_int), (x2_int, y2_int), color, linewidth)

    return Image.fromarray(img_array)


def draw_grid_boxes_with_legend(image, grid_boxes, use_det_class=False):
    """
    Draw grid boxes on image with class labels next to grid cells.
    For groups of same class, only one label is shown.
    """
    w, h = image.size
    linewidth = max(int(max(w, h) / 800), 2)
    fontsize = 2 * max(int(max(w, h) / 80), 16)

    if len(grid_boxes) == 0:
        return image

    # Convert PIL to numpy for OpenCV drawing
    img_array = np.array(image)


    for box in grid_boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)

        if use_det_class:
            color = DET_COLOR_MAP.get(cls, (255, 255, 255))
        else:
            color = GRID_COLOR_MAP.get(cls, (255, 255, 255))

        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)

        # cv2 writes directly into this RGB buffer; keep RGB channel order.
        cv2.rectangle(img_array, (x1_int, y1_int), (x2_int, y2_int), color, linewidth)


    image = Image.fromarray(img_array)
    draw = ImageDraw.Draw(image)


    font = load_chinese_font(fontsize)
    text_stroke = max(1, round(fontsize / 40))


    class_groups = {}
    for box in grid_boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append((x1, y1, x2, y2))


    for cls, boxes in class_groups.items():
        if use_det_class:
            color = DET_COLOR_MAP.get(cls, (255, 255, 255))
            label = DET_LABEL_MAP.get(cls, f"Class {cls}")
        else:
            color = GRID_COLOR_MAP.get(cls, (255, 255, 255))
            label = GRID_LABEL_MAP.get(cls, f"Class {cls}")


        centers = [(x1 + x2) / 2 for x1, y1, x2, y2 in boxes]
        avg_center_x = sum(centers) / len(centers)


        min_dist = float('inf')
        best_box = boxes[0]
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            dist = abs(center_x - avg_center_x)
            if dist < min_dist:
                min_dist = dist
                best_box = box

        x1, y1, x2, y2 = best_box


        if hasattr(font, 'getbbox'):
            bbox = draw.textbbox((0, 0), label, font=font, stroke_width=text_stroke)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(label, font)
            bbox = (0, 0, text_width, text_height)


        text_x = int(x1)
        text_y = int(y1) - text_height - 4 - linewidth // 2


        if text_y < 0:
            text_y = int(y2) + linewidth + 2
        if text_x + text_width + 4 > w:
            text_x = w - text_width - 4
        text_x = max(0, text_x)


        pad = max(4, text_stroke + 2)
        text_x = max(pad, min(text_x, w - text_width - pad))
        text_y = max(pad, min(text_y, h - text_height - pad))
        text_background = [text_x - pad, text_y - pad, text_x + text_width + pad, text_y + text_height + pad]
        draw.rectangle(text_background, fill=color)
        draw.text((text_x - bbox[0], text_y - bbox[1]), label, fill="white", font=font,
                  stroke_width=text_stroke, stroke_fill="white")

    return image


def resize_keep_aspect(image, target_size, stride=32):
    """Resize image keeping aspect ratio.
    Height is fixed to target_size, width is scaled proportionally and aligned to stride."""
    w, h = image.size

    # Scale to target height, keep aspect ratio
    ratio = h / target_size
    new_h = target_size
    new_w = max(stride, round(w / ratio / stride) * stride)  # Align to stride

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    return resized, (new_w, new_h)


@torch.inference_mode()
def process_dataset(model, dataset_path, output_path,
                    det_thrh=0.5, grid_thrh=0.5,
                    size=(640, 640), vit_backbone=False,
                    save_det=True, save_grid=True, save_matched=True, save_combined=True,
                    show_det_conf=True, show_grid_conf=False, expand_grid_cells=0,
                    keep_aspect_ratio=False, edge_filter=0.0,
                    runtime_info=None, vis_classes=None, show_det_label=True):
    """Decode D and G, refine G by same-class IoA > 0, then render.

    All matching uses original 1-cell boxes. Optional expansion affects drawing
    only. Every requested image is saved, including empty predictions, so a
    rerun cannot silently retain an older result for an input image.
    """
    if not (0 <= det_thrh <= 1 and 0 <= grid_thrh <= 1):
        raise ValueError("Detection and grid thresholds must lie in [0, 1].")
    if not 0 <= edge_filter < .5 or expand_grid_cells < 0:
        raise ValueError("Use 0 <= edge_filter < 0.5 and expand_grid_cells >= 0.")
    visible_ids = None
    visible_grid_ids = None
    if vis_classes is not None:
        vis_classes = [name.upper() for name in vis_classes]
        name_to_id = {name: category for category, name in DET_LABEL_MAP.items()}
        if not vis_classes or any(name not in name_to_id for name in vis_classes):
            raise ValueError("vis_classes must contain category names: LC, TC, LR, SR.")
        visible_ids = [name_to_id[name] for name in vis_classes]
        grid_names = globals().get('GRID_LABEL_MAP', DET_LABEL_MAP)
        visible_grid_ids = [i for i, name in grid_names.items() if name in vis_classes]
    os.makedirs(output_path, exist_ok=True)
    enabled = dict(det=save_det, grid=save_grid, matched=save_matched, combined=save_combined)
    for folder, save in enabled.items():
        if save:
            os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    image_paths = sorted(os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    target_h = size[0] if isinstance(size, (list, tuple)) else size
    target_w = size[1] if isinstance(size, (list, tuple)) else size
    if min(target_h, target_w) <= 0:
        raise ValueError("Inference image size must be positive.")
    print(f"Found {len(image_paths)} images; detection score > {det_thrh}; grid score >= {grid_thrh}")
    print(f"Inference size: {size}; keep aspect ratio: {keep_aspect_ratio}")
    print(f"Displayed classes: {', '.join(vis_classes) if vis_classes is not None else 'all'}")
    print(f"Output directory: {os.path.abspath(output_path)}")
    if edge_filter:
        print("Note: edge filtering is additional to the paper's IoA association; use --edge-filter 0 to reproduce it.")
    records = []
    for idx, file_path in enumerate(tqdm(image_paths, file=sys.stdout, desc="Processing")):
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()
        if keep_aspect_ratio:
            im_resized, actual_infer_size = resize_keep_aspect(im_pil, target_h, 32)
        else:
            im_resized = im_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
            actual_infer_size = (target_w, target_h)
        im_tensor = T.ToTensor()(im_resized)
        if vit_backbone:
            im_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(im_tensor)
        output = model(im_tensor.unsqueeze(0).cuda(), orig_size)[0]
        raw_scores = output['scores']
        det_mask = raw_scores > det_thrh
        det_labels = output['labels'][det_mask]
        det_boxes = output['boxes'][det_mask]
        det_scores = raw_scores[det_mask]
        grid_logits = output.get('pred_grid_logits')
        grid_boxes = det_boxes.new_empty((0, 6))
        matched_grid = grid_boxes
        grid_display = grid_boxes
        matched_display = grid_boxes
        if grid_logits is not None:
            grid_boxes = grid2box(grid_logits, grid_thrh, (w, h), actual_infer_size, edge_filter=edge_filter)
            matched_grid = match_grid_to_det(grid_boxes, det_boxes, det_labels)
            grid_shape = grid_logits.shape[-2:]
            grid_display = expand_grid_boxes(grid_boxes, (w, h), grid_shape, actual_infer_size, expand_grid_cells)
            matched_display = expand_grid_boxes(matched_grid, (w, h), grid_shape, actual_infer_size, expand_grid_cells)
        basename = os.path.basename(file_path)
        # Filter only the rendered copies, after full-class decoding and matching.
        display_labels, display_boxes, display_scores = det_labels, det_boxes, det_scores
        if visible_ids is not None:
            selected = det_labels.new_tensor(visible_ids)
            visible_det = (det_labels[:, None] == selected[None, :]).any(dim=1)
            display_labels = det_labels[visible_det]
            display_boxes = det_boxes[visible_det]
            display_scores = det_scores[visible_det]
            selected_grid = grid_display.new_tensor(visible_grid_ids)
            grid_display = grid_display[(grid_display[:, 5:6] == selected_grid[None, :]).any(dim=1)]
            matched_display = matched_display[(matched_display[:, 5:6] == selected_grid[None, :]).any(dim=1)]
        if save_det:
            draw_boxes(im_pil.copy(), display_labels, display_boxes, display_scores, det_thrh,
                       show_conf=show_det_conf, style_scale=2,
                       show_label=show_det_label).save(os.path.join(output_path, 'det', basename))
        if save_grid:
            draw_grid_boxes(im_pil.copy(), grid_display, use_det_class=False, style_scale=2).save(
                os.path.join(output_path, 'grid', basename))
        if save_matched:
            draw_grid_boxes(im_pil.copy(), matched_display, use_det_class=False, style_scale=2).save(
                os.path.join(output_path, 'matched', basename))
        if save_combined:
            combined = draw_grid_boxes_with_legend(im_pil.copy(), matched_display, use_det_class=False)
            draw_boxes(combined, display_labels, display_boxes, display_scores, det_thrh,
                       show_conf=show_det_conf,
                       show_label=show_det_label).save(os.path.join(output_path, 'combined', basename))
        records.append(dict(image=basename, original_size=[w, h], inference_size=list(actual_infer_size),
                            raw_detection_count=len(raw_scores), detection_count=len(det_scores),
                            grid_count=len(grid_boxes), matched_grid_count=len(matched_grid),
                            displayed_detection_count=len(display_scores),
                            displayed_grid_count=len(grid_display), displayed_matched_grid_count=len(matched_display),
                            detections=[dict(box=box.tolist(), score=float(score), category=int(label))
                                        for box, score, label in zip(det_boxes.cpu(), det_scores.cpu(), det_labels.cpu())]))
    report = dict(runtime=runtime_info or {},
                  settings=dict(det_threshold=det_thrh, grid_threshold=grid_thrh,
                                detection_threshold_comparison=">", grid_threshold_comparison=">=",
                                image_size=size, keep_aspect_ratio=keep_aspect_ratio,
                                vis_classes=vis_classes,
                                detection_class_map=DET_LABEL_MAP,
                                grid_class_map=globals().get('GRID_LABEL_MAP', DET_LABEL_MAP),
                                edge_filter=edge_filter, expand_grid_cells=expand_grid_cells,
                                imagenet_normalization=bool(vit_backbone), combined="D + G_star",
                                association="same category and positive IoA; unexpanded cells"), images=records)
    with open(os.path.join(output_path, 'inference_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Visualization complete: {output_path}; audit: inference_summary.json")


def select_checkpoint_state(checkpoint, source='model'):
    """Choose a checkpoint source explicitly, matching the unified Road export."""
    if not isinstance(checkpoint, dict) or source not in ('model', 'ema', 'auto'):
        raise ValueError('Expected a checkpoint mapping and model, ema or auto source')
    if source == 'auto':
        source = 'ema' if checkpoint.get('ema') is not None else 'model'
    state = checkpoint.get(source)
    if source == 'ema':
        if not isinstance(state, dict) or 'module' not in state:
            raise ValueError('Requested EMA weights are missing or do not contain ema.module')
        state = state['module']
    if not isinstance(state, dict):
        raise ValueError(f'Requested {source} weights are missing or are not a state dictionary')
    return source, state


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    annotation_path = getattr(args, 'annotations', None)
    if annotation_path is None:
        for loader in ('train_dataloader', 'val_dataloader'):
            candidate = cfg.yaml_cfg.get(loader, {}).get('dataset', {}).get('ann_file')
            if candidate and os.path.isfile(candidate):
                annotation_path = candidate
                break
    if annotation_path:
        with open(annotation_path, encoding='utf-8') as annotation_file:
            configure_class_maps(json.load(annotation_file)['categories'], cfg.yaml_cfg.get('num_classes', 4))
        print(f"Category metadata: {annotation_path}")
    else:
        print(f"[Warning] No annotation metadata found; using category IDs {DET_LABEL_MAP}. "
              "Pass --annotations to verify the training category mapping.")

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    if 'DINOv3STAs' in cfg.yaml_cfg:
        # The full checkpoint below restores the backbone as well as both heads.
        cfg.yaml_cfg['DINOv3STAs']['weights_path'] = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_source, state = select_checkpoint_state(checkpoint, getattr(args, 'model_source', 'model'))
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state
    cfg.model.load_state_dict(state, strict=True)
    print(f'Loaded full checkpoint strictly: {args.resume} ({model_source})')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            # Post-process detection results
            det_outputs = self.postprocessor(outputs, orig_target_sizes)

            # Add grid logits to output if available
            if 'pred_grid_logits' in outputs:
                for i, det_out in enumerate(det_outputs):
                    det_out['pred_grid_logits'] = outputs['pred_grid_logits'][i]

            return det_outputs

    model = Model()

    # Enable dynamic spatial size for variable input sizes AFTER creating Model
    if args.img is not None or args.keep_aspect_ratio:
        print("Using dynamic spatial sizes for the requested resize mode")
        # Enable dynamic spatial size for variable input sizes
        if hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'eval_spatial_size'):
            model.model.encoder.eval_spatial_size = None
            print("Enabled dynamic position embedding in encoder")
        if hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'eval_spatial_size'):
            model.model.decoder.eval_spatial_size = None
            # Delete pre-registered anchors and valid_mask
            if hasattr(model.model.decoder, 'anchors'):
                delattr(model.model.decoder, 'anchors')
            if hasattr(model.model.decoder, 'valid_mask'):
                delattr(model.model.decoder, 'valid_mask')
            print("Enabled dynamic spatial size in decoder and removed pre-registered buffers")
        # The explicit resize flag is respected even with --img.

    # Get inference size from args or config
    if args.img is not None:
        img_size = args.img  # Use command line argument
    else:
        img_size = cfg.yaml_cfg["eval_spatial_size"]
        print(f"Using inference size from config: {img_size}")

    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)
    with open(__file__, 'rb') as script_file:
        script_sha256 = hashlib.sha256(script_file.read()).hexdigest()

    process_dataset(
        model,
        args.dataset,
        args.output,
        det_thrh=args.det_thrh,
        grid_thrh=args.grid_thrh,
        size=img_size,
        vit_backbone=vit_backbone,
        save_det=args.save_det,
        save_grid=args.save_grid,
        save_matched=args.save_matched,
        save_combined=args.save_combined,
        show_det_conf=args.show_det_conf,
        show_det_label=not getattr(args, 'no_det_label', False),
        show_grid_conf=args.show_grid_conf,
        expand_grid_cells=args.expand_grid_cells,
        keep_aspect_ratio=args.keep_aspect_ratio,
        edge_filter=args.edge_filter,
        vis_classes=getattr(args, 'vis_classes', None),
        runtime_info=dict(config=os.path.abspath(args.config), checkpoint=os.path.abspath(args.resume),
                          model_source=model_source,
                          requested_model_source=getattr(args, 'model_source', 'model'),
                          script=os.path.abspath(__file__),
                          script_sha256=script_sha256)
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('--model-source', choices=['model', 'ema', 'auto'], default='model',
                        help="Checkpoint source. model matches the unified Road evaluation; auto prefers EMA.")
    parser.add_argument('-d', '--dataset', type=str, default='./data/fiftyone/validation/data')
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save visualized results")
    parser.add_argument('--annotations', type=str, default=None,
                        help="COCO annotation JSON for category IDs; defaults to the config's training/validation annotations")

    # Inference size
    parser.add_argument('--img', '--imgsz', type=int, default=None,
                       help="Override input size (e.g., 640, 1024); square unless --keep-aspect-ratio is set.")

    # Thresholds
    parser.add_argument('--det-thrh', type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument('--grid-thrh', type=float, default=0.5, help="Grid confidence threshold")

    # Output options
    parser.add_argument('--save-det', action='store_true', help="Save detection-only images")
    parser.add_argument('--save-grid', action='store_true', help="Save raw grid images before box association")
    parser.add_argument('--save-matched', action='store_true', help="Save matched grid images")
    parser.add_argument('--save-combined', action='store_true', help="Save combined images")
    parser.add_argument('--save-all', action='store_true', help="Save all types of images")
    parser.add_argument('--vis-classes', nargs='+', type=str.upper,
                        choices=['LC', 'TC', 'LR', 'SR'], default=None,
                        help="Classes to draw, e.g. --vis-classes LC TC. Defaults to all; inference is unchanged.")

    # Display options
    parser.add_argument('--show-det-conf', action='store_true', help="Show confidence on detection boxes")
    parser.add_argument('--show-grid-conf', action='store_true', help="Show confidence on grid cells")
    parser.add_argument('--no-det-conf', action='store_true', help="Hide confidence on detection boxes")
    parser.add_argument('--no-det-label', action='store_true',
                        help="Hide detection class, confidence and text background; draw boxes only")
    parser.add_argument('--expand-grid-cells', type=int, default=0,
                       help="Expand grid boxes for visualization (0=1x1 cell, 1=3x3 cells, 2=5x5 cells)")
    resize_group = parser.add_mutually_exclusive_group()
    resize_group.add_argument('--keep-aspect-ratio', dest='keep_aspect_ratio', action='store_true',
                              help="Resize to the target height and a stride-aligned width; no padding")
    resize_group.add_argument('--no-keep-aspect-ratio', dest='keep_aspect_ratio', action='store_false',
                              help="Direct resize to the configured H,W, or --img square (default)")
    parser.set_defaults(keep_aspect_ratio=False)
    parser.add_argument('--edge-filter', type=float, default=0.0,
                       help="Filter out grids near edges (0.0-1.0,)")

    args = parser.parse_args()

    # Handle save-all option
    if args.save_all:
        args.save_det = True
        args.save_grid = True
        args.save_matched = True
        args.save_combined = True

    # Default: save combined if nothing specified
    if not any([args.save_det, args.save_grid, args.save_matched, args.save_combined]):
        args.save_combined = True

    # Handle confidence display
    if args.no_det_conf or args.no_det_label:
        args.show_det_conf = False
    elif not args.show_det_conf:
        args.show_det_conf = True  # Default: show detection confidence

    # Handle aspect ratio option
    if args.keep_aspect_ratio:
        args.keep_aspect_ratio = True
    else:
        args.keep_aspect_ratio = False

    main(args)
