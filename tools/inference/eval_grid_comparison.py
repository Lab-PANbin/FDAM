"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Offline raw/matched crack grid metrics from train.py's grid_export records.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def load_records(manifest_path):
    path = Path(manifest_path)
    manifest = read_json(path)
    if manifest.get('format_version') != 1 or not manifest.get('complete'):
        raise ValueError('Expected a complete version-1 grid export manifest')
    if len(manifest['images']) != manifest['image_count'] or not manifest['images']:
        raise ValueError('Manifest image count is inconsistent or empty')
    ids = set()
    records = []
    for info in manifest['images']:
        if info['image_id'] in ids:
            raise ValueError('Duplicate image ID in manifest')
        ids.add(info['image_id'])
        with np.load(path.parent/info['record'], allow_pickle=False) as data:
            item = {key: data[key].copy() for key in
                    ('probability', 'gt', 'boxes', 'scores', 'labels', 'orig_size')}
        prob, gt = item['probability'], item['gt']
        if prob.ndim != 2 or list(prob.shape) != manifest['grid_shape'] or gt.shape != prob.shape:
            raise ValueError(f'{info}: grid shape does not match the manifest')
        if (not np.issubdtype(prob.dtype, np.floating) or not np.isfinite(prob).all()
                or np.any((prob < 0) | (prob > 1)) or not np.isin(gt, [0, 1]).all()):
            raise ValueError(f'{info}: invalid probability or GT')
        if item['orig_size'].shape != (2,) or np.any(item['orig_size'] <= 0):
            raise ValueError('Invalid original image size')
        boxes, scores, labels = item['boxes'], item['scores'], item['labels']
        if boxes.shape != (len(scores), 4) or scores.ndim != 1 or labels.shape != scores.shape:
            raise ValueError('Invalid detection shapes')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('Invalid detections')
        item.update(info)
        item['_association_cache'] = {}
        records.append(item)
    return records


def association_mask(grid_shape, orig_size, boxes, scores, labels, det_threshold):
    """Class 0 grid matches class 0 detections with score > threshold and IoA > 0."""
    gh, gw = grid_shape
    width, height = orig_size
    keep = (scores > det_threshold) & (labels == 0)
    boxes = np.asarray(boxes)[keep]
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if not len(boxes):
        return np.zeros(grid_shape, dtype=bool)
    row, col = np.indices(grid_shape)
    x1, y1 = col[..., None]*width/gw, row[..., None]*height/gh
    x2, y2 = (col[..., None]+1)*width/gw, (row[..., None]+1)*height/gh
    # Positive area, rather than edge contact or a minimum IoU, is the paper rule.
    overlap = ((np.minimum(x2, boxes[:, 2]) > np.maximum(x1, boxes[:, 0])) &
               (np.minimum(y2, boxes[:, 3]) > np.maximum(y1, boxes[:, 1])))
    return overlap.any(axis=-1)


def matched_mask(item, det_threshold):
    cache = item['_association_cache']
    if det_threshold not in cache:
        cache[det_threshold] = association_mask(item['probability'].shape, item['orig_size'],
            item['boxes'], item['scores'], item['labels'], det_threshold)
    return cache[det_threshold]


def confusion(gt, prediction):
    # Rows GT, columns predicted: 0 crack, 1 background, including all cells.
    indices = 2*(1-gt.astype(np.int64)) + (1-prediction.astype(np.int64))
    return np.bincount(indices.ravel(), minlength=4).reshape(2, 2)


def metrics(cm):
    per_class = []
    for c in range(2):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum()-tp); fn = int(cm[c, :].sum()-tp)
        per_class.append(dict(precision=tp/(tp+fp) if tp+fp else 0.,
            recall=tp/(tp+fn) if tp+fn else 0.,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            iou=tp/(tp+fp+fn) if tp+fp+fn else 0., tp=tp, fp=fp, fn=fn,
            gt_count=int(cm[c, :].sum())))
    return dict(foreground_only=per_class[0],
        include_background={k: float(np.mean([p[k] for p in per_class]))
                            for k in ('precision', 'recall', 'f1', 'iou')},
        per_class=dict(crack=per_class[0], background=per_class[1]),
        confusion_matrix=cm.tolist(), confusion_order=['crack', 'background'])


def score_records(records, grid_threshold, det_threshold=None):
    cm = np.zeros((2, 2), dtype=np.int64)
    for item in records:
        pred = item['probability'] >= grid_threshold  # Match existing dense evaluator.
        if det_threshold is not None:
            pred = pred & matched_mask(item, det_threshold)
        cm += confusion(item['gt'], pred)
    return dict(grid_threshold=float(grid_threshold),
                det_threshold=None if det_threshold is None else float(det_threshold), **metrics(cm))


def select_thresholds(records, grid_thresholds, det_thresholds, metadata):
    rows = []
    best_raw = best_matched = None
    for grid_threshold in sorted(set(grid_thresholds)):
        raw = score_records(records, grid_threshold)
        if best_raw is None or raw['foreground_only']['f1'] > best_raw['foreground_only']['f1']:
            best_raw = raw
        rows.append(dict(variant='raw', grid_threshold=grid_threshold, det_threshold='',
                         **raw['foreground_only']))
        for det_threshold in sorted(set(det_thresholds)):
            matched = score_records(records, grid_threshold, det_threshold)
            if best_matched is None or matched['foreground_only']['f1'] > best_matched['foreground_only']['f1']:
                best_matched = matched
            rows.append(dict(variant='matched', grid_threshold=grid_threshold, det_threshold=det_threshold,
                             **matched['foreground_only']))
    selection = dict(format_version=1, split='val', metric='foreground_f1',
        tie_break='lowest grid threshold, then lowest detection threshold',
        source_annotation_sha256=metadata['annotation_sha256'],
        checkpoint=metadata.get('checkpoint'), checkpoint_sha256=metadata.get('checkpoint_sha256'),
        model_source=metadata.get('model_source'),
        raw=dict(grid_threshold=best_raw['grid_threshold']),
        matched=dict(grid_threshold=best_matched['grid_threshold'], det_threshold=best_matched['det_threshold']))
    return selection, rows


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_overlays(item, raw, matched, image_root, output, matched_only=False):
    import cv2
    image_path = Path(image_root)/item['file_name']
    image = cv2.imread(str(image_path))
    if image is None or list(image.shape[1::-1]) != item['orig_size'].tolist():
        raise ValueError(f'Original image missing or wrong size: {image_path}')
    height, width = image.shape[:2]
    gh, gw = raw.shape
    truth = item['gt'].astype(bool)

    def draw(mask, colors=None):
        canvas = image.copy()
        for row, col in np.argwhere(mask):
            color = (255, 255, 0) if colors is None else tuple(int(x) for x in colors[row, col])
            cv2.rectangle(canvas, (round(col*width/gw), round(row*height/gh)),
                (min(width-1, round((col+1)*width/gw)), min(height-1, round((row+1)*height/gh))),
                color, max(int(max(width, height)/800), 2))
        return canvas

    if matched_only:
        panels = {'matched': draw(matched)}
    else:
        panels = dict(gt=draw(truth), raw=draw(raw), matched=draw(matched))
        for variant, prediction in (('raw', raw), ('matched', matched)):
            colors = np.zeros((gh, gw, 3), dtype=np.uint8)
            colors[truth & prediction] = (0, 180, 0)       # TP green
            colors[~truth & prediction] = (0, 0, 255)     # FP red
            colors[truth & ~prediction] = (0, 255, 255)   # FN yellow
            panels[f'errors_{variant}'] = draw(truth | prediction, colors)
        # Titles occupy a separate strip; standalone overlays keep their original pixels and size.
        titles = ('GT', 'Raw Grid', 'Matched Grid')
        font = cv2.FONT_HERSHEY_SIMPLEX
        padding = max(4, round(width/100))
        scale = max(.4, width/650)
        thickness = max(1, round(2*scale))
        longest = max(cv2.getTextSize(title, font, scale, thickness)[0][0] for title in titles)
        scale *= min(1., max(1, width-2*padding)/longest)
        text_sizes = [cv2.getTextSize(title, font, scale, thickness) for title in titles]
        header_height = max(32, round(width*.07), max(size[0][1]+size[1] for size in text_sizes)+2*padding)
        titled = []
        for key, title, ((text_w, text_h), baseline) in zip(('gt', 'raw', 'matched'), titles, text_sizes):
            header = np.full((header_height, width, 3), 255, np.uint8)
            position = (max(0, (width-text_w)//2), (header_height-text_h-baseline)//2+text_h)
            cv2.putText(header, title, position, font, scale, (30, 30, 30), thickness, cv2.LINE_AA)
            titled.append(np.concatenate([header, panels[key]], axis=0))
        gap = np.full((height+header_height, 12, 3), 255, np.uint8)
        panels['compare'] = np.concatenate([titled[0], gap, titled[1], gap, titled[2]], axis=1)
    for folder, pixels in panels.items():
        directory = output/folder
        directory.mkdir(exist_ok=True)
        name = f'{item["image_id"]}_{Path(item["file_name"]).stem}.png'
        if not cv2.imwrite(str(directory/name), pixels):
            raise OSError(f'Could not write {directory/name}')


def write_visualization_guide(output, raw_threshold, matched_threshold, det_threshold, matched_only=False):
    if matched_only:
        text = f'''matched：最终网格结果，保持原图尺寸，不添加检测框、标题或逐格文字。
保留概率 >= {matched_threshold:g} 且与置信度 > {det_threshold:g} 的同类别检测框有正面积交集的网格。
青色线表示保留的正类网格；没有网格通过筛选时保存原图。
'''
        (output/'visualization_guide.txt').write_text(text, encoding='utf-8')
        return
    text = f'''compare 图片从左到右为三个面板（横向排列）：
1. GT：真实网格标签，由标注区域的九点投影生成，不是像素级分割掩膜。
2. Raw Grid：预测概率 >= {raw_threshold:g} 的网格，未经过检测框关联过滤。
3. Matched Grid：预测概率 >= {matched_threshold:g}，且与置信度 > {det_threshold:g} 的同类别检测框有正面积交集的网格。

compare 中的三张图只显示网格，不绘制检测框。青色线表示该面板中的正类网格。
Raw 和 Matched 分别使用各自的阈值；阈值不同时，Matched 不一定是图中 Raw 的子集。
comparison.json 中 selected.raw_at_matched_threshold 可用于同一网格阈值下比较过滤前后的指标。

gt、raw、matched：对应的独立图片，保持原图尺寸、不添加标题。
errors_raw、errors_matched：绿色 = TP（预测正确），红色 = FP（误检），黄色 = FN（漏检）；正确背景不画线。
所有图片都对应导出记录中的网格分辨率，并映射回原图坐标。
'''
    (output/'visualization_guide.txt').write_text(text, encoding='utf-8')


def evaluate(args):
    if args.sweep and (args.split != 'val' or args.selection):
        raise ValueError('Threshold search requires validation (--split val) and no --selection')
    for value in [args.grid_threshold, args.det_threshold, *args.grid_thresholds, *args.det_thresholds]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Thresholds must be finite and within [0,1]')
    if not args.grid_thresholds or not args.det_thresholds:
        raise ValueError('Threshold search lists cannot be empty')
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    metadata = manifest['metadata']
    box_metrics = manifest.get('box_metrics')
    if getattr(args, 'require_box_metrics', False) and box_metrics is None:
        raise ValueError('Missing box metrics; re-export with the updated evaluator for a joint report')
    if box_metrics is not None:
        if any(key not in box_metrics or not np.isfinite(box_metrics[key])
               or not -1 <= box_metrics[key] <= 1 for key in ('ap', 'ap50', 'ap75')):
            raise ValueError('Invalid box metrics in manifest')
    if args.sweep and 'test' in Path(metadata['annotation_file']).stem.lower():
        raise ValueError('Threshold search requires validation annotations, not a test annotation file')
    print(f'Loading grid predictions: {manifest_path.resolve()}', flush=True)
    records = load_records(manifest_path)
    print(f'Loaded {len(records)} images; computing metrics (split={args.split}).', flush=True)
    selection = None
    rows = []
    if args.sweep:
        selection, rows = select_thresholds(records, args.grid_thresholds, args.det_thresholds, metadata)
        selection.update(grid_shape=manifest['grid_shape'], infer_shape=manifest['infer_shape'],
                         grid_delta=metadata['grid_delta'])
    elif args.selection:
        selection = read_json(args.selection)
        if selection.get('format_version') != 1 or selection.get('split') != 'val':
            raise ValueError('Selection must come from a validation sweep')
        if args.split == 'test' and selection['source_annotation_sha256'] == metadata['annotation_sha256']:
            raise ValueError('Test evaluation uses the same annotation file content as threshold selection')
        for key, actual in (('grid_shape', manifest['grid_shape']), ('infer_shape', manifest['infer_shape']),
                            ('grid_delta', metadata['grid_delta']), ('model_source', metadata.get('model_source'))):
            if selection[key] != actual:
                raise ValueError(f'Validation/test {key} differs')
        key = 'checkpoint_sha256' if selection.get('checkpoint_sha256') else 'checkpoint'
        if selection.get(key) != metadata.get(key):
            raise ValueError('Validation/test checkpoints differ; select thresholds with the same checkpoint')
        for value in (selection['raw']['grid_threshold'], selection['matched']['grid_threshold'],
                      selection['matched']['det_threshold']):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid threshold in selection file')
    out = Path(args.output_dir) if args.output_dir else manifest_path.parent/'comparison'
    out.mkdir(parents=True, exist_ok=True)
    reference = dict(raw=score_records(records, args.grid_threshold),
                     matched=score_records(records, args.grid_threshold, args.det_threshold))
    raw_g = selection['raw']['grid_threshold'] if selection else args.grid_threshold
    matched_g = selection['matched']['grid_threshold'] if selection else args.grid_threshold
    matched_d = selection['matched']['det_threshold'] if selection else args.det_threshold
    selected = dict(raw=score_records(records, raw_g), matched=score_records(records, matched_g, matched_d),
                    raw_at_matched_threshold=score_records(records, matched_g))
    settings = dict(split=args.split, image_count=len(records), grid_shape=manifest['grid_shape'],
        infer_shape=manifest['infer_shape'], grid_delta=metadata['grid_delta'],
        annotation_sha256=metadata['annotation_sha256'],
        manifest=str(manifest_path.resolve()),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_metadata=metadata, selection_file=args.selection,
        selection_policy='validation foreground F1' if selection else 'fixed thresholds',
        grid_rule='probability >= grid threshold',
        matching_rule='detection score > threshold, same category, positive intersection area',
        confusion_order=['crack', 'background'])
    report = dict(settings=settings, box=box_metrics, reference=reference, selected=selected)
    if args.sweep:
        write_json(out/'selection.json', selection)
        write_csv(out/'threshold_sweep.csv', rows)
    per_image = []
    for item in records:
        raw = item['probability'] >= raw_g
        matched = (item['probability'] >= matched_g) & matched_mask(item, matched_d)
        row = dict(image_id=item['image_id'], file_name=item['file_name'], gt_count=int(item['gt'].sum()))
        for name, pred in (('raw', raw), ('matched', matched)):
            fg = metrics(confusion(item['gt'], pred))['foreground_only']
            row.update({f'{name}_{key}': fg[key] for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')})
        per_image.append(row)
    # Numeric evaluation is complete. Commit reports before optional image I/O so
    # missing images, write failures, or an interrupted rendering job cannot lose them.
    write_json(out/'comparison.json', report)
    write_csv(out/'per_image.csv', sorted(per_image, key=lambda row: (-row['raw_fp'], row['image_id'])))
    print(f'Saved metrics for {len(records)} images: {(out/"comparison.json").resolve()}', flush=True)
    if box_metrics is not None:
        print('Box (same checkpoint, COCO AP): ' + ', '.join(
            f'{name}={box_metrics[name]:.4f}' for name in ('ap', 'ap50', 'ap75')), flush=True)
    for group, values in (('reference', reference), ('selected' if selection else 'fixed', selected)):
        for variant, value in values.items():
            fg = value['foreground_only']
            print(f'{group}/{variant}: grid={value["grid_threshold"]:.3f}, det={value["det_threshold"]}; '
                  f'P={fg["precision"]:.4f}, R={fg["recall"]:.4f}, F1={fg["f1"]:.4f}, '
                  f'IoU={fg["iou"]:.4f}, TP={fg["tp"]}, FP={fg["fp"]}, FN={fg["fn"]}', flush=True)
    matched_only = getattr(args, 'matched_only', False)
    if args.save_images or matched_only:
        write_visualization_guide(out, raw_g, matched_g, matched_d, matched_only=matched_only)
        files_per_image = 1 if matched_only else 6
        print(f'Saving overlays for {len(records)} images ({files_per_image} PNG files per image). '
              'Numeric reports are already saved.', flush=True)
        for index, item in enumerate(records, 1):
            raw = item['probability'] >= raw_g
            matched = (item['probability'] >= matched_g) & matched_mask(item, matched_d)
            try:
                save_overlays(item, raw, matched, args.image_root or metadata['image_root'], out,
                              matched_only=matched_only)
            except Exception:
                print(f'Overlay export failed at image {index}/{len(records)}: {item["file_name"]}. '
                      f'Numeric reports remain available in {out.resolve()}.', flush=True)
                raise
            if index == 1 or index % 25 == 0 or index == len(records):
                print(f'Overlays saved: {index}/{len(records)} images.', flush=True)
    print(f'Evaluation complete: {out.resolve()}', flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--sweep', action='store_true', help='Select thresholds on validation only')
    parser.add_argument('--selection', help='Frozen selection.json from validation')
    parser.add_argument('--grid-threshold', type=float, default=.6, help='Reference raw/matched grid threshold')
    parser.add_argument('--det-threshold', type=float, default=.5, help='Reference detection threshold')
    parser.add_argument('--grid-thresholds', type=float, nargs='+',
                        default=[round(i*.05, 2) for i in range(1, 20)] + [.975, .99])
    parser.add_argument('--det-thresholds', type=float, nargs='+', default=[.3, .5, .7])
    parser.add_argument('--save-images', action='store_true',
                        help='Save GT/raw/matched grids, errors, and a labeled comparison')
    parser.add_argument('--matched-only', action='store_true',
                        help='Save only final matched grid images; implies --save-images')
    parser.add_argument('--require-box-metrics', action='store_true',
                        help='Require Box AP from the same exported inference pass for a joint report')
    parser.add_argument('--image-root', help='Override original image directory if evaluating on another host')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
