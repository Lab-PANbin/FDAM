"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Shared NumPy Road evaluation; select on val, apply frozen thresholds to test.
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

PROTOCOL = 'road-grid-v1'
NAMES = ['LC', 'TC', 'LR', 'SR']
GRID = [32, 32]
INFER = [1024, 1024]
GT_RULE = dict(projection='clipped_xywh_nine_points_floor', delta=.125,
               offsets=[-1, 0, 1], deduplicate=True, independent_channels=True)
GRID_THRESHOLDS = [i / 20 for i in range(1, 20)]
DET_THRESHOLDS = [.3, .5, .7]
ALIASES = {'纵向裂缝': 'LC', '横向裂缝': 'TC', '修补.块状': 'LR', '修补.条状': 'SR',
           '修补块状': 'LR', '修补条状': 'SR'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def category_maps(coco, class_names):
    """Map both category families by semantic name, independently of ID/order."""
    if len(class_names) != 4 or set(class_names) != set(NAMES):
        raise ValueError('Model class_names must contain LC, TC, LR, SR exactly once')
    positive, negative, ids = {}, {}, set()
    for category in coco['categories']:
        cid, raw = category['id'], category['name'].strip()
        name = ALIASES.get(raw, raw.upper())
        if not isinstance(cid, int) or cid in ids or name not in class_names:
            raise ValueError(f'Invalid or duplicate category: {category}')
        ids.add(cid)
        mapping = negative if cid < 0 else positive
        index = class_names.index(name)
        if index in mapping.values():
            raise ValueError(f'Duplicate category name: {name}')
        mapping[cid] = index
    if set(positive.values()) != set(range(4)) or set(negative.values()) != set(range(4)):
        raise ValueError('Positive and negative categories must each describe all four classes')
    return positive, negative


def build_targets(coco, class_names):
    positive, negative = category_maps(coco, class_names)
    images = {im['id']: im for im in coco['images']}
    targets = {iid: np.zeros((4, *GRID), dtype=bool) for iid in images}
    for ann in coco['annotations']:
        iid, cid = ann['image_id'], ann['category_id']
        if iid not in images or cid not in (positive | negative):
            raise ValueError('Annotation has unknown image or category ID')
        if cid >= 0:
            continue
        box = np.asarray(ann['bbox'], dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('Invalid grid annotation bbox')
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        width, height = images[iid]['width'], images[iid]['height']
        x1, x2 = np.clip([x, x + w], 0, width)
        y1, y2 = np.clip([y, y + h], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        xs = np.floor(((x1 + x2) / 2 + np.array([-1, 0, 1]) * .125 * (x2 - x1)) * GRID[1] / width).astype(int)
        ys = np.floor(((y1 + y2) / 2 + np.array([-1, 0, 1]) * .125 * (y2 - y1)) * GRID[0] / height).astype(int)
        xs, ys = xs[(xs >= 0) & (xs < GRID[1])], ys[(ys >= 0) & (ys < GRID[0])]
        targets[iid][negative[cid], ys[:, None], xs] = True
    return targets


def validate_record(record, image):
    probability = record['probability']
    if probability.dtype != np.float32 or probability.shape != (4, *GRID):
        raise ValueError('Expected float32 probability with shape 4x32x32; resampling is forbidden')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid grid probabilities')
    size = record['orig_size']
    if size.shape != (2,) or not np.array_equal(size, [image['width'], image['height']]):
        raise ValueError('Record original size does not match COCO image size')
    boxes, scores, labels = (record[key] for key in ('boxes', 'scores', 'labels'))
    if scores.ndim != 1 or boxes.shape != (len(scores), 4) or labels.shape != scores.shape:
        raise ValueError('Invalid detection array shapes')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid boxes or detection scores')
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 4)):
        raise ValueError('Detection labels must be zero-based model channel indices')
    if np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError('Detection xyxy boxes have reversed corners')


def load_export(manifest_path, coco_path, split):
    path = Path(manifest_path)
    manifest, coco = read_json(path), read_json(coco_path)
    if manifest.get('format_version') != 1 or manifest.get('complete') is not True or manifest.get('protocol') != PROTOCOL:
        raise ValueError('Expected a complete version-1 road-grid-v1 manifest')
    if manifest.get('split') != split or split not in ('val', 'test'):
        raise ValueError('Manifest split does not match requested split')
    if manifest.get('infer_shape') != INFER or manifest.get('grid_shape') != GRID:
        raise ValueError('Road protocol requires inference 1024x1024 and source grid 32x32')
    if manifest.get('annotation_sha256') != file_digest(coco_path):
        raise ValueError('COCO annotation_sha256 does not match exported predictions')
    for key in ('backend', 'checkpoint', 'checkpoint_sha256', 'model_source', 'inference_signature',
                'preprocessing', 'decoder', 'config_sha256'):
        if key not in manifest or manifest[key] in ('', None):
            raise ValueError(f'Manifest missing provenance field: {key}')
    names = manifest['class_names']
    category_maps(coco, names)
    images = {im['id']: im for im in coco['images']}
    if not images or len(images) != len(coco['images']):
        raise ValueError('COCO image list is empty or contains duplicate IDs')
    if any(im['width'] <= 0 or im['height'] <= 0 for im in images.values()):
        raise ValueError('Invalid COCO image dimensions')
    entries = manifest['images']
    ids = [entry['image_id'] for entry in entries]
    if len(entries) != manifest['image_count'] or len(set(ids)) != len(ids) or set(ids) != set(images):
        raise ValueError('Manifest must contain complete, unique COCO image coverage')
    targets, records, paths = build_targets(coco, names), [], set()
    for info in sorted(entries, key=lambda entry: entry['image_id']):
        image = images[info['image_id']]
        if info['file_name'] != image['file_name']:
            raise ValueError('Manifest file_name does not match COCO')
        record_path = (path.parent / info['record']).resolve()
        if not record_path.is_relative_to(path.parent.resolve()) or record_path in paths:
            raise ValueError('Record paths must be unique and inside the export directory')
        paths.add(record_path)
        with np.load(record_path, allow_pickle=False) as data:
            record = {key: data[key].copy() for key in ('probability', 'boxes', 'scores', 'labels', 'orig_size')}
        validate_record(record, image)
        record.update(info, gt=targets[info['image_id']])
        records.append(record)
    return manifest, coco, records


def association_mask(record, class_index, det_threshold):
    """A cell is retained iff a same-class box with score > threshold overlaps it."""
    boxes = record['boxes'][(record['labels'] == class_index) & (record['scores'] > det_threshold)]
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if not len(boxes):
        return np.zeros(GRID, bool)
    width, height = record['orig_size']
    row, col = np.indices(GRID)
    x1, x2 = col[..., None] * width / GRID[1], (col[..., None] + 1) * width / GRID[1]
    y1, y2 = row[..., None] * height / GRID[0], (row[..., None] + 1) * height / GRID[0]
    return ((np.minimum(x2, boxes[:, 2]) > np.maximum(x1, boxes[:, 0])) &
            (np.minimum(y2, boxes[:, 3]) > np.maximum(y1, boxes[:, 1]))).any(axis=-1)


def ranked_data(records, class_names, classes, det_threshold=None):
    result = {}
    for name in classes:
        index, confidence, truth, count = class_names.index(name), [], [], 0
        for record in records:
            probability = record['probability']
            winner = probability.argmax(axis=0)  # All channels participate, including out-of-scope classes.
            keep = winner == index
            if det_threshold is not None:
                keep &= association_mask(record, index, det_threshold)
            confidence.append(probability[index][keep])
            truth.append(record['gt'][index][keep])
            count += int(record['gt'][index].sum())
        scores, correct = np.concatenate(confidence), np.concatenate(truth)
        order = np.argsort(-scores, kind='stable')
        scores, correct = scores[order], correct[order]
        ap = 0.
        if count and len(scores):
            tp = correct.cumsum()
            recall, precision = tp / count, tp / np.arange(1, len(tp) + 1)
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([1.], precision, [0.]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            x = np.linspace(0, 1, 101)
            values = np.interp(x, mrec, mpre)
            ap = float(np.sum((values[1:] + values[:-1]) * np.diff(x) / 2))
        result[name] = (scores, correct, count, ap)
    return result


def summarize(data, grid_threshold, det_threshold=None):
    per_class = {}
    for name, (scores, correct, count, ap) in data.items():
        keep = scores >= grid_threshold
        tp, predictions = int(correct[keep].sum()), int(keep.sum())
        fp, fn = predictions - tp, count - tp
        per_class[name] = dict(precision=tp / predictions if predictions else 0.,
                               recall=tp / count if count else 0.,
                               f1=2 * tp / (predictions + count) if predictions + count else 0.,
                               ap=ap, tp=tp, fp=fp, fn=fn, gt_count=count, predictions=predictions)
    macro = {key: float(np.mean([row[key] for row in per_class.values()]))
             for key in ('precision', 'recall', 'f1', 'ap')}
    return dict(grid_threshold=grid_threshold, det_threshold=det_threshold, per_class=per_class, macro=macro)


def score_records(records, class_names, classes, grid_threshold, det_threshold=None):
    return summarize(ranked_data(records, class_names, classes, det_threshold), grid_threshold, det_threshold)


def provenance(manifest, classes, box_backend):
    if not classes or len(set(classes)) != len(classes) or not set(classes) <= set(manifest['class_names']):
        raise ValueError('Selected classes must be unique known model classes')
    try:
        version = importlib.metadata.version(box_backend)
    except importlib.metadata.PackageNotFoundError:
        version = 'unavailable'
    settings = dict(protocol=PROTOCOL, infer_shape=INFER, grid_shape=GRID, gt_rule=GT_RULE,
                    candidate='all-channel argmax, ties first channel', association='same-class positive area',
                    grid_comparison='>=', detection_comparison='>', ap='precision-envelope 101-point trapezoid',
                    macro='all selected classes, absent class scores zero', box_backend=box_backend)
    metadata = {key: manifest[key] for key in ('backend', 'checkpoint_sha256', 'model_source',
                'inference_signature', 'class_names', 'infer_shape', 'grid_shape', 'preprocessing',
                'decoder', 'config_sha256')}
    metadata.update(protocol=PROTOCOL, settings=settings, protocol_signature=digest(settings),
                    gt_signature=digest(GT_RULE), scope_signature=digest(classes), classes=classes,
                    annotation_sha256=manifest['annotation_sha256'], box_backend=box_backend,
                    box_backend_version=version, implementation_sha256=file_digest(__file__))
    return metadata


def select_thresholds(records, class_names, classes, metadata):
    rows, best = [], {}
    for variant, detection_thresholds in (('raw', [None]), ('matched', DET_THRESHOLDS)):
        candidates = []
        for detection_threshold in detection_thresholds:
            data = ranked_data(records, class_names, classes, detection_threshold)
            for grid_threshold in GRID_THRESHOLDS:
                score = summarize(data, grid_threshold, detection_threshold)
                row = dict(variant=variant, grid_threshold=grid_threshold, det_threshold=detection_threshold,
                           **score['macro'])
                rows.append(row)
                candidates.append(row)
        chosen = min(candidates, key=lambda row: (-row['f1'], row['grid_threshold'], row['det_threshold'] or 0))
        best[variant] = dict(grid_threshold=chosen['grid_threshold'])
        if variant == 'matched':
            best[variant]['det_threshold'] = chosen['det_threshold']
    return dict(format_version=1, split='val', objective='selected-class macro F1',
                tie_break='lowest grid threshold, then lowest detection threshold',
                provenance=metadata, **best), rows


def validate_selection(selection, metadata):
    if selection.get('format_version') != 1 or selection.get('split') != 'val':
        raise ValueError('Threshold selection must originate from validation')
    previous = selection['provenance']
    for key, value in metadata.items():
        if key != 'annotation_sha256' and previous.get(key) != value:
            raise ValueError(f'Incompatible validation selection: {key}')
    if not previous.get('annotation_sha256'):
        raise ValueError('Validation selection missing source annotation digest')
    for variant in ('raw', 'matched'):
        if selection[variant]['grid_threshold'] not in GRID_THRESHOLDS:
            raise ValueError('Invalid frozen grid threshold')
    if selection['matched']['det_threshold'] not in DET_THRESHOLDS:
        raise ValueError('Invalid frozen detection threshold')


def box_inputs(coco, records, class_names):
    positive, _ = category_maps(coco, class_names)
    category_ids = {index: cid for cid, index in positive.items()}
    clean = copy.deepcopy(coco)
    clean['categories'] = [cat for cat in clean['categories'] if cat['id'] >= 0]
    clean['annotations'] = [ann for ann in clean['annotations'] if ann['category_id'] >= 0]
    clean.setdefault('info', {})
    for ann in clean['annotations']:
        ann.setdefault('iscrowd', 0)
        ann.setdefault('area', ann['bbox'][2] * ann['bbox'][3])
    predictions = []
    for record in records:
        for box, score, label in zip(record['boxes'], record['scores'], record['labels']):
            x1, y1, x2, y2 = map(float, box)
            predictions.append(dict(image_id=record['image_id'], category_id=category_ids[int(label)],
                                    bbox=[x1, y1, x2 - x1, y2 - y1], score=float(score)))
    return clean, predictions


def evaluate_boxes(coco, records, class_names, classes, backend):
    try:
        if backend == 'pycocotools':
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        else:
            from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    except ImportError as exc:
        raise RuntimeError(f'Install the common box backend in both environments: pip install {backend}') from exc
    clean, predictions = box_inputs(coco, records, class_names)
    ground_truth = COCO()
    ground_truth.dataset = clean
    ground_truth.createIndex()
    if predictions:
        detected = ground_truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(images=clean['images'], categories=clean['categories'], annotations=[])
        detected.createIndex()
    positive, _ = category_maps(coco, class_names)
    result = {}
    for scope, names in (('selected', classes), ('all', class_names)):
        evaluator = COCOeval(ground_truth, detected, 'bbox')
        evaluator.params.imgIds = sorted(im['id'] for im in coco['images'])
        evaluator.params.catIds = [cid for cid, index in positive.items() if class_names[index] in names]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        result[scope] = dict(classes=names, **{key: float(value) if value >= 0 else None
                              for key, value in zip(('ap', 'ap50', 'ap75'), evaluator.stats[:3])})
    return dict(backend=backend, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--coco', required=True)
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sweep', action='store_true')
    mode.add_argument('--selection')
    parser.add_argument('--classes', nargs='+', default=['LC', 'TC'], choices=NAMES)
    parser.add_argument('--box-backend', default='pycocotools', choices=['pycocotools', 'faster-coco-eval'])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if args.sweep and args.split != 'val':
        parser.error('--sweep is permitted only for validation')
    manifest, coco, records = load_export(args.manifest, args.coco, args.split)
    metadata = provenance(manifest, args.classes, args.box_backend)
    if args.sweep:
        selection, sweep = select_thresholds(records, manifest['class_names'], args.classes, metadata)
    else:
        selection = read_json(args.selection)
        validate_selection(selection, metadata)
    boxes = evaluate_boxes(coco, records, manifest['class_names'], args.classes, args.box_backend)
    result = dict(format_version=1, protocol=PROTOCOL, split=args.split, provenance=metadata,
                  image_count=len(records), threshold_selection=selection, boxes=boxes)
    for variant in ('raw', 'matched'):
        result[variant] = score_records(records, manifest['class_names'], args.classes, **selection[variant])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'comparison.json', result)
    if args.sweep:
        write_json(output / 'selection.json', selection)
        write_json(output / 'sweep.json', sweep)
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        rows = [dict(variant=variant, **{key: result[variant][key] for key in ('grid_threshold', 'det_threshold')},
                     **result[variant]['macro'],
                     **{f'box_{key}': boxes['selected'][key] for key in ('ap', 'ap50', 'ap75')})
                for variant in ('raw', 'matched')]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for variant in ('raw', 'matched'):
        metric = result[variant]
        print(f'{variant} ({", ".join(args.classes)}): grid={metric["grid_threshold"]:.2f}, '
              f'detection={metric["det_threshold"]}; ' +
              ', '.join(f'{key}={100 * value:.2f}' for key, value in metric['macro'].items()))
    print('Boxes (' + ', '.join(args.classes) + '): ' + ', '.join(
        f'{key}={100 * boxes["selected"][key]:.2f}' if boxes['selected'][key] is not None else f'{key}=N/A'
        for key in ('ap', 'ap50', 'ap75')))
    print(f'Wrote {output / "comparison.json"}')
    return result


if __name__ == '__main__':
    main()
