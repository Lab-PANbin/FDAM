"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Lossless, opt-in records for offline single-class crack grid evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class GridPredictionExporter:
    def __init__(self, output_dir, dataset, metadata=None):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        # A failed rerun must not leave an older successful manifest looking current.
        (self.directory / 'manifest.json').unlink(missing_ok=True)
        self.dataset = dataset
        self.expected_ids = set(int(i) for i in dataset.ids)
        if not self.expected_ids:
            raise ValueError('Cannot export an empty evaluation dataset')
        self.records = {}
        self.geometry = None
        self.metadata = dict(metadata or {})
        checkpoint = self.metadata.get('checkpoint')
        if checkpoint and Path(checkpoint).is_file():
            self.metadata['checkpoint_sha256'] = file_sha256(checkpoint)
        self.metadata.update(annotation_file=str(dataset.ann_file),
            annotation_sha256=file_sha256(dataset.ann_file), image_root=str(dataset.img_folder),
            gt_rule='GridEvaluator.box2grid(use_expansion=True, grid_delta=0.125)',
            grid_delta=.125, grid_class=0, detection_class=0,
            coordinates='decoded boxes in original image; grid cells divide original image directly')

    def update(self, probabilities, gt, targets, detections, infer_shape):
        probabilities = as_numpy(probabilities).astype(np.float32)
        gt = as_numpy(gt)
        if probabilities.ndim != 4 or probabilities.shape[1] != 1:
            raise ValueError('Grid export currently supports a single crack channel only')
        if gt.shape != probabilities.shape or not np.isin(gt, [0, 1]).all():
            raise ValueError('GT must be binary and have the same shape as probabilities')
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('Expected finite probabilities in [0,1]')
        if not len(targets) == len(detections) == len(probabilities):
            raise ValueError('Batch lengths differ')
        geometry = (tuple(int(x) for x in infer_shape), tuple(probabilities.shape[-2:]))
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError('Evaluation input/grid shape changed during export')
        self.geometry = geometry
        for probability, truth, target, det in zip(probabilities, gt, targets, detections):
            image_id = int(as_numpy(target['image_id']).reshape(-1)[0])
            if image_id in self.records:
                raise ValueError(f'Duplicate image ID {image_id}; export with one process')
            if image_id not in self.expected_ids:
                raise ValueError(f'Unexpected image ID {image_id}')
            info = self.dataset.coco.imgs[image_id]
            orig_size = as_numpy(target['orig_size']).reshape(-1).astype(np.int64)
            if orig_size.tolist() != [info['width'], info['height']]:
                raise ValueError(f'Original size differs from COCO for image {image_id}')
            boxes = as_numpy(det['boxes']).astype(np.float32).reshape(-1, 4)
            scores = as_numpy(det['scores']).astype(np.float32).reshape(-1)
            labels = as_numpy(det['labels']).astype(np.int64).reshape(-1)
            if not len(boxes) == len(scores) == len(labels):
                raise ValueError('Decoded detection lengths differ')
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError('Non-finite decoded detections')
            filename = f'{image_id}.npz'
            np.savez_compressed(self.directory / filename, probability=probability[0],
                gt=truth[0].astype(np.uint8), boxes=boxes, scores=scores, labels=labels,
                orig_size=orig_size)
            self.records[image_id] = dict(image_id=image_id, file_name=info['file_name'],
                                          record=filename)

    def finish(self, box_metrics=None):
        missing = self.expected_ids - self.records.keys()
        if missing:
            raise ValueError(f'Missing {len(missing)} images in export: {sorted(missing)[:5]}')
        manifest = dict(format_version=1, complete=True, image_count=len(self.records),
            infer_shape=list(self.geometry[0]), grid_shape=list(self.geometry[1]),
            metadata=self.metadata, images=[self.records[i] for i in sorted(self.records)])
        if box_metrics is not None:
            values = {key: float(box_metrics[key]) for key in ('ap', 'ap50', 'ap75')}
            if not all(np.isfinite(v) and -1 <= v <= 1 for v in values.values()):
                raise ValueError('Invalid COCO box metrics')
            # Supplied by the same evaluate() call that generated these predictions.
            manifest['box_metrics'] = values
        path = self.directory / 'manifest.json'
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(f'Exported {len(self.records)} raw grid/GT/detection records to {path}', flush=True)
        return path
