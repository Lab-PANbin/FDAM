#!/usr/bin/env python3
"""FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

Export original FDAM / GCBD checkpoints into the road-grid-v1 contract.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image, __version__ as pillow_version


CLASS_NAMES = ['LC', 'TC', 'LR', 'SR']
ALIASES = {'纵向裂缝': 'LC', '横向裂缝': 'TC', '修补.块状': 'LR', '修补.条状': 'SR',
           '修补块状': 'LR', '修补条状': 'SR'}
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
BOX_GEOMETRY_POLICY = 'clamp_negative_extents_to_zero_keep_scores_labels_and_ranking'
SIGNATURE_KEYS = ('backend', 'checkpoint_sha256', 'model_source', 'class_names',
                  'infer_shape', 'grid_shape', 'preprocessing', 'decoder',
                  'config_sha256', 'inference_source_sha256', 'runtime')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inference_signature(metadata):
    return digest_json({key: metadata[key] for key in SIGNATURE_KEYS})


def canonical_name(name):
    text = str(name).strip()
    return ALIASES.get(text, text.upper())


def validate_names(names, expected):
    """Checkpoint names dictionaries may include both positive and negative IDs."""
    if isinstance(names, dict):
        by_id = {int(key): canonical_name(value) for key, value in names.items()}
        positive = {key: value for key, value in by_id.items() if key >= 0}
        if set(positive) != set(range(4)):
            raise ValueError(f'Checkpoint detection class IDs must be 0..3: {positive}')
        ordered = [positive[i] for i in range(4)]
        for key, name in by_id.items():
            if key < 0 and (key < -4 or name != expected[-key - 1]):
                raise ValueError(f'Checkpoint grid class order mismatch: {by_id}')
    else:
        ordered = [canonical_name(name) for name in names]
    if ordered != list(expected):
        raise ValueError(f'Checkpoint class order {ordered} differs from --class-names {expected}')


def validate_categories(categories, expected):
    if len(expected) != 4 or set(expected) != set(CLASS_NAMES):
        raise ValueError('The protocol requires exactly LC, TC, LR, SR')
    by_id = {int(cat['id']): canonical_name(cat['name']) for cat in categories}
    if len(by_id) != len(categories) or set(by_id) != set(range(-4, 4)):
        raise ValueError('COCO must define unique box IDs 0..3 and grid IDs -1..-4')
    validate_names(by_id, expected)


def select_source(checkpoint, requested):
    if not isinstance(checkpoint, dict) or requested not in ('auto', 'model', 'ema'):
        raise ValueError('Expected checkpoint mapping and model|ema|auto source')
    source = ('ema' if checkpoint.get('ema') is not None else 'model') if requested == 'auto' else requested
    if checkpoint.get(source) is None:
        raise ValueError(f'Requested checkpoint source {source!r} is missing or None')
    return source, checkpoint[source]


def restore_boxes(boxes, orig_size, image_size):
    width, height = orig_size
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * np.asarray(
        [width, height, width, height], dtype=np.float32) / image_size


def make_record(probability, boxes, scores, labels, orig_size):
    probability = np.asarray(probability, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    scores, labels = np.asarray(scores, dtype=np.float32), np.asarray(labels)
    size = np.asarray(orig_size)
    if probability.shape != (4, 32, 32):
        raise ValueError(f'Expected probability shape (4,32,32), got {probability.shape}; no interpolation allowed')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Grid probability must be finite and in [0,1]')
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),) or labels.shape != scores.shape:
        raise ValueError('Expected boxes Nx4, scores N and labels N')
    nonfinite = np.flatnonzero(~np.isfinite(boxes).all(axis=1))
    if len(nonfinite):
        sample = nonfinite[:3]
        raise ValueError(f'Non-finite decoded boxes (NaN/Inf): rows={sample.tolist()}, '
                         f'boxes={boxes[sample].tolist()}, scores={scores[sample].tolist()}')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Detection scores must be finite and in [0,1]')
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels >= 4) | (labels != np.floor(labels))):
        raise ValueError('Detection labels must be integral zero-based indices 0..3')
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size != np.floor(size)):
        raise ValueError('orig_size must contain positive integer width, height')
    # DEIM's signed distance regression can produce finite negative extents.
    # Keep these detections (including scores/ranking) as zero-area COCO inputs.
    # Swapping corners would manufacture valid boxes; dropping them could improve AP.
    decoded_boxes = boxes.copy()
    boxes = boxes.copy()
    boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2])
    return dict(probability=probability, boxes=boxes, decoded_boxes=decoded_boxes, scores=scores,
                labels=labels.astype(np.int64), orig_size=size.astype(np.int64))


def load_image(path, image_size, backend):
    with Image.open(path) as image:
        orig_size = image.size
        image = image.convert('RGB').resize((image_size, image_size), Image.Resampling.BILINEAR)
        data = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.
    if backend == 'fdam':
        data = (data - np.asarray(MEAN, np.float32)[:, None, None]) / np.asarray(STD, np.float32)[:, None, None]
    return np.ascontiguousarray(data), orig_size


def atomic_json(path, content):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


class ExportWriter:
    """Invalidate a previous manifest first; publish a fresh run only on success."""
    def __init__(self, directory):
        self.root, self.images = Path(directory), []
        self.geometry = dict(total_detections=0, nonpositive_detections=0,
                             images_with_nonpositive=0, max_nonpositive_score=None)
        self.run_id = uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / 'manifest.json', dict(format_version=1, protocol='road-grid-v1',
                                                     complete=False, run_id=self.run_id))
        self.records = self.root / 'records' / self.run_id
        self.records.mkdir(parents=True)

    def write(self, image, record):
        path = self.records / f'{len(self.images):08d}.npz'
        np.savez_compressed(path, **record)
        decoded_boxes = record.get('decoded_boxes', record['boxes'])
        nonpositive = np.any(decoded_boxes[:, 2:] <= decoded_boxes[:, :2], axis=1)
        count = int(nonpositive.sum())
        self.geometry['total_detections'] += len(decoded_boxes)
        self.geometry['nonpositive_detections'] += count
        if count:
            self.geometry['images_with_nonpositive'] += 1
            score = float(record['scores'][nonpositive].max())
            previous = self.geometry['max_nonpositive_score']
            self.geometry['max_nonpositive_score'] = score if previous is None else max(previous, score)
            if self.geometry['images_with_nonpositive'] <= 5:
                print(f'Box geometry: image_id={image["id"]}, file={image["file_name"]}, '
                      f'{count}/{len(decoded_boxes)} nonpositive boxes retained with zero area, '
                      f'max_score={score:.6f}', flush=True)
        self.images.append(dict(image_id=image['id'], file_name=image['file_name'],
                                record=path.relative_to(self.root).as_posix(),
                                nonpositive_box_count=count))

    def finish(self, metadata, expected_ids):
        ids = [image['image_id'] for image in self.images]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(expected_ids):
            raise ValueError('Incomplete or duplicate image coverage')
        manifest = dict(metadata, format_version=1, protocol='road-grid-v1', complete=True,
                        image_count=len(ids), images=self.images, run_id=self.run_id,
                        detection_geometry_summary=self.geometry)
        atomic_json(self.root / 'manifest.json', manifest)
        print(f'Box geometry totals: {json.dumps(self.geometry)}', flush=True)


def trusted_load(torch, path):
    kwargs = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **kwargs)


def import_project(module_name, root):
    sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError(f'{module_name} was imported from {module.__file__}, outside --model-root')
    return module


def validate_fdam_preprocessing(config):
    ops = config['val_dataloader']['dataset']['transforms']['ops']
    if [op['type'] for op in ops] != ['Resize', 'ConvertPILImage', 'Normalize', 'ConvertBoxes']:
        raise ValueError('Unsupported FDAM validation transforms; expected direct resize and ImageNet normalization')
    if ops[1].get('dtype') != 'float32' or ops[1].get('scale') is not True:
        raise ValueError('FDAM must convert RGB to float32 /255')
    if ops[2].get('mean') != MEAN or ops[2].get('std') != STD:
        raise ValueError('FDAM normalization differs from the road-grid-v1 ImageNet settings')


class FDAMAdapter:
    def __init__(self, args, torch):
        core = import_project('engine.core', args.model_root)
        config = core.YAMLConfig(str(args.config))
        cfg = config.yaml_cfg
        self.config_sha256 = digest_json(cfg)  # Includes recursively merged YAML inputs.
        validate_fdam_preprocessing(cfg)
        if cfg.get('num_classes') != 4 or cfg.get('remap_mscoco_category', False):
            raise ValueError('FDAM requires four classes and remap_mscoco_category=False')
        if cfg.get('DEIM', {}).get('grid_head', {}).get('num_classes') != 4:
            raise ValueError('FDAM grid head must have four channels')
        checkpoint = trusted_load(torch, args.checkpoint)
        self.source, state = select_source(checkpoint, args.model_source)
        if self.source == 'ema':
            if not isinstance(state, dict) or 'module' not in state:
                raise ValueError('FDAM EMA checkpoint must contain ema.module')
            state = state['module']
        self.class_order_evidence = ['explicit --class-names', 'COCO category IDs (training uses IDs directly)']
        if checkpoint.get('names') is not None:
            validate_names(checkpoint['names'], args.class_names)
            self.class_order_evidence.append('checkpoint.names')
        if 'DINOv3STAs' in cfg:
            cfg['DINOv3STAs']['weights_path'] = None
        if 'HGNetv2' in cfg:
            cfg['HGNetv2']['pretrained'] = False
        self.model = config.model
        self.model.load_state_dict(state, strict=True)
        print(f'Loaded full FDAM checkpoint strictly: {args.checkpoint} ({self.source})', flush=True)
        # Keep checkpoint anchor buffers for strict loading; use dynamic geometry afterwards.
        for module in self.model.modules():
            if hasattr(module, 'eval_spatial_size'):
                module.eval_spatial_size = None
        self.model = self.model.to(args.device).float().eval()
        self.post = config.postprocessor.to(args.device).eval()
        self.decoder = dict(type=type(self.post).__name__, num_top_queries=self.post.num_top_queries,
                            use_focal_loss=self.post.use_focal_loss, remap_mscoco_category=False,
                            grid_activation='sigmoid', dynamic_spatial_size=True, nms=None,
                            box_geometry=BOX_GEOMETRY_POLICY,
                            score_floor=None, coordinate_scale='original_width_height')

    def predict(self, tensor, sizes, torch):
        output = self.model(tensor)
        if output.get('pred_grid_logits') is None:
            raise ValueError('FDAM did not produce pred_grid_logits')
        probabilities = output['pred_grid_logits'].sigmoid().cpu().numpy()
        decoded = self.post(output, torch.as_tensor(sizes, device=tensor.device))
        if len(decoded) != len(sizes) or len(probabilities) != len(sizes):
            raise ValueError('FDAM output batch size differs from input')
        return [make_record(p, d['boxes'].cpu().numpy(), d['scores'].cpu().numpy(),
                            d['labels'].cpu().numpy(), size)
                for p, d, size in zip(probabilities, decoded, sizes)]


class GCBDAdapter:
    def __init__(self, args, torch):
        import_project('models.yolo', args.model_root)
        nms_module = import_project('utils.patch_classify.general', args.model_root)
        self.nms = nms_module.non_max_suppression
        self.source, model = select_source(trusted_load(torch, args.checkpoint), args.model_source)
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Original GCBD checkpoint must store a torch model object')
        if getattr(model, 'names', None) is None:
            raise ValueError('GCBD checkpoint is missing class names; class order cannot be verified')
        validate_names(model.names, args.class_names)
        head = model.model[-1]
        if getattr(head, 'nc', None) != 4 or getattr(head, 'nc_grid', None) != 4:
            raise ValueError('Expected original GCBD PatchClassify head with nc=nc_grid=4')
        self.class_order_evidence = ['checkpoint model.names', 'COCO category IDs']
        self.config_sha256 = digest_json(model.yaml)
        self.model = model.to(args.device).float().eval()
        self.image_size = args.image_size
        self.decoder = dict(type='original_gcbd_non_max_suppression', conf_thres=.001,
                            iou_thres=.5, max_det=300, agnostic=False, multi_label=True,
                            grid_activation='model_sigmoid', augment=False, ars=False,
                            nms_batch_size=1, coordinate_scale='original_width_height')
        self.decoder['box_geometry'] = BOX_GEOMETRY_POLICY

    def predict(self, tensor, sizes, torch):
        output = self.model(tensor, augment=False)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise ValueError('Original GCBD must return (decoded_predictions, grid_probabilities, ...)')
        predictions, probabilities = output[:2]
        if predictions.ndim != 3 or predictions.shape[-1] != 9:
            raise ValueError('GCBD detection output must have 5+4 channels')
        if len(predictions) != len(sizes) or len(probabilities) != len(sizes):
            raise ValueError('GCBD output batch size differs from input')
        records = []
        for i, size in enumerate(sizes):
            # Native NMS can terminate a batch early on timeout; one image cannot be skipped.
            det = self.nms(predictions[i:i + 1], conf_thres=.001, iou_thres=.5,
                           agnostic=False, multi_label=True, max_det=300)[0].cpu().numpy()
            records.append(make_record(probabilities[i].cpu().numpy(),
                           restore_boxes(det[:, :4], size, self.image_size), det[:, 4], det[:, 5], size))
        return records


def source_fingerprints(root, backend):
    folders = ['engine'] if backend == 'fdam' else ['models', 'utils']
    files = {path.relative_to(root).as_posix(): sha256_file(path)
             for folder in folders for path in sorted((root / folder).rglob('*.py'))}
    files['shared_export_adapter'] = sha256_file(__file__)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['fdam', 'gcbd'])
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='FDAM YAML config, relative to model-root or absolute')
    parser.add_argument('--checkpoint', required=True, type=Path, help='Trusted checkpoint only (Python pickle)')
    parser.add_argument('--model-source', choices=['model', 'ema', 'auto'], default='auto')
    parser.add_argument('--coco', required=True, type=Path)
    parser.add_argument('--image-root', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--image-size', type=int, default=1024, choices=[1024])
    parser.add_argument('--class-names', nargs=4, default=CLASS_NAMES)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.backend == 'fdam' and args.config is None):
        parser.error('Positive --batch-size and FDAM --config are required')
    writer = ExportWriter(args.output_dir)
    args.model_root, args.image_root = args.model_root.resolve(), args.image_root.resolve()
    if args.config is not None:
        args.config = args.config if args.config.is_absolute() else args.model_root / args.config
    args.class_names = [canonical_name(name) for name in args.class_names]
    annotations = json.loads(args.coco.read_text(encoding='utf-8-sig'))
    validate_categories(annotations['categories'], args.class_names)
    images = annotations['images']
    ids = [image['id'] for image in images]
    if not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('COCO images must have nonempty unique integer image IDs')
    import torch
    import torchvision
    adapter = (FDAMAdapter if args.backend == 'fdam' else GCBDAdapter)(args, torch)
    source_files = source_fingerprints(args.model_root, args.backend)
    metadata = dict(backend=args.backend, checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=sha256_file(args.checkpoint), model_source=adapter.source,
                    requested_model_source=args.model_source, annotation_sha256=sha256_file(args.coco),
                    split=args.split, infer_shape=[1024, 1024], grid_shape=[32, 32],
                    class_names=args.class_names, class_order_evidence=adapter.class_order_evidence,
                    preprocessing=dict(color='RGB', resize='direct_square', padding=False,
                                       interpolation='PIL_BILINEAR', scale=1 / 255,
                                       mean=MEAN if args.backend == 'fdam' else [0, 0, 0],
                                       std=STD if args.backend == 'fdam' else [1, 1, 1]),
                    decoder=adapter.decoder, config_sha256=adapter.config_sha256,
                    inference_source_sha256=digest_json(source_files), source_files=source_files,
                    runtime=dict(precision='float32', autocast=False, batch_size=args.batch_size,
                                 torch=torch.__version__, torchvision=torchvision.__version__,
                                 numpy=np.__version__, pillow=pillow_version))
    metadata['inference_signature'] = inference_signature(metadata)
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = images[start:start + args.batch_size]
            arrays, sizes = [], []
            for image in batch:
                path = (args.image_root / image['file_name']).resolve()
                if not path.is_relative_to(args.image_root):
                    raise ValueError(f'Image path escapes --image-root: {image["file_name"]}')
                array, size = load_image(path, args.image_size, args.backend)
                if size != (image['width'], image['height']):
                    raise ValueError(f'COCO dimensions differ from image {image["id"]}: {size}')
                arrays.append(array)
                sizes.append(size)
            tensor = torch.from_numpy(np.stack(arrays)).to(args.device)
            try:
                records = adapter.predict(tensor, sizes, torch)
            except ValueError as exc:
                context = [dict(image_id=image['id'], file_name=image['file_name']) for image in batch]
                raise ValueError(f'{args.backend} export failed for batch {context}: {exc}') from exc
            if len(records) != len(batch):
                raise ValueError('Adapter returned an incomplete batch')
            for image, record in zip(batch, records):
                writer.write(image, record)
            print(f'Exported {start + len(batch)}/{len(images)} {args.backend} {args.split}', flush=True)
    writer.finish(metadata, ids)
    print(f'Complete manifest: {writer.root / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
