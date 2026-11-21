from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
import threading
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

CLASSES = ('antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent')
DIM = 1280
BASE = Path(__file__).resolve().parent


def rows(folder):
    with (folder / 'train_features.csv').open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def label_map(folder):
    with (folder / 'train_labels.csv').open(encoding='utf-8-sig', newline='') as f:
        result = {}
        for row in csv.DictReader(f):
            selected = [i for i, name in enumerate(CLASSES) if float(row[name]) == 1]
            if len(selected) != 1 or row['id'] in result:
                raise ValueError(f"Expected one-hot label for {row['id']}")
            result[row['id']] = selected[0]
    return result


def build_backbone(folder):
    info = json.loads((folder / 'info.json').read_text(encoding='utf-8'))
    if (info['version'], info['type']) != ('4.0.3a', 'always_crop'):
        raise ValueError(f'Expected SpeciesNet 4.0.3a always_crop, got {info}')
    model = torch.load(folder / info['classifier'], map_location='cpu', weights_only=False)
    target = next(n.target for n in model.graph.nodes if n.op == 'call_module' and 'avg_pool/Mean_Squeeze' in str(n.target))
    pool = next(n for n in model.graph.nodes if n.op == 'call_module' and n.target == target)
    output = next(n for n in model.graph.nodes if n.op == 'output')
    output.args = (pool,)
    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.delete_all_unused_submodules()
    model.recompile()
    return model.eval().requires_grad_(False), info, target


def crop_tensor(image, box):
    width, height = image.size
    x1, y1, x2, y2 = box
    x1p, y1p = int(x1 * width), int(y1 * height)
    wp, hp = max(1, int((x2 - x1) * width)), max(1, int((y2 - y1) * height))
    tensor = TF.convert_image_dtype(TF.pil_to_tensor(image.convert('RGB')), torch.float32)
    tensor = TF.crop(tensor, y1p, x1p, hp, wp)
    tensor = TF.resize(tensor, [480, 480], antialias=False)
    return TF.convert_image_dtype(tensor, torch.uint8).permute(1, 2, 0)


def cache_key(split, detector, model):
    h = hashlib.sha256()
    h.update(f'{detector.resolve()}:{detector.stat().st_size}:{detector.stat().st_mtime_ns}'.encode())
    h.update(f'{model.resolve()}:{(model / "info.json").stat().st_mtime_ns}'.encode())
    for name in ('train_features.csv', 'train_labels.csv'):
        h.update((split / name).read_bytes())
    return h.hexdigest()[:16]


def detect_batch(detector, images):
    results = detector.predict(source=[np.asarray(im) for im in images], conf=0.2, imgsz=1280, classes=[0], verbose=False, device=0)
    records = []
    for image, result in zip(images, results):
        if len(result.boxes) == 0:
            records.append((None, 0.0))
            continue
        index = int(torch.argmax(result.boxes.conf).item())
        x1, y1, x2, y2 = result.boxes.xyxy[index].cpu().tolist()
        confidence = float(result.boxes.conf[index])
        width, height = image.size
        dx, dy = (x2 - x1) * 0.12, (y2 - y1) * 0.12
        box = (max(0, x1 - dx) / width, max(0, y1 - dy) / height,
               min(width, x2 + dx) / width, min(height, y2 + dy) / height)
        records.append((box, confidence))
    return records


def run_split(name, split, detector, backbone, out, cache_id, limit, batch):
    cache_path = out / f'{name}_crop_features_{cache_id}.pt'
    partial_path = out / f'{name}_crop_partial_{cache_id}.pt'
    if cache_path.exists():
        print(f'{name}: reusing {cache_path.name}', flush=True)
        return
    source_rows = rows(split)
    labels = label_map(split)
    if limit:
        source_rows = source_rows[:limit]
    if set(r['id'] for r in source_rows) - labels.keys():
        raise ValueError(f'Missing labels in {split}')
    records, feature_rows = [], []
    if partial_path.exists():
        partial = torch.load(partial_path, map_location='cpu', weights_only=True)
        records = partial['records']
        feature_rows = list(partial['features'])
        if len(records) != len(feature_rows) or len(records) > len(source_rows):
            raise ValueError(f'Invalid partial cache: {partial_path}')
        for old, new in zip(records, source_rows):
            if old['id'] != new['id']:
                raise ValueError(f"Partial cache does not match current split at {old['id']}")
        print(f'{name}: resuming from {len(records)}/{len(source_rows)}', flush=True)
    started = time.monotonic()
    for offset in range(len(records), len(source_rows), batch):
        current = source_rows[offset:offset + batch]
        images = []
        for row in current:
            with Image.open(split / row['filepath']) as im:
                images.append(im.convert('RGB').copy())
        detections = detect_batch(detector, images)
        crop_positions, crop_inputs = [], []
        for local, (row, image, (box, confidence)) in enumerate(zip(current, images, detections)):
            index = offset + local
            record = {'id': row['id'], 'filepath': row['filepath'], 'site': row['site'],
                      'label': labels[row['id']], 'detected': box is not None,
                      'confidence': confidence, 'bbox': list(box) if box else None}
            records.append(record)
            if box is None:
                feature_rows.append(torch.zeros(DIM))
            else:
                crop_positions.append(index)
                crop_inputs.append(crop_tensor(image, box))
                feature_rows.append(None)
        if crop_inputs:
            inputs = torch.stack(crop_inputs).cuda(non_blocking=True).float().div_(255)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                features = backbone(inputs).float().cpu()
            if features.shape != (len(crop_positions), DIM) or not features.isfinite().all():
                raise RuntimeError(f'Invalid features from {name} batch at offset {offset}')
            for index, feature in zip(crop_positions, features):
                feature_rows[index] = feature
        partial_payload = {'features': torch.stack(feature_rows), 'records': records, 'classes': CLASSES, 'model': 'SpeciesNet v4.0.3a', 'feature_dim': DIM, 'cache_key': cache_id, 'processed': len(records)}
        temp_partial = partial_path.with_suffix('.tmp')
        torch.save(partial_payload, temp_partial)
        temp_partial.replace(partial_path)
        done = len(records)
        if done == len(source_rows) or done % max(batch * 8, 1) == 0:
            elapsed = time.monotonic() - started
            detected = sum(r['detected'] for r in records)
            print(f'{name}: {done}/{len(source_rows)} images, detected={detected}, {done / max(elapsed, 1e-6):.2f} images/s, {elapsed:.1f}s', flush=True)
    features = torch.stack(feature_rows) if feature_rows else torch.empty((0, DIM))
    payload = {'features': features, 'records': records, 'classes': CLASSES,
               'model': 'SpeciesNet v4.0.3a', 'feature_dim': DIM,
               'cache_key': cache_id, 'processed': len(source_rows)}
    temp = cache_path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(cache_path)
    partial_path.unlink(missing_ok=True)
    latest = out / f'{name}_crop_features.pt'
    latest.write_bytes(cache_path.read_bytes())
    det_path = out / f'{name}_detections.json'
    det_path.write_text(json.dumps(records, indent=2), encoding='utf-8')


def monitor_gpu(stop):
    command = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']
    while not stop.is_set():
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for line in result.stdout.splitlines():
                print(f"[{time.strftime('%H:%M:%S')}] GPU | {line}", flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            print(f'GPU monitoring unavailable: {error}', flush=True)
            return
        stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description='Detect and extract SpeciesNet 4.0.3a crop features.')
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--detector', type=Path, default=BASE / 'models/megadetector/MDV6-yolov10-e-1280.pt')
    parser.add_argument('--model', type=Path, default=BASE / 'models/speciesnet-v4.0.3a')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error('--limit must be nonnegative and --batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; activate .venv-cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    backbone, info, target = build_backbone(args.model)
    backbone = backbone.cuda()
    detector = YOLO(str(args.detector))
    key = cache_key(args.split / 'train', args.detector, args.model) + (f'-n{args.limit}' if args.limit else '-full')
    print(f"SpeciesNet {info['version']} feature_node={target}; detector=MDV6-yolov10-e; cache={key}", flush=True)
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop,), daemon=True)
    monitor.start()
    try:
        for name in ('train', 'validation'):
            run_split(name, args.split / name, detector, backbone, args.output, key,
                      args.limit, args.batch_size)
    finally:
        stop.set()
        monitor.join()


if __name__ == '__main__':
    main()
