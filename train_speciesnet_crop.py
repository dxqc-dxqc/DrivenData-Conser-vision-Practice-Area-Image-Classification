from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ('antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent')
DIM = 1280


def load_features(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    keep = torch.tensor([r['detected'] for r in data['records']], dtype=torch.bool)
    features = data['features'][keep]
    labels = torch.tensor([r['label'] for r in data['records']], dtype=torch.long)[keep]
    if len(features) == 0:
        raise ValueError(f'No animal detections in {path}')
    if features.shape != (len(labels), DIM) or not features.isfinite().all():
        raise ValueError(f'Invalid feature data in {path}')
    return data, features, labels


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; activate .venv-cuda')
    train_data, train_x, train_y = load_features(args.features / 'train_crop_features.pt')
    val_data, val_x, val_y = load_features(args.features / 'validation_crop_features.pt')
    train_x, train_y = train_x.cuda(), train_y.cuda()
    val_x, val_y = val_x.cuda(), val_y.cuda()
    head = nn.Linear(DIM, len(CLASSES)).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device='cuda').manual_seed(42)
    best, history = None, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_x), generator=generator, device='cuda')
        weighted_loss = 0.0
        for offset in range(0, len(train_x), args.batch_size):
            ids = permutation[offset:offset + args.batch_size]
            logits = head(train_x[ids])
            loss = F.cross_entropy(logits, train_y[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss += loss.detach().item() * len(ids)
        head.eval()
        with torch.inference_mode():
            logits = head(val_x)
            val_loss = F.cross_entropy(logits, val_y).item()
            accuracy = (logits.argmax(1) == val_y).float().mean().item()
            cm = torch.bincount(val_y * len(CLASSES) + logits.argmax(1), minlength=len(CLASSES) ** 2)
        metrics = {'epoch': epoch, 'train_loss': weighted_loss / len(train_x),
                   'validation_log_loss_detected_only': val_loss,
                   'validation_accuracy_detected_only': accuracy,
                   'confusion_matrix_detected_only': cm.reshape(len(CLASSES), -1).cpu().tolist()}
        history.append(metrics)
        print(f"epoch {epoch}/{args.epochs}: train_loss={metrics['train_loss']:.4f}, "
              f"detected-only val_log_loss={val_loss:.4f}, accuracy={accuracy:.2%}", flush=True)
        if best is None or val_loss < best['validation_log_loss_detected_only']:
            best = metrics
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        'classes': CLASSES, 'feature_dim': DIM, 'speciesnet_version': '4.0.3a',
                        'detector': 'MDV6-yolov10-e', 'metrics': best}, args.output / 'best_head.pt')
    train_detected = sum(r['detected'] for r in train_data['records'])
    val_detected = sum(r['detected'] for r in val_data['records'])
    report = {'classes': CLASSES, 'trainable_parameters': sum(p.numel() for p in head.parameters()),
              'train_images_total': len(train_data['records']), 'train_images_detected': train_detected,
              'validation_images_total': len(val_data['records']), 'validation_images_detected': val_detected,
              'validation_detection_coverage': val_detected / max(len(val_data['records']), 1),
              'best_detected_only': best, 'epochs': history,
              'head_training_seconds': time.monotonic() - started,
              'note': 'No-detection samples are excluded from head fitting and should fall back to full-image probabilities in fusion.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


parser = argparse.ArgumentParser(description='Train a frozen SpeciesNet 4.0.3a feature head for eight classes.')
parser.add_argument('--features', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.003)
args = parser.parse_args()
if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
    parser.error('epochs, batch-size and learning rate must be positive')
main(args)
