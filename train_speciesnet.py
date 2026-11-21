from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import functional as TF

CLASSES = ('antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard',
           'monkey_prosimian', 'rodent')
FEATURE_DIM = 1280
POOL = 'SpeciesNet/efficientnetv2-m/avg_pool/Mean_Squeeze__3839'
BASE = Path(__file__).resolve().parent


class SplitImages(Dataset):
    def __init__(self, folder: Path):
        self.folder = folder
        self.rows = list(csv.DictReader((folder / 'train_features.csv').open(encoding='utf-8-sig')))
        with (folder / 'train_labels.csv').open(encoding='utf-8-sig') as handle:
            labels = {}
            for row in csv.DictReader(handle):
                targets = [index for index, name in enumerate(CLASSES) if float(row[name]) == 1]
                if len(targets) != 1 or row['id'] in labels:
                    raise ValueError(f"Invalid label row for {row['id']}")
                labels[row['id']] = targets[0]
        self.ids = {row['id'] for row in self.rows}
        self.sites = {row['site'] for row in self.rows}
        if not self.rows or len(self.ids) != len(self.rows) or self.ids != set(labels):
            raise ValueError(f'Duplicate or mismatched image IDs in {folder}')
        self.targets = torch.tensor([labels[row['id']] for row in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.folder / self.rows[index]['filepath']) as image:
            pixels = TF.to_tensor(image.convert('RGB'))
            pixels = TF.resize(pixels, [480, 480], antialias=False)
        return TF.convert_image_dtype(pixels, torch.uint8).permute(1, 2, 0), self.targets[index]


def checkpoint_path(folder: Path) -> Path:
    info = json.loads((folder / 'info.json').read_text(encoding='utf-8'))
    if (info['version'], info['type']) != ('4.0.3b', 'full_image'):
        raise ValueError('Expected SpeciesNet v4.0.3b full-image weights')
    return folder / info['classifier']


def build_backbone(folder: Path) -> torch.fx.GraphModule:
    # The official .pt contains a serialized graph, not just a state dict.
    model = torch.load(checkpoint_path(folder), map_location='cpu', weights_only=False)
    pool = next(node for node in model.graph.nodes if node.op == 'call_module' and node.target == POOL)
    output = next(node for node in model.graph.nodes if node.op == 'output')
    output.args = (pool,)
    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.delete_all_unused_submodules()
    model.recompile()
    return model.eval().requires_grad_(False)


def new_head() -> nn.Linear:
    return nn.Linear(FEATURE_DIM, len(CLASSES))


def cache_key(dataset: SplitImages, weights: Path) -> str:
    digest = hashlib.sha256(f'pool-v1-fp16:{weights.resolve()}:{weights.stat().st_size}:{weights.stat().st_mtime_ns}'.encode())
    for name in ('train_features.csv', 'train_labels.csv'):
        digest.update((dataset.folder / name).read_bytes())
    return digest.hexdigest()[:16]


def save_torch(value, path: Path) -> None:
    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        torch.save(value, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def extract(model, batches, label, total):
    parts = []
    start = time.monotonic()
    done = 0
    with torch.inference_mode():
        for images, _ in batches:
            pixels = images.cuda(non_blocking=True).float().div_(255)
            with torch.autocast('cuda', dtype=torch.float16):
                features = model(pixels)
            features = features.float().cpu()
            if features.shape != (len(images), FEATURE_DIM) or not features.isfinite().all():
                raise RuntimeError(f'Bad features for {label} at image {done}')
            parts.append(features)
            done += len(images)
            if len(parts) in (1, 8) or len(parts) % 64 == 0 or done == total:
                elapsed = time.monotonic() - start
                print(f'{label}: {done}/{total} images, {done / elapsed:.1f} images/s, {elapsed:.1f}s', flush=True)
    return torch.cat(parts)


def train_head(train, validation, output, epochs):
    features, labels = train
    val_features, val_labels = validation
    torch.manual_seed(42)
    device = torch.device('cuda')
    head = new_head().to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    print(f'Head training and validation on {torch.cuda.get_device_name(device)} (CUDA)', flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
    best_loss = float('inf')
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = torch.zeros((), device=device)
        for batch_features, targets in loader:
            batch_features = batch_features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(targets)
        head.eval()
        with torch.no_grad():
            logits = head(val_features)
            val_loss = F.cross_entropy(logits, val_labels).item()
            accuracy = (logits.argmax(1) == val_labels).float().mean().item()
        metrics = dict(epoch=epoch, train_loss=total_loss.item() / len(labels), val_log_loss=val_loss, val_accuracy=accuracy)
        history.append(metrics)
        print(f'Epoch {epoch}/{epochs} train_loss={metrics["train_loss"]:.4f} '
              f'val_log_loss={val_loss:.4f} val_accuracy={accuracy:.2%}', flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            confusion = torch.bincount(val_labels * len(CLASSES) + logits.argmax(1), minlength=len(CLASSES)**2)
            best = {**metrics, 'confusion_matrix': confusion.view(len(CLASSES), -1).tolist()}
            state = {name: tensor.cpu() for name, tensor in head.state_dict().items()}
            save_torch({'state_dict': state, 'classes': CLASSES, 'feature_dim': FEATURE_DIM,
                        'speciesnet_version': '4.0.3b', 'feature_node': POOL, 'metrics': best}, output / 'best_head.pt')
    report = {'classes': CLASSES, 'train_images': len(labels), 'validation_images': len(val_labels),
              'trainable_parameters': sum(p.numel() for p in head.parameters()), 'head_device': device.type,
              'best': best, 'epochs': history, 'head_training_seconds': time.monotonic() - start}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def monitor_gpu(stop: threading.Event):
    command = [
        'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits',
    ]
    while not stop.is_set():
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True,
                                    timeout=3, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for index, utilization, used, total in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                print(f'[{time.strftime("%H:%M:%S")}] GPU {index} | Utilization {utilization}% | VRAM {used}/{total} MiB', flush=True)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print(f'GPU monitoring unavailable: {error}', flush=True)
            return
        stop.wait(1)


def main():
    parser = argparse.ArgumentParser(description='Freeze SpeciesNet and train an 8-class head on a site split.')
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=BASE / 'models' / 'speciesnet-v4.0.3b')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--benchmark', type=int, default=0, metavar='IMAGES')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.benchmark < 0:
        parser.error('Expected positive batch size and epoch count, and a nonnegative benchmark size')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run SpeciesNet training. Use the .venv-cuda environment.')
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop,), daemon=True)
    monitor.start()
    try:
        run_training(args)
    finally:
        stop.set()
        monitor.join()


def run_training(args):
    torch.set_num_threads(2)
    datasets = {name: SplitImages(args.split / name) for name in ('train', 'validation')}
    train, validation = datasets.values()
    if train.ids & validation.ids or train.sites & validation.sites:
        raise ValueError('Train and validation contain overlapping images or sites')
    if args.benchmark:
        sample_size = min(args.benchmark, len(train))
        indices = [index * len(train) // sample_size for index in range(sample_size)]
        sample = torch.utils.data.Subset(train, indices)
        batches = DataLoader(sample, batch_size=args.batch_size, pin_memory=True)
        extract(build_backbone(args.model).cuda(), batches, 'benchmark', len(sample))
        return
    output = args.split / 'speciesnet_v4.0.3b'
    output.mkdir(exist_ok=True)
    weights = checkpoint_path(args.model)
    backbone = None
    tensors = {}
    for name, dataset in datasets.items():
        cache = output / f'{name}_features_{cache_key(dataset, weights)}.pt'
        if cache.exists():
            features = torch.load(cache, weights_only=True)
            print(f'{name}: reusing {cache.name}', flush=True)
        else:
            if backbone is None:
                backbone = build_backbone(args.model).cuda()
            features = extract(backbone, DataLoader(dataset, batch_size=args.batch_size, pin_memory=True), name, len(dataset))
            save_torch(features, cache)
        if features.shape != (len(dataset), FEATURE_DIM) or not features.isfinite().all():
            raise ValueError(f'Invalid cached feature matrix: {cache}')
        tensors[name] = (features, dataset.targets)
    del backbone
    train_head(tensors['train'], tensors['validation'], output, args.epochs)
    print(f'Saved: {output / "best_head.pt"}', flush=True)


if __name__ == '__main__':
    main()
