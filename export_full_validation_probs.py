from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from torch.nn import functional as F

CLASSES=('antelope_duiker','bird','blank','civet_genet','hog','leopard','monkey_prosimian','rodent')
BASE=Path(__file__).resolve().parent

def main(a):
    split=a.split; run=a.run; device='cuda' if torch.cuda.is_available() else 'cpu'
    train_script=BASE/'train_speciesnet.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('train_speciesnet',train_script); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    dataset=mod.SplitImages(split/'validation')
    pack=torch.load(run/'validation_features_c569598afcdcc068.pt',map_location='cpu',weights_only=True)
    if isinstance(pack,dict):
        X=pack.get('features'); ids=pack.get('ids')
    else:
        X=pack; ids=[row['id'] for row in dataset.rows]
    if X is None or len(X)!=len(dataset): raise ValueError('Cached full-image validation features do not align with validation split')
    if ids is not None and list(ids)!=[row['id'] for row in dataset.rows]: raise ValueError('Feature IDs do not align with split')
    ck=torch.load(run/'best_head.pt',map_location=device,weights_only=True)
    head=torch.nn.Linear(mod.FEATURE_DIM,len(CLASSES)).to(device); head.load_state_dict(ck['state_dict']); head.eval()
    with torch.inference_mode(): probs=F.softmax(head(X.to(device)),1).cpu()
    target=a.output
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['id',*CLASSES]);
        for row,p in zip(dataset.rows,probs): writer.writerow([row['id'],*[f'{float(v):.10g}' for v in p]])
    print(f'Wrote {len(dataset)} aligned validation probabilities to {target}')

p=argparse.ArgumentParser(); p.add_argument('--split',type=Path,required=True); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
