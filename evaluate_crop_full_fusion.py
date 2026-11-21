from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from torch.nn import functional as F

CLASSES=('antelope_duiker','bird','blank','civet_genet','hog','leopard','monkey_prosimian','rodent')

def read_probs(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return {r['id']: torch.tensor([float(r[c]) for c in CLASSES]) for r in rows}

def main(a):
    crop=torch.load(a.features/'validation_crop_features.pt',map_location='cpu',weights_only=True); full=read_probs(a.full); ck=torch.load(a.head,map_location='cpu',weights_only=True)
    head=torch.nn.Linear(1280,8); head.load_state_dict(ck['state_dict']); head.eval()
    with torch.inference_mode(): cp=F.softmax(head(crop['features']),1)
    rows=[]; missing=[]
    for i,rec in enumerate(crop['records']):
        if rec['id'] not in full: missing.append(rec['id']); continue
        fp=full[rec['id']]; p_crop=cp[i] if rec['detected'] else fp
        y=rec['label']
        rows.append((rec,fp,p_crop,y))
    if missing: raise ValueError(f'Missing full probabilities for {len(missing)} IDs')
    alpha=a.crop_weight
    losses=[]; correct=0; predictions=[]
    for rec,fp,cp,y in rows:
        p=fp*(1-alpha)+cp*alpha
        losses.append(float(-torch.log(p[y].clamp_min(1e-8))))
        correct+=int(p.argmax().item()==y)
        predictions.append((rec['id'],p))
    out={'classes':CLASSES,'images':len(rows),'crop_weight':alpha,'full_weight':1-alpha,
         'crop_detected':sum(r['detected'] for r,_,_,_ in rows),
         'crop_detection_coverage':sum(r['detected'] for r,_,_,_ in rows)/len(rows),
         'log_loss':sum(losses)/len(losses),'accuracy':correct/len(rows)}
    a.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    if a.submission:
        with a.submission.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(['id',*CLASSES])
            for image_id,probs in predictions:
                writer.writerow([image_id,*[f'{float(v):.10g}' for v in probs]])
    print(json.dumps(out,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--full',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--crop-weight',type=float,default=.5)
    p.add_argument('--submission',type=Path)
    args=p.parse_args()
    if not 0 <= args.crop_weight <= 1: p.error('--crop-weight must be between 0 and 1')
    main(args)
