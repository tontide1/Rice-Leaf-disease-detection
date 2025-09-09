# split_dataset.py
import os, shutil, random
from pathlib import Path
from PIL import Image
import imagehash

SRC = Path("data/Rice_Leaf_Disease_Images")
DST = Path("data/splits")
CLASSES = ["brown_spot","blast","bacterial_leaf_blight","normal"]
SPLITS = {"train":0.7, "val":0.15, "test":0.15}
EXTS = {".jpg",".jpeg",".png"}
SEED=42; random.seed(SEED)

def list_images(p): 
    return [q for q in p.rglob("*") if q.suffix.lower() in EXTS]

def group_by_phash(files, thresh=5):
    buckets=[]
    for f in files:
        try:
            h=imagehash.phash(Image.open(f).convert("RGB"), hash_size=16)
        except Exception:
            continue
        for b in buckets:
            if abs(h - b["h"]) <= thresh:
                b["fs"].append(f); break
        else:
            buckets.append({"h":h,"fs":[f]})
    return buckets

for c in CLASSES:
    files = list_images(SRC/c)
    buckets = group_by_phash(files, thresh=5)
    random.shuffle(buckets)

    n=len(buckets); n_tr=int(n*SPLITS["train"]); n_val=int(n*SPLITS["val"])
    parts={"train":buckets[:n_tr],
           "val":  buckets[n_tr:n_tr+n_val],
           "test": buckets[n_tr+n_val:]}

    for sp, bks in parts.items():
        out = DST/sp/c; out.mkdir(parents=True, exist_ok=True)
        for bk in bks:
            for fp in bk["fs"]:
                shutil.copy2(fp, out/fp.name)

# Thống kê nhanh
for sp in ["train","val","test"]:
    for c in CLASSES:
        cnt=len(list((DST/sp/c).glob("*")))
        print(f"{sp}/{c}: {cnt}")