#!/usr/bin/env python
"""Stera-10M 切片下载：只下 annotation.hdf5 + calibrations（不下 rgb.mp4 / *.rrd / mesh.ply）。
gated：先在 https://huggingface.co/datasets/fpvlabs/stera-10m 同意条款，并设置 HF_TOKEN。
按 annotation.hdf5 大小从小到大取，直到 --max-gb（默认 40 GB，适配 Kaggle 数据集）。
"""
import argparse, json
from huggingface_hub import HfApi, hf_hub_download
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/stera"); ap.add_argument("--max-gb", type=float, default=40)
a = ap.parse_args()
info = HfApi().dataset_info("fpvlabs/stera-10m", files_metadata=True)
ann = sorted([(s.size, s.rfilename) for s in info.siblings if s.rfilename.endswith("annotation.hdf5")])
pick, tot = [], 0
for size, fn in ann:
    if tot + size > a.max_gb * 1e9:
        break
    pick.append(fn); tot += size
print(len(ann), "sessions; picking", len(pick), "=", round(tot / 1e9, 1), "GB", flush=True)
for fn in pick:
    sess = fn.rsplit("/", 1)[0]
    for f in (fn, sess + "/calibrations/R_optical_to_link.npy", sess + "/calibrations/meta.json"):
        try:
            hf_hub_download("fpvlabs/stera-10m", f, repo_type="dataset", local_dir=a.out)
        except Exception as e:  # noqa: BLE001
            print("skip", f, repr(e)[:120], flush=True)
json.dump(pick, open(a.out + "/stera_slice.json", "w"), indent=1)
