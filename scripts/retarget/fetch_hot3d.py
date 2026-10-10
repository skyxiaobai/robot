# -*- coding: utf-8 -*-
"""下载一小部分 HOT3D-Clips（Quest3）和物体包围盒。

只要 json：手腕、物体位姿、info。图像和 tar 解完就删，不进 git。
默认 16 段（clip-000000 到 clip-000015），每段 tar 约 100 MB。

    python scripts/retarget/fetch_hot3d.py --dest /tmp/hot3d
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main"
MODELS = HF + "/object_models/models_info.json"


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print("download", url, flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def _extract_json(tar_path, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    suffixes = (".hands.json", ".objects.json", ".info.json")
    with tarfile.open(tar_path, "r") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if not name.endswith(suffixes):
                continue
            target = dest / name
            source = archive.extractfile(member)
            if source is None:
                continue
            target.write_bytes(source.read())
    return dest


def fetch(dest, count=16, start=0):
    dest = Path(dest)
    clips = dest / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    models = _download(MODELS, dest / "models_info.json")
    written = []
    for index in range(int(start), int(start) + int(count)):
        name = "clip-%06d" % index
        out = clips / name
        if list(out.glob("*.hands.json")):
            written.append(str(out))
            print("have", name, flush=True)
            continue
        url = "%s/train_quest3/%s.tar" % (HF, name)
        tar_path = dest / "tars" / ("%s.tar" % name)
        try:
            _download(url, tar_path)
        except Exception as exc:  # noqa: BLE001 - 把下载失败写进调用方的报告
            print("fail", name, exc, flush=True)
            continue
        _extract_json(tar_path, out)
        tar_path.unlink(missing_ok=True)
        n_hands = len(list(out.glob("*.hands.json")))
        print("ok", name, n_hands, flush=True)
        written.append(str(out))
    return {"models_info": str(models), "clips": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载 HOT3D-Clips 的一小部分 json")
    parser.add_argument("--dest", default="/tmp/hot3d")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args(argv)
    result = fetch(args.dest, count=args.count, start=args.start)
    print("models", result["models_info"])
    print("clips", len(result["clips"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
