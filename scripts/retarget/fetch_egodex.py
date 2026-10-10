# -*- coding: utf-8 -*-
"""按 HTTP Range 只取出 EgoDex test.zip 里的 basic_pick_place HDF5。

不下载整个约 17GB 的压缩包，也不取 mp4。已存在的文件会跳过。
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_hand_refine import EGODEX_ZIP_URL, HttpRangeSource, extract_member, read_zip_index  # noqa: E402

TASK_PREFIX = "test/basic_pick_place/"


def list_basic_pick_place(index):
    names = []
    for name in index:
        if not name.startswith(TASK_PREFIX) or not name.endswith(".hdf5"):
            continue
        stem = Path(name).stem
        if not stem.isdigit():
            continue
        names.append(name)
    return sorted(names, key=lambda item: int(Path(item).stem))


def _write_one(source, index, name, dest):
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return "skip"
    payload = extract_member(source, index[name])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".hdf5.part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return "wrote"


def download_basic_pick_place(dest, url=EGODEX_ZIP_URL, workers=8, retries=3):
    """把 ``test/basic_pick_place/*.hdf5`` 写到 ``dest/<序号>.hdf5``。返回 (写出, 跳过, 失败)。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    source = HttpRangeSource(url)
    index = read_zip_index(source)
    names = list_basic_pick_place(index)
    if not names:
        raise FileNotFoundError("test.zip 的目录里没有 test/basic_pick_place/*.hdf5")

    def attempt(name):
        target = dest / ("%s.hdf5" % Path(name).stem)
        last = None
        for _ in range(int(retries)):
            try:
                return name, _write_one(source, index, name, target), None
            except Exception as exc:  # noqa: BLE001 — 网络失败要重试，最后汇总
                last = "%s: %s" % (type(exc).__name__, exc)
        return name, "fail", last

    wrote = 0
    skipped = 0
    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(attempt, name) for name in names]
        for future in as_completed(futures):
            name, status, error = future.result()
            done += 1
            if status == "wrote":
                wrote += 1
            elif status == "skip":
                skipped += 1
            else:
                failed.append((name, error))
            if done % 20 == 0 or done == len(names):
                print("egodex %d/%d wrote=%d skip=%d fail=%d" % (done, len(names), wrote, skipped, len(failed)), flush=True)
    return {"n_members": len(names), "wrote": wrote, "skipped": skipped, "failed": failed}


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载 EgoDex basic_pick_place 的 HDF5（不含 mp4）")
    parser.add_argument("--dest", default="data/egodex_basic_pick_place")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = download_basic_pick_place(args.dest, workers=args.workers)
    print(result["n_members"], "members", "wrote", result["wrote"], "skip", result["skipped"], "fail", len(result["failed"]))
    for name, error in result["failed"]:
        print("FAIL", name, error)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
