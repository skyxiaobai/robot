# 快速运动评测辅助脚本（见 docs/hot3d_stereo/fast_motion.md）。在工作目录下运行，gt/ 为下载缓存。
import sys,os,json,tarfile,requests,concurrent.futures as cf
BASE="https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/"
def go(p):
    out="gt/"+p.replace("/","_").replace(".tar",".json")
    if os.path.exists(out): return p
    fr={}
    for _ in range(3):
        try:
            r=requests.get(BASE+p,stream=True,timeout=120)
            with tarfile.open(fileobj=r.raw,mode="r|") as t:
                for m in t:
                    if m.name.endswith((".hands.json",".info.json",".cameras.json")):
                        fid,k=m.name.split(".")[0],m.name.split(".")[-2]
                        fr.setdefault(fid,{})[k]=json.load(t.extractfile(m))
            break
        except Exception as e: fr={}; print("retry",p,e,flush=True)
    json.dump(fr,open(out,"w")); return p
ps=[l.strip() for l in open(sys.argv[1])]
with cf.ThreadPoolExecutor(8) as ex:
    for i,p in enumerate(ex.map(go,ps)): print(i,p,flush=True)
