# 快速运动评测辅助脚本（见 docs/hot3d_stereo/fast_motion.md）。在工作目录下运行，gt/ 为下载缓存。
import json,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fs=[f for f in fm.findSystemFonts() if any(k in f for k in ("NotoSansCJK","wqy","SourceHan","Noto Sans CJK"))]
if fs: fm.fontManager.addfont(fs[0]); plt.rcParams["font.family"]=fm.FontProperties(fname=fs[0]).get_name()
d=json.load(open("docs/hot3d_stereo/fast_motion_data/sim_fast.json"));R={r["label"]:r for r in d["runs"]}
X=[0.15,0.5,1,2];p90=lambda r:[b.get("final_p90_cm") for b in r["bins"]]
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
for s in (0.1,1,5,10,30): ax[0].plot(X,p90(R["同步偏差 %g ms"%s]),"o-",label="同步偏差 %g ms"%s)
ax[0].plot(X,p90(R["卷帘快门 读出 30 ms"]),"k--",label="卷帘 读出30ms")
for e in (2,10): ax[0].plot(X,p90(R["曝光 %g ms"%e]),":",label="曝光 %g ms"%e)
ax[0].set_title("【仿真】只三角化：硬件因素");
for fps in (30,60,120):
    ax[1].plot(X,p90(R["%d fps 现管线（门限2cm/帧 + q=0.3）"%fps]),"--",label="%dfps 现管线"%fps)
    ax[1].plot(X,p90(R["%d fps 新管线（匀加速预测门限2cm + RTS q=300）"%fps]),"o-",label="%dfps 新管线"%fps)
ax[1].set_title("【仿真】管线 p90 误差")
for fps in (30,60,120):
    k=lambda r:[100*b.get("kept_frac",0) for b in r["bins"]]
    ax[2].plot(X,k(R["%d fps 现管线（门限2cm/帧 + q=0.3）"%fps]),"--",label="%dfps 现管线"%fps)
    ax[2].plot(X,k(R["%d fps 新管线（匀加速预测门限2cm + RTS q=300）"%fps]),"o-",label="%dfps 新管线"%fps)
ax[2].set_title("【仿真】保留下来的帧 %");ax[2].set_ylabel("%")
for a in ax[:2]: a.axhline(2.5,color="r",lw=.8);a.set_ylabel("手腕误差 p90 (cm)")
for a in ax: a.set_xscale("log");a.minorticks_off();a.set_xticks(X);a.set_xticklabels(["0.15","0.5","1","2"]);a.set_xlabel("手腕速度 m/s");a.legend(fontsize=7);a.grid(alpha=.3)
plt.tight_layout();plt.savefig("docs/hot3d_stereo/fast_motion.png",dpi=110)
