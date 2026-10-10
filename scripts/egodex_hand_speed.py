# 快速运动评测辅助脚本（见 docs/hot3d_stereo/fast_motion.md）。在工作目录下运行，gt/ 为下载缓存。
import h5py,glob,numpy as np,json,random
fs=sorted(glob.glob("/workspace/egodex_data/**/*.hdf5",recursive=True));random.seed(0);fs=random.sample(fs,min(1500,len(fs)))
vw,vc,acc,ep_p99=[],[],[],[]
for fn in fs:
    try: f=h5py.File(fn)
    except Exception: continue
    C=f["transforms/camera"][:]; Ci=np.linalg.inv(C)
    for h in ("leftHand","rightHand"):
        T=f["transforms/"+h][:]; p=T[:,:3,3]
        pc=np.einsum("nij,nj->ni",Ci[:,:3,:3],p)+Ci[:,:3,3]
        d=np.linalg.norm(np.diff(p,axis=0),axis=1)*30; dc=np.linalg.norm(np.diff(pc,axis=0),axis=1)*30
        ok=d<5; vw+=list(d[ok]); vc+=list(dc[dc<5])
        v=np.diff(p,axis=0)*30; a=np.linalg.norm(np.diff(v,axis=0),axis=1)*30; acc+=list(a[a<200])
        if len(d): ep_p99.append(np.percentile(d,99))
q=lambda x:{k:round(float(np.percentile(x,v)),3) for k,v in (("median",50),("p90",90),("p99",99),("max",100))}
out={"files":len(fs),"fps":30,"wrist_speed_world_mps":q(vw),"wrist_speed_cam_mps":q(vc),"accel_mps2":q(acc),
 "frac_world_gt_1mps":float(np.mean(np.array(vw)>1)),"frac_world_gt_0_5":float(np.mean(np.array(vw)>.5)),"frac_cam_gt_1":float(np.mean(np.array(vc)>1))}
print(json.dumps(out,indent=1));json.dump(out,open("egodex_speed.json","w"),indent=1)
