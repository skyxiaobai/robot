# 快速运动评测辅助脚本（见 docs/hot3d_stereo/fast_motion.md）。在工作目录下运行，gt/ 为下载缓存。
import json,glob,numpy as np
from scipy.spatial.transform import Rotation as R
def Tcam(c):
    q=c["T_world_from_camera"]["quaternion_wxyz"];t=c["T_world_from_camera"]["translation_xyz"]
    T=np.eye(4);T[:3,:3]=R.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();T[:3,3]=t;return T
res={};clips=[]
for dev in ("quest3","aria"):
    V,VC,A,W=[],[],[],[]
    for fn in sorted(glob.glob("gt/train_%s_*.json"%dev)):
        d=json.load(open(fn));keys=sorted(d)
        if len(keys)<10: continue
        for side in ("left","right"):
            t,p,pc,ang=[],[],[],[]
            for k in keys:
                fr=d[k];h=fr.get("hands",{}).get(side)
                if not h or "info" not in fr: t.append(None);continue
                x=np.array(h["mano_pose"]["wrist_xform"][3:])
                cam=sorted(fr["cameras"])[0];T=np.linalg.inv(Tcam(fr["cameras"][cam]))
                xc=T[:3,:3]@x+T[:3,3]
                t.append(fr["info"]["image_timestamps_ns"][cam]*1e-9);p.append(x);pc.append(xc)
            ok=[i for i in range(len(t)) if t[i] is not None]
            sp=[];spc=[];om=[]
            for a,b in zip(ok,ok[1:]):
                if b!=a+1: continue
                ia,ib=ok.index(a),ok.index(b);dt=t[b]-t[a]
                if dt<=0: continue
                sp.append(np.linalg.norm(p[ib]-p[ia])/dt);spc.append(np.linalg.norm(pc[ib]-pc[ia])/dt)
                # 图像角速度：横向分量/深度
                u=lambda X:X[:2]/max(X[2],0.05);om.append(np.linalg.norm(u(pc[ib])-u(pc[ia]))/dt)
            sp=np.array(sp)
            if len(sp)<5: continue
            acc=np.abs(np.diff(sp))/ (1/30)
            V+=list(sp);VC+=list(spc);W+=list(om);A+=list(acc)
            clips.append({"dev":dev,"clip":fn,"side":side,"median":float(np.median(sp)),"p90":float(np.percentile(sp,90)),"max":float(sp.max()),"max_cam":float(np.max(spc)),"max_omega":float(np.max(om))})
    q=lambda x:{k:round(float(np.percentile(x,v)),3) for k,v in (("median",50),("p90",90),("p99",99),("max",100))}
    res[dev]={"n_clips":len(glob.glob("gt/train_%s_*.json"%dev)),"n_samples":len(V),"wrist_speed_world_mps":q(V),"wrist_speed_cam_mps":q(VC),
      "image_angular_speed_radps":q(W),"speed_change_mps2":q(A),"frac_cam_gt_0.5":float(np.mean(np.array(VC)>.5)),"frac_cam_gt_1":float(np.mean(np.array(VC)>1))}
clips.sort(key=lambda c:-c["max_cam"]);res["fastest_clips"]=clips[:15]
json.dump(res,open("hot3d_speed.json","w"),indent=1);print(json.dumps(res,indent=1)[:3500])
