import os, subprocess, glob, sys, time, json, shutil
def sh(c):
    print("+", c, flush=True); t=time.time(); r=subprocess.run(c, shell=True); print("rc", r.returncode, "sec %.0f"%(time.time()-t), flush=True); return r.returncode
sh("nvidia-smi; ls -R /kaggle/input | head -30")
inp = glob.glob("/kaggle/input/**/scripts.tgz", recursive=True)[0]; base = os.path.dirname(inp)
W="/kaggle/working"; os.chdir(W)
sh(f"tar xzf {inp} -C {W}")
src = glob.glob(base+"/**/meta/info.json", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(src)), W+"/ds")
sh("pip install -q 'lerobot[dataset]==0.6.1' av 2>&1 | tail -2; python -c 'import torch,lerobot;print(torch.__version__, torch.cuda.is_available())'")
assert sh("python -c 'import av; from lerobot.datasets.lerobot_dataset import LeRobotDataset; from lerobot.policies.act.modeling_act import ACTPolicy'")==0
split=json.load(open("docs/egodex_act/split.json")); eps=",".join(map(str,split["train"]))
STEPS=int(os.environ.get("STEPS","4000"))
def train(name, seed, nomask):
    env = "EGO_ACT_NO_MASK=1 " if nomask else ""
    rc = sh(f"{env}python scripts/ego_act_train.py --dataset.repo_id=local/egodex --dataset.root={W}/ds '--dataset.episodes=[{eps}]' "
       f"--policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.chunk_size=16 --policy.n_action_steps=16 "
       f"--batch_size=32 --steps={STEPS} --eval_steps=0 --save_freq={STEPS} --log_freq=200 --num_workers=4 --seed={seed} "
       f"--wandb.enable=false --output_dir={W}/runs/{name} 2>&1 | grep -E 'step:|屏蔽|Error|Traceback|rror' | tail -40")
    sh(f"python scripts/egodex_act_eval.py act --dataset {W}/ds --split docs/egodex_act/split.json --policy {W}/runs/{name} --out {W}/preds/{name}.npz --batch 64")
    sh(f"rm -rf {W}/runs/{name}/checkpoints/*/training_state")
os.makedirs(W+"/preds", exist_ok=True)
sh(f"python scripts/egodex_act_eval.py baseline --dataset {W}/ds --split docs/egodex_act/split.json --out {W}/preds/zero.npz")
sh(f"python scripts/egodex_act_eval.py linear --dataset {W}/ds --split docs/egodex_act/split.json --out {W}/preds/lin_mask.npz")
for name, seed, nm in [("act_mask_s0",0,False),("act_nomask_s0",0,True),("act_mask_s1",1,False),("act_nomask_s1",1,True),("act_mask_s2",2,False)]:
    train(name, seed, nm)
    preds=" ".join(f"{os.path.basename(p)[:-4]}={p}" for p in sorted(glob.glob(W+"/preds/*.npz")))
    sh(f"python scripts/egodex_act_eval.py metrics --dataset {W}/ds --pred {preds} --out {W}/metrics.json > /dev/null")
sh(f"cat {W}/metrics.json")
