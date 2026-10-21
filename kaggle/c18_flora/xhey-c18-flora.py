# C18: NVIDIA Flora-TempSeg (9B LoRA on Qwen3.5-9B, NVIDIA OneWay Noncommercial license -> research eval only)
# segments Open-AoE nano + xPhi-Ego-100 videos into timestamped action intervals. Segmentation only (no 27B captioner).
import subprocess, os, glob, json, time, sys, shutil
T0 = time.time(); W = '/kaggle/working/flora'; os.makedirs(W, exist_ok=True); TMP = '/kaggle/tmp'; os.makedirs(TMP, exist_ok=True)
BUDGET_S = float(os.environ.get('BUDGET_S', 10.8 * 3600))
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
def log(**k):
    k['t'] = round(time.time() - T0, 1); print('C18LOG ' + json.dumps(k), flush=True)
    open(W + '/progress.jsonl', 'a').write(json.dumps(k) + '\n')
sh('nvidia-smi; df -h /kaggle/tmp /kaggle/working | cat')
sh('pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2')
sh('pip install -q transformers==5.8.1 peft==0.18.1 accelerate==1.12.0 huggingface_hub==1.33.0 safetensors==0.7.0 av==16.0.1 2>&1 | tail -2')
rc_fla = sh('pip install -q flash-linear-attention==0.4.1 2>&1 | tail -2 || pip install -q flash-linear-attention 2>&1 | tail -2')  # v5: v4 prefill of ONE 16 s window took 651 s with the torch GDN fallback
try:
    import fla; rc_fla = f'fla {getattr(fla, "__version__", "?")} rc={rc_fla}'
except Exception as e: rc_fla = f'fla import failed: {e!r}'[:300]
log(step='install', fla_rc=rc_fla)
os.environ['HF_HOME'] = TMP + '/hf'; os.environ['FLORA_MODEL_CACHE'] = TMP + '/flora_models'
from huggingface_hub import snapshot_download
REPO = snapshot_download('nvidia/Flora-TempSeg', revision='78a0e4ac75d33509f077fdf985ebb51153920e7c', local_dir=TMP + '/flora_repo',
                         ignore_patterns=['*.safetensors', '*.pdf'])
log(step='repo', files=sum(len(f) for _, _, f in os.walk(REPO)), has_bu=os.path.isdir(REPO + '/batch_utils'))
if not os.path.isdir(REPO + '/batch_utils'): raise SystemExit('NO batch_utils in Flora-TempSeg snapshot')
sys.path.insert(0, REPO)
import torch
log(step='torch', v=torch.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    mem_gb=[round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(torch.cuda.device_count())],
    bf16=torch.cuda.is_bf16_supported())
from batch_utils import pipeline as P
from batch_utils.media import probe as _probe
from pathlib import Path
def probe(p): return _probe(Path(p))  # v4 fix: v3 passed str -> AttributeError 'str' has no 'open'
from batch_utils.models import Session

# ---- Kaggle has no Ampere+/80GB GPU for this account: NvidiaL4X1 request fell back to 2x T4 (15 GB each).
# Same loader logic as vendor/tempseg_fp32_loader.load (BF16 base, FP32 LoRA + FP32 GDN A_log/dt_bias, bf16 autocast),
# but the 9B base is split across both GPUs with accelerate device_map instead of model.to('cuda:0'). Vendor files untouched.
def multi_gpu_load(spec, save):
    import types
    from collections import Counter
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
    from safetensors.torch import load_file
    from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
    from production_gdn_fix import _keep_native_gdn_fp32
    from verify_gdn import verify
    import tempseg_fp32_loader as L
    _keep_native_gdn_fp32(Qwen3_5PreTrainedModel)
    n = torch.cuda.device_count()
    mem = {i: f'{int(torch.cuda.get_device_properties(i).total_memory / 2**30) - (4 if i == 0 else 2)}GiB' for i in range(n)}
    base, info = AutoModelForImageTextToText.from_pretrained(spec['base'], local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
                                                             output_loading_info=True, device_map='auto' if n > 1 else 'cuda:0', max_memory=mem if n > 1 else None)
    path = os.path.join(spec['export'], 'adapter_model.safetensors'); assert L.digest(path) == spec['adapter_sha256']
    source = load_file(path, device='cpu'); assert len(source) == 496
    model = get_peft_model(base, PeftConfig.from_pretrained(spec['export'], local_files_only=True), adapter_name='candidate', autocast_adapter_dtype=True)
    li = set_peft_model_state_dict(model, source, adapter_name='candidate'); assert not li.unexpected_keys, li
    model.eval()
    gdn = verify(model, spec['base'], peft_prefix=True); save('GDN_VALUE_VERIFICATION.json', gdn)
    save('ADAPTER_VALUE_VERIFICATION.json', dict(tensors=496, all_saved_values_exact_before_and_after_placement=True, holders_fp32_before_assignment=True))
    og = model.generate
    def generate_bf16(self, *a, **k):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return og(*a, **k)
    model.generate = types.MethodType(generate_bf16, model)
    log(step='multi_gpu_load', device_map=dict(Counter(str(v) for v in getattr(base, 'hf_device_map', {}).values())),
        dtypes=dict(Counter(str(p.dtype) for p in model.parameters())), alloc=[round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(n)])
    return model
from batch_utils.models import Session  # noqa (bootstrap puts vendor on sys.path)
import tempseg_fp32_loader
if not any(torch.cuda.get_device_properties(i).total_memory > 70 * 2**30 for i in range(torch.cuda.device_count())):
    tempseg_fp32_loader.load = multi_gpu_load
_orig_load = Session.load; _cache = {}
def _load(self, kind):  # load the 9B model ONCE, not per video
    if kind not in _cache:
        t = time.time(); _cache[kind] = _orig_load(self, kind); log(step='model_loaded', kind=kind, s=round(time.time() - t))
    return _cache[kind]
Session.load = _load; Session.close = lambda self: None
session = Session()
# ---- inputs
ORDER = [["xphi", "xphi_000159"], ["aoe", "poc_raw_video_20260131_193028"], ["xphi", "xphi_000160"], ["aoe", "poc_raw_video_20260131_193655_part002"], ["xphi", "xphi_000161"], ["aoe", "poc_raw_video_20260131_204056_part001"], ["xphi", "xphi_000162"], ["aoe", "poc_raw_video_20260131_204056_part003"], ["xphi", "xphi_000163"], ["aoe", "poc_raw_video_20260131_205349"], ["xphi", "xphi_000164"], ["aoe", "poc_raw_video_20260201_191442_part000"], ["xphi", "xphi_000165"], ["aoe", "poc_raw_video_20260201_191442_part001"], ["xphi", "xphi_000166"], ["aoe", "poc_raw_video_20260201_191651_part004"], ["xphi", "xphi_000167"], ["aoe", "poc_raw_video_20260201_192308_part000"], ["xphi", "xphi_000168"], ["aoe", "poc_raw_video_20260201_192308_part001"], ["xphi", "xphi_000169"], ["aoe", "poc_raw_video_20260201_192308_part002"], ["xphi", "xphi_000170"], ["aoe", "poc_raw_video_20260201_192308_part006"], ["xphi", "xphi_000171"], ["aoe", "poc_raw_video_20260201_192308_part007"], ["xphi", "xphi_000172"], ["aoe", "poc_raw_video_20260201_193100_part000"], ["xphi", "xphi_000173"], ["aoe", "poc_raw_video_20260201_193100_part001"], ["xphi", "xphi_000174"], ["aoe", "poc_raw_video_20260201_193100_part002"], ["xphi", "xphi_000175"], ["aoe", "poc_raw_video_20260201_193100_part003"], ["xphi", "xphi_000176"], ["aoe", "poc_raw_video_20260201_193100_part009"], ["xphi", "xphi_000177"], ["aoe", "poc_raw_video_20260201_193100_part010"], ["xphi", "xphi_000178"], ["aoe", "poc_raw_video_20260201_230005_part000"], ["xphi", "xphi_000179"], ["aoe", "poc_raw_video_20260201_230005_part002"], ["xphi", "xphi_000180"], ["aoe", "poc_raw_video_20260202_214320_part000"], ["xphi", "xphi_000181"], ["aoe", "poc_raw_video_20260202_214320_part001"], ["xphi", "xphi_000501"], ["aoe", "poc_raw_video_20260202_214320_part003"], ["xphi", "xphi_000502"], ["aoe", "poc_raw_video_20260202_214320_part006"], ["xphi", "xphi_000503"], ["aoe", "poc_raw_video_20260202_214320_part007"], ["xphi", "xphi_000504"], ["aoe", "poc_raw_video_20260202_214320_part008"], ["xphi", "xphi_000505"], ["aoe", "poc_raw_video_20260202_215703_part000"], ["xphi", "xphi_000506"], ["aoe", "poc_raw_video_20260202_220043_part001"], ["xphi", "xphi_000507"], ["aoe", "poc_raw_video_20260202_220043_part004"], ["xphi", "xphi_000508"], ["aoe", "poc_raw_video_20260207_202326_part000"], ["xphi", "xphi_000509"], ["aoe", "poc_raw_video_20260207_202326_part009"], ["xphi", "xphi_000510"], ["aoe", "poc_raw_video_20260207_202326_part010"], ["xphi", "xphi_000511"], ["aoe", "poc_raw_video_20260207_202326_part011"], ["xphi", "xphi_000512"], ["aoe", "poc_raw_video_20260207_203148"], ["xphi", "xphi_000513"], ["aoe", "poc_raw_video_20260207_203314"], ["xphi", "xphi_000514"], ["aoe", "poc_raw_video_20260207_203427_part000"], ["xphi", "xphi_000515"], ["aoe", "poc_raw_video_20260207_203427_part008"], ["xphi", "xphi_000516"], ["aoe", "poc_raw_video_20260207_204222"], ["xphi", "xphi_000517"], ["aoe", "poc_raw_video_20260214_232847_part000"], ["xphi", "xphi_000518"], ["aoe", "poc_raw_video_20260214_232847_part002"], ["xphi", "xphi_000519"], ["aoe", "poc_raw_video_20260214_233341_part000"], ["xphi", "xphi_000520"], ["aoe", "poc_raw_video_20260214_233341_part001"], ["xphi", "xphi_000521"], ["aoe", "poc_raw_video_20260214_233341_part002"], ["xphi", "xphi_000785"], ["aoe", "poc_raw_video_20260214_233559_part000"], ["xphi", "xphi_000786"], ["aoe", "poc_raw_video_20260214_233559_part001"], ["xphi", "xphi_000787"], ["aoe", "poc_raw_video_20260214_233830"], ["xphi", "xphi_000788"], ["aoe", "poc_raw_video_20260215_021222"], ["xphi", "xphi_000789"], ["aoe", "poc_raw_video_20260215_022506"], ["xphi", "xphi_000790"], ["aoe", "poc_raw_video_20260226_215034_part000"], ["xphi", "xphi_000791"], ["aoe", "poc_raw_video_20260226_215034_part001"], ["xphi", "xphi_000792"], ["aoe", "poc_raw_video_20260227_221514_part_5"], ["xphi", "xphi_000793"], ["aoe", "poc_raw_video_20260302_195810_eps2"], ["xphi", "xphi_000794"], ["aoe", "poc_raw_video_20260306_185932_part_5"], ["xphi", "xphi_000795"], ["aoe", "poc_raw_video_20260307_195104_part_5"], ["xphi", "xphi_000796"], ["aoe", "poc_raw_video_20260308_193952_part_10"], ["xphi", "xphi_000797"], ["aoe", "poc_raw_video_20260308_193952_part_9"], ["xphi", "xphi_000798"], ["aoe", "poc_raw_video_20260316_200454"], ["xphi", "xphi_000799"], ["aoe", "poc_raw_video_20260321_201633"], ["xphi", "xphi_000800"], ["aoe", "poc_raw_video_20260321_214344"], ["xphi", "xphi_000802"], ["aoe", "poc_raw_video_20260329_143722_part000"], ["xphi", "xphi_000803"], ["aoe", "poc_raw_video_20260329_143842_part000"], ["xphi", "xphi_000804"], ["aoe", "poc_raw_video_20260329_154934_part003"], ["xphi", "xphi_000805"], ["aoe", "poc_raw_video_20260329_154934_part005"], ["xphi", "xphi_000806"], ["aoe", "poc_raw_video_20260329_172344_part000"], ["xphi", "xphi_000947"], ["aoe", "poc_raw_video_20260329_185945_part001"], ["xphi", "xphi_000948"], ["aoe", "poc_raw_video_20260329_191023_part002"], ["xphi", "xphi_000949"], ["aoe", "poc_raw_video_20260329_213353_part003"], ["xphi", "xphi_000950"], ["aoe", "poc_raw_video_20260329_223617_part000"], ["xphi", "xphi_000951"], ["aoe", "poc_raw_video_20260429_204421_part000"], ["xphi", "xphi_000952"], ["aoe", "poc_raw_video_20260429_215845_part000"], ["xphi", "xphi_000953"], ["aoe", "poc_raw_video_20260430_211816_part000"], ["xphi", "xphi_000954"], ["aoe", "poc_raw_video_20260501_185231_part000"], ["xphi", "xphi_000955"], ["aoe", "poc_raw_video_20260501_185305_part001"], ["xphi", "xphi_000957"], ["aoe", "poc_raw_video_20260501_185335_part001"], ["xphi", "xphi_000960"], ["aoe", "poc_raw_video_20260501_185435_part000"], ["xphi", "xphi_000961"], ["aoe", "poc_raw_video_20260501_185435_part001"], ["xphi", "xphi_000963"], ["aoe", "poc_raw_video_20260501_191214_part001"], ["xphi", "xphi_000964"], ["aoe", "poc_raw_video_20260502_230029_part000"], ["xphi", "xphi_000965"], ["aoe", "poc_raw_video_20260502_230045_part000"], ["xphi", "xphi_000966"], ["aoe", "poc_raw_video_20260503_220944_part000"], ["xphi", "xphi_000967"], ["aoe", "poc_raw_video_20260503_221012_part000"], ["xphi", "xphi_000968"], ["aoe", "poc_raw_video_20260503_221020_part000"], ["xphi", "xphi_000969"], ["aoe", "poc_raw_video_20260505_101350_part000"], ["xphi", "xphi_001787"], ["aoe", "poc_raw_video_20260505_105634_part000"], ["xphi", "xphi_001789"], ["aoe", "poc_raw_video_20260505_105645_part000"], ["xphi", "xphi_001791"], ["aoe", "poc_raw_video_20260505_155559_part000"], ["xphi", "xphi_001792"], ["aoe", "poc_raw_video_20260505_210922_part000"], ["xphi", "xphi_001793"], ["aoe", "poc_raw_video_20260505_213228_part009"], ["xphi", "xphi_001794"], ["aoe", "poc_raw_video_20260506_210434_part000"], ["xphi", "xphi_001796"], ["aoe", "poc_raw_video_20260506_210508_part001"], ["xphi", "xphi_001797"], ["aoe", "poc_raw_video_20260506_220216_part002"], ["xphi", "xphi_001800"], ["aoe", "poc_raw_video_20260506_220253_part001"], ["xphi", "xphi_001802"], ["aoe", "poc_raw_video_20260506_222956_part000"], ["xphi", "xphi_001803"], ["aoe", "poc_raw_video_20260506_222956_part001"], ["xphi", "xphi_001804"], ["aoe", "poc_raw_video_20260507_211235_part000"], ["xphi", "xphi_001855"], ["aoe", "poc_raw_video_20260507_211256_part000"], ["xphi", "xphi_001856"], ["aoe", "poc_raw_video_20260507_211306_part001"], ["xphi", "xphi_001857"], ["aoe", "poc_raw_video_20260507_211306_part004"], ["xphi", "xphi_001860"], ["aoe", "poc_raw_video_20260507_212120_part000"], ["xphi", "xphi_001867"], ["aoe", "poc_raw_video_20260507_223812"], ["xphi", "xphi_001868"], ["aoe", "poc_raw_video_20260523_120216_part003"], ["aoe", "poc_raw_video_20260523_185806_part000"], ["aoe", "poc_raw_video_20260523_192000_part000"], ["aoe", "poc_raw_video_20260523_193213_part000"], ["aoe", "poc_raw_video_20260523_231754_part000"], ["aoe", "poc_raw_video_20260524_124807_part001"], ["aoe", "poc_raw_video_20260524_125158_part006"], ["aoe", "poc_raw_video_20260524_210145_part000"], ["aoe", "poc_raw_video_20260524_210328_part000"], ["aoe", "poc_raw_video_20260525_161715_part000"]]
AOE_PATHS = ["poc_raw_video_20260131_193028/raw_video.mp4", "poc_raw_video_20260131_193655_part002/raw_video.mp4", "poc_raw_video_20260131_204056_part001/raw_video.mp4", "poc_raw_video_20260131_204056_part003/raw_video.mp4", "poc_raw_video_20260131_205349/raw_video.mp4", "poc_raw_video_20260201_191442_part000/raw_video.mp4", "poc_raw_video_20260201_191442_part001/raw_video.mp4", "poc_raw_video_20260201_191651_part004/raw_video.mp4", "poc_raw_video_20260201_192308_part000/raw_video.mp4", "poc_raw_video_20260201_192308_part001/raw_video.mp4", "poc_raw_video_20260201_192308_part002/raw_video.mp4", "poc_raw_video_20260201_192308_part006/raw_video.mp4", "poc_raw_video_20260201_192308_part007/raw_video.mp4", "poc_raw_video_20260201_193100_part000/raw_video.mp4", "poc_raw_video_20260201_193100_part001/raw_video.mp4", "poc_raw_video_20260201_193100_part002/raw_video.mp4", "poc_raw_video_20260201_193100_part003/raw_video.mp4", "poc_raw_video_20260201_193100_part009/raw_video.mp4", "poc_raw_video_20260201_193100_part010/raw_video.mp4", "poc_raw_video_20260201_230005_part000/raw_video.mp4", "poc_raw_video_20260201_230005_part002/raw_video.mp4", "poc_raw_video_20260202_214320_part000/raw_video.mp4", "poc_raw_video_20260202_214320_part001/raw_video.mp4", "poc_raw_video_20260202_214320_part003/raw_video.mp4", "poc_raw_video_20260202_214320_part006/raw_video.mp4", "poc_raw_video_20260202_214320_part007/raw_video.mp4", "poc_raw_video_20260202_214320_part008/raw_video.mp4", "poc_raw_video_20260202_215703_part000/raw_video.mp4", "poc_raw_video_20260202_220043_part001/raw_video.mp4", "poc_raw_video_20260202_220043_part004/raw_video.mp4", "poc_raw_video_20260207_202326_part000/raw_video.mp4", "poc_raw_video_20260207_202326_part009/raw_video.mp4", "poc_raw_video_20260207_202326_part010/raw_video.mp4", "poc_raw_video_20260207_202326_part011/raw_video.mp4", "poc_raw_video_20260207_203148/raw_video.mp4", "poc_raw_video_20260207_203314/raw_video.mp4", "poc_raw_video_20260207_203427_part000/raw_video.mp4", "poc_raw_video_20260207_203427_part008/raw_video.mp4", "poc_raw_video_20260207_204222/raw_video.mp4", "poc_raw_video_20260214_232847_part000/raw_video.mp4", "poc_raw_video_20260214_232847_part002/raw_video.mp4", "poc_raw_video_20260214_233341_part000/raw_video.mp4", "poc_raw_video_20260214_233341_part001/raw_video.mp4", "poc_raw_video_20260214_233341_part002/raw_video.mp4", "poc_raw_video_20260214_233559_part000/raw_video.mp4", "poc_raw_video_20260214_233559_part001/raw_video.mp4", "poc_raw_video_20260214_233830/raw_video.mp4", "poc_raw_video_20260215_021222/raw_video.mp4", "poc_raw_video_20260215_022506/raw_video.mp4", "poc_raw_video_20260226_215034_part000/raw_video.mp4", "poc_raw_video_20260226_215034_part001/raw_video.mp4", "poc_raw_video_20260227_221514_part_5/raw_video.mp4", "poc_raw_video_20260302_195810_eps2/raw_video.mp4", "poc_raw_video_20260306_185932_part_5/raw_video.mp4", "poc_raw_video_20260307_195104_part_5/raw_video.mp4", "poc_raw_video_20260308_193952_part_10/raw_video.mp4", "poc_raw_video_20260308_193952_part_9/raw_video.mp4", "poc_raw_video_20260316_200454/raw_video.mp4", "poc_raw_video_20260321_201633/raw_video.mp4", "poc_raw_video_20260321_214344/raw_video.mp4", "poc_raw_video_20260329_143722_part000/raw_video.mp4", "poc_raw_video_20260329_143842_part000/raw_video.mp4", "poc_raw_video_20260329_154934_part003/raw_video.mp4", "poc_raw_video_20260329_154934_part005/raw_video.mp4", "poc_raw_video_20260329_172344_part000/raw_video.mp4", "poc_raw_video_20260329_185945_part001/raw_video.mp4", "poc_raw_video_20260329_191023_part002/raw_video.mp4", "poc_raw_video_20260329_213353_part003/raw_video.mp4", "poc_raw_video_20260329_223617_part000/raw_video.mp4", "poc_raw_video_20260429_204421_part000/raw_video.mp4", "poc_raw_video_20260429_215845_part000/raw_video.mp4", "poc_raw_video_20260430_211816_part000/raw_video.mp4", "poc_raw_video_20260501_185231_part000/raw_video.mp4", "poc_raw_video_20260501_185305_part001/raw_video.mp4", "poc_raw_video_20260501_185335_part001/raw_video.mp4", "poc_raw_video_20260501_185435_part000/raw_video.mp4", "poc_raw_video_20260501_185435_part001/raw_video.mp4", "poc_raw_video_20260501_191214_part001/raw_video.mp4", "poc_raw_video_20260502_230029_part000/raw_video.mp4", "poc_raw_video_20260502_230045_part000/raw_video.mp4", "poc_raw_video_20260503_220944_part000/raw_video.mp4", "poc_raw_video_20260503_221012_part000/raw_video.mp4", "poc_raw_video_20260503_221020_part000/raw_video.mp4", "poc_raw_video_20260505_101350_part000/raw_video.mp4", "poc_raw_video_20260505_105634_part000/raw_video.mp4", "poc_raw_video_20260505_105645_part000/raw_video.mp4", "poc_raw_video_20260505_155559_part000/raw_video.mp4", "poc_raw_video_20260505_210922_part000/raw_video.mp4", "poc_raw_video_20260505_213228_part009/raw_video.mp4", "poc_raw_video_20260506_210434_part000/raw_video.mp4", "poc_raw_video_20260506_210508_part001/raw_video.mp4", "poc_raw_video_20260506_220216_part002/raw_video.mp4", "poc_raw_video_20260506_220253_part001/raw_video.mp4", "poc_raw_video_20260506_222956_part000/raw_video.mp4", "poc_raw_video_20260506_222956_part001/raw_video.mp4", "poc_raw_video_20260507_211235_part000/raw_video.mp4", "poc_raw_video_20260507_211256_part000/raw_video.mp4", "poc_raw_video_20260507_211306_part001/raw_video.mp4", "poc_raw_video_20260507_211306_part004/raw_video.mp4", "poc_raw_video_20260507_212120_part000/raw_video.mp4", "poc_raw_video_20260507_223812/raw_video.mp4", "poc_raw_video_20260523_120216_part003/raw_video.mp4", "poc_raw_video_20260523_185806_part000/raw_video.mp4", "poc_raw_video_20260523_192000_part000/raw_video.mp4", "poc_raw_video_20260523_193213_part000/raw_video.mp4", "poc_raw_video_20260523_231754_part000/raw_video.mp4", "poc_raw_video_20260524_124807_part001/raw_video.mp4", "poc_raw_video_20260524_125158_part006/raw_video.mp4", "poc_raw_video_20260524_210145_part000/raw_video.mp4", "poc_raw_video_20260524_210328_part000/raw_video.mp4", "poc_raw_video_20260525_161715_part000/raw_video.mp4"]
def fetch_aoe():  # Open-AoE nano raw videos straight from Hugging Face (inclusionAI/OpenAoE-2000h, rev 'nano'; license: commercial ok w/ attribution)
    import urllib.request, concurrent.futures as cf
    os.makedirs(TMP + '/aoe_vid', exist_ok=True)
    def get(p):
        sid = p.split('/')[0]; out = f'{TMP}/aoe_vid/{sid}.mp4'; raw = out + '.raw.mp4'
        if os.path.exists(out): return
        for k in range(3):
            try:
                urllib.request.urlretrieve('https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/resolve/nano/' + p, raw); break
            except Exception as e: print('dl retry', sid, e, flush=True); time.sleep(5)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', raw, '-vf', 'fps=10,scale=640:-2', '-c:v', 'libx264', '-crf', '26', '-preset', 'veryfast', '-an', out])
        os.remove(raw)
    with cf.ThreadPoolExecutor(8) as ex: list(ex.map(get, AOE_PATHS))
import threading
t_dl = time.time(); th = threading.Thread(target=fetch_aoe, daemon=True); th.start()
# v4: smoke test (model load + one 30 s chunk) while Open-AoE downloads, so a broken run fails fast
_xp0 = sorted(glob.glob('/kaggle/input/**/xphi_*/rgb.mp4', recursive=True))[0]
sh(f'ffmpeg -v error -y -ss 0 -t 30 -i {_xp0} -vf fps=10,scale=320:-2 -c:v libx264 -preset veryfast -crf 23 -g 10 -an {TMP}/smoke.mp4')
try:
    _t = time.time(); _v = probe(f'{TMP}/smoke.mp4')
    _segs, _calls = P.segment({'main': _v}, 'human', session, '', lambda *x, **y: None)
    log(step='smoke_ok', clip=_xp0, windows=len(_calls), nseg=len(_segs), s=round(time.time() - _t, 1),
        segs=[(round(x['start_s'], 1), round(x['end_s'], 1)) for x in _segs], out0=(_calls[0]['output'][:300] if _calls else None),
        peak_gib=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])
except Exception as e:
    import traceback; traceback.print_exc(); log(step='smoke_fail', err=repr(e)[:600]); raise SystemExit('smoke failed')
if os.environ.get('SMOKE_ONLY', '1') == '1': log(step='smoke_only_exit'); os._exit(0)
th.join(); log(step='aoe_fetched', s=round(time.time() - t_dl))
CHUNK = 100.0
jobs = []
xp = sorted(glob.glob('/kaggle/input/**/xphi_*/rgb.mp4', recursive=True))
order = ORDER
aoe = sorted(glob.glob(TMP + '/aoe_vid/*.mp4'))
xpd = {p.split('/')[-2]: p for p in xp}; aod = {os.path.basename(p)[:-4]: p for p in aoe}
if order:
    seq = [(s, k) for s, k in order if (s == 'xphi' and k in xpd) or (s == 'aoe' and k in aod)]
else:
    seq = [x for pair in zip([('xphi', k) for k in xpd], [('aoe', k) for k in aod]) for x in pair]
log(step='inputs', xphi=len(xpd), aoe=len(aod), jobs=len(seq))
def dur(p):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]).decode().strip())
n_done = 0
for src, key in seq:
    out = f'{W}/{src}__{key}.json'
    if os.path.exists(out): continue
    if time.time() - T0 > BUDGET_S: log(step='budget_stop', done=n_done); break
    path = xpd[key] if src == 'xphi' else aod[key]
    D = dur(path); res = dict(source=src, clip=key, duration_s=D, chunk_s=CHUNK, chunks=[])
    t_clip = time.time()
    for ci, a in enumerate([i * CHUNK for i in range(int(D // CHUNK) + (1 if D % CHUNK > 1.0 else 0))]):
        cp = f'{TMP}/chunk.mp4'
        sh(f'ffmpeg -v error -y -ss {a} -t {CHUNK} -i {path} -vf fps=10,scale=320:-2 -c:v libx264 -preset veryfast -crf 23 -g 10 -an {cp}')
        try:
            v = probe(cp); t = time.time()
            segs, calls = P.segment({'main': v}, 'human', session, '', lambda *x, **y: None)
            res['chunks'].append(dict(offset_s=a, dur_s=v.duration_s, windows=len(calls), infer_s=round(time.time() - t, 1),
                                      segments=[dict(start_s=a + s['start_s'], end_s=a + s['end_s'],
                                                     prov=s['metadata'].get('segmentation_provenance', '')) for s in segs]))
        except Exception as e:  # noqa
            import traceback; traceback.print_exc()
            res['chunks'].append(dict(offset_s=a, error=repr(e)[:400]))
    res['wall_s'] = round(time.time() - t_clip, 1)
    json.dump(res, open(out, 'w')); n_done += 1
    if n_done >= 4 and not os.environ.get('NOABORT') and all(all('error' in c for c in json.load(open(f'{W}/{a}__{b}.json'))['chunks']) for a, b in seq[:4]):
        log(step='abort_all_errors'); break
    log(step='clip', src=src, clip=key, dur=round(D, 1), wall=res['wall_s'], nseg=sum(len(c.get('segments', [])) for c in res['chunks']),
        errs=sum('error' in c for c in res['chunks']), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
log(step='DONE', clips=n_done)
