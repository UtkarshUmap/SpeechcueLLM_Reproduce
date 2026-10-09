# ============================================================
# SpeechCueLLM - Phi-3 Mini FULL FT | 2x T4 | FSDP
# TEXT-ONLY (NO audio description) | MELD + IEMOCAP
# Fully separate repo / checkpoints / outputs / Kaggle handles
# ============================================================
%pip install -q "accelerate>=1.15.0,<2" "transformers>=4.56.1,<5" \
    "bitsandbytes>=0.50.2" "kagglehub>=1.0.0" "scikit-learn>=1.4" safetensors

import os, sys, subprocess
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_GPUS = 2
SCRIPT_PATH = Path("/kaggle/working/speechcuelm_phi3mini_nodesc_fsdp.py")

subprocess.run([sys.executable, "-c",
    "import torch;print('GPUs:',torch.cuda.device_count());"
    "[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"],
    check=True)

TRAINING_SCRIPT = r'''
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import functools, gc, json, math, random, re, shutil, subprocess, sys, time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict, StateDictOptions)

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from safetensors.torch import save_file
import bitsandbytes as bnb

# ------------------------------------------------------------
# CONFIG  (everything NODESC-specific is isolated here)
# ------------------------------------------------------------
SEED = 42
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASETS = tuple(os.environ.get("RUN_DATASETS", "meld,iemocap").split(","))

REPO = Path("/kaggle/working/SpeechCueLLM_nodesc")            # separate clone
OUTPUT_ROOT = Path("/kaggle/working/phi3_speechcuelm_nodesc_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = int(os.environ.get("MAX_LEN", "512"))
PROMPT_LIMIT = MAX_LENGTH - 16
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EPOCHS = 1
EVAL_BATCH = 8

LEARNING_RATES = {"meld": 2e-5, "iemocap": 2e-5}

CHECKPOINT_EVERY_UPDATES = 400
UPLOAD_CHECKPOINTS = True
UPLOAD_FINAL = True
DELETE_LOCAL_AFTER_UPLOAD = True     # frees disk once final model is on Kaggle
KAGGLE_USER = "utkarshumap"
# NEW handles -> your description-run datasets are never touched
CHECKPOINT_DATASET = f"{KAGGLE_USER}/speechcuelm-phi3mini-nodesc-fullft-checkpoints"
FINAL_HANDLE = KAGGLE_USER + "/speechcuelm-phi3mini-nodesc-{ds}-final"

STORE = Path("/kaggle/working/speechcuelm_nodesc_ckpt_store")
STORE.mkdir(parents=True, exist_ok=True)
DATA_INFO = STORE / "data_dirs.json"

# ------------------------------------------------------------
# DISTRIBUTED INIT
# ------------------------------------------------------------
assert torch.cuda.is_available()
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
assert WORLD_SIZE == 2, f"Expected 2 processes, got {WORLD_SIZE}"

torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

dist.init_process_group(
    backend="nccl", init_method="env://", rank=RANK, world_size=WORLD_SIZE,
    device_id=DEVICE, timeout=timedelta(hours=3))
IS_MAIN = RANK == 0

def barrier():
    dist.barrier(device_ids=[LOCAL_RANK])

def log(*a):
    if IS_MAIN:
        print(*a, flush=True)

random.seed(SEED + RANK)
torch.manual_seed(SEED + RANK)
torch.cuda.manual_seed_all(SEED + RANK)

log("=" * 70)
log("Phi-3 Mini FULL FT | TEXT-ONLY (no audio description) | FSDP FULL_SHARD |",
    "effective batch", BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM_STEPS,
    "| datasets:", DATASETS, "| max_len:", MAX_LENGTH)
log("=" * 70)

# ------------------------------------------------------------
# REPO + DATA PREP (audio_description = False)
# ------------------------------------------------------------
if not REPO.is_dir() and IS_MAIN:
    subprocess.run(["git", "clone", "https://github.com/zehuiwu/SpeechCueLLM.git",
                    str(REPO)], check=True)
barrier()
SCRIPT_DIR = REPO / "LLM_code"
assert SCRIPT_DIR.is_dir()

def find_data_dirs(name, since):
    base = REPO / "PROCESSED_DATASET" / name
    cands = []
    for t in base.rglob("train.json"):
        d = t.parent
        if all((d / f).is_file() for f in ("valid.json", "test.json")):
            cands.append((max((d / f).stat().st_mtime for f in
                              ("train.json", "valid.json", "test.json")), d))
    fresh = [c for c in cands if c[0] >= since]
    pool = fresh if fresh else cands
    assert pool, f"No processed data found under {base}"
    return max(pool, key=lambda x: x[0])[1]


def sanity_check_no_audio(d):
    text = (d / "train.json").read_text(encoding="utf-8").strip()
    try:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows] if "input" in rows else next(
                v for v in rows.values() if isinstance(v, list))
    except json.JSONDecodeError:                      # JSONL
        rows = [json.loads(l) for l in text.splitlines()[:300] if l.strip()]
    sample = rows[:300]
    hits = sum(any(k in str(r.get("input", "")).lower()
                   for k in ("pitch", "speaking rate", "volume"))
               for r in sample)
    print(f"[sanity] {d}: {hits}/{len(sample)} sample prompts mention acoustic terms", flush=True)
    if hits > 0.2 * len(sample):
        raise RuntimeError(f"Prompts in {d} still contain audio descriptions!")
    print("[sanity] sample input:\n", str(sample[0].get("input", ""))[:600], flush=True)


if IS_MAIN:
    info = {}
    for name in DATASETS:
        start = time.time() - 2
        subprocess.run([
            sys.executable, "data_process.py", "--dataset", name,
            "--historical_window", "12", "--audio_description", "False",
            "--audio_impression", "False", "--audio_only", "False",
            "--audio_context", "False", "--experiments_setting", "lora",
        ], cwd=SCRIPT_DIR, check=True)
        d = find_data_dirs(name, start)
        sanity_check_no_audio(d)
        info[name] = str(d)
    DATA_INFO.write_text(json.dumps(info, indent=2))
barrier()
DATA_DIRS = {k: Path(v) for k, v in json.loads(DATA_INFO.read_text()).items()}

def data_dir(name):
    return DATA_DIRS[name]

# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

def build_text(row):
    text = str(row.get("input", ""))
    ins = str(row.get("instruction", "") or "").strip()
    if ins and ins not in text:
        text = ins + "\n" + text
    return text

def get_target(row):
    return str(row.get("target", row.get("output", ""))).strip()

def encode_prompt(text):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(text)}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return [int(x) for x in ids[-PROMPT_LIMIT:]]   # keep the TAIL

def encode_example(row):
    prompt_ids = encode_prompt(build_text(row))
    ans = tokenizer.encode(" " + get_target(row), add_special_tokens=False)
    ans = [int(x) for x in ans[:12]] + [int(tokenizer.eos_token_id)]
    return {"input_ids": prompt_ids + ans,
            "labels": [-100] * len(prompt_ids) + ans}

def read_rows(filename):
    text = Path(filename).read_text(encoding="utf-8").strip()
    try:
        raw = json.loads(text)
        if isinstance(raw, dict):
            if "input" in raw:
                raw = [raw]
            else:
                raw = next(v for v in raw.values() if isinstance(v, list))
        rows = raw
    except json.JSONDecodeError:                     # JSONL
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [r for r in rows if get_target(r)]

class PromptDataset(Dataset):
    def __init__(self, filename):
        self.raw_rows = read_rows(filename)
        self.rows = [encode_example(r) for r in self.raw_rows]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

def collate_fn(rows):
    w = max(len(r["input_ids"]) for r in rows)
    ids, mask, labs = [], [], []
    for r in rows:
        p = w - len(r["input_ids"])
        ids.append(r["input_ids"] + [PAD_ID] * p)
        mask.append([1] * len(r["input_ids"]) + [0] * p)
        labs.append(r["labels"] + [-100] * p)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labs)}

# ------------------------------------------------------------
# PROGRESS (per dataset)
# ------------------------------------------------------------
PROGRESS_FILE = STORE / "progress.json"

def read_progress_file():
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        return {}

def get_progress(ds):
    return read_progress_file().get(ds)

def set_progress(ds, **kw):
    if not IS_MAIN:
        return
    data = read_progress_file()
    data[ds] = kw
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))

def drop_progress(ds):
    if not IS_MAIN:
        return
    data = read_progress_file()
    data.pop(ds, None)
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))

def ckpt_dir(ds):
    return STORE / f"{ds}_latest"

# ------------------------------------------------------------
# RESTORE FROM KAGGLE (only the NODESC checkpoint dataset)
# ------------------------------------------------------------
def restore_persistent():
    if PROGRESS_FILE.is_file() or not IS_MAIN:
        return
    try:
        import kagglehub
        remote = Path(kagglehub.dataset_download(CHECKPOINT_DATASET))
        if (remote / "progress.json").is_file():
            for item in remote.iterdir():
                if item.name == "data_dirs.json":     # paths are session-specific
                    continue
                if item.is_dir():
                    shutil.copytree(item, STORE / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, STORE / item.name)
            print("Persistent checkpoint restored:", sorted(p.name for p in STORE.iterdir()),
                  flush=True)
    except Exception as e:
        print("No persistent checkpoint:", repr(e)[:200], flush=True)

restore_persistent()
barrier()

def upload_checkpoint():
    if not (UPLOAD_CHECKPOINTS and IS_MAIN):
        return
    try:
        import kagglehub
        print("Uploading checkpoint to Kaggle...", flush=True)
        kagglehub.dataset_upload(
            CHECKPOINT_DATASET, str(STORE),
            version_notes="nodesc ckpt " + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Upload done.", flush=True)
    except Exception as e:
        print("Upload failed (local checkpoint is kept):", repr(e)[:300], flush=True)

def upload_final(ds, final_dir):
    if not (UPLOAD_FINAL and IS_MAIN):
        return False
    try:
        import kagglehub
        handle = FINAL_HANDLE.format(ds=ds)
        print("Uploading final model to", handle, flush=True)
        kagglehub.dataset_upload(
            handle, str(final_dir),
            version_notes="final nodesc " + ds + " " + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Final model upload done.", flush=True)
        return True
    except Exception as e:
        print("Final upload failed (local copy kept):", repr(e)[:300], flush=True)
        return False

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
def build_model():
    log("Loading Phi-3 Mini (FP16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, low_cpu_mem_usage=True, dtype=torch.float16,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Phi3DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.float16,
                                       reduce_dtype=torch.float16,
                                       buffer_dtype=torch.float16),
        device_id=DEVICE,
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True)
    barrier()
    log("FSDP ready. GPU mem (GB):", round(torch.cuda.memory_allocated() / 1e9, 2))
    return model

def build_scheduler(optimizer, total_updates):
    warm = max(10, int(total_updates * 0.03))
    def fn(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, 1.0 - (step - warm) / max(1, total_updates - warm))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)

# ------------------------------------------------------------
# CHECKPOINT (model weights only)
# ------------------------------------------------------------
SD_OPTS = StateDictOptions(full_state_dict=False)

def save_checkpoint(model, ds, epoch, next_batch, update_count):
    barrier()
    final = ckpt_dir(ds)
    if IS_MAIN:
        drop_progress(ds)
        shutil.rmtree(final, ignore_errors=True)
        final.mkdir(parents=True)
    barrier()
    dcp.save({"model": get_model_state_dict(model, options=SD_OPTS)},
             checkpoint_id=str(final))
    barrier()
    if IS_MAIN:
        set_progress(ds, epoch=epoch, next_batch=next_batch,
                     update_count=update_count, completed=False)
        print(f"\nCHECKPOINT SAVED | {ds} epoch={epoch} next_batch={next_batch} "
              f"update={update_count}", flush=True)
    barrier()
    upload_checkpoint()
    barrier()

def load_checkpoint(model, ds):
    state = {"model": get_model_state_dict(model, options=SD_OPTS)}
    dcp.load(state, checkpoint_id=str(ckpt_dir(ds)))
    set_model_state_dict(model, state["model"], options=SD_OPTS)
    barrier()

# ------------------------------------------------------------
# EVALUATION (all ranks participate)
# ------------------------------------------------------------
def evaluate(model, dataset, split_name):
    rows = dataset.raw_rows
    labels = sorted({get_target(r) for r in rows})
    my_idx = list(range(RANK, len(rows), WORLD_SIZE))
    prompts = {i: encode_prompt(build_text(rows[i])) for i in my_idx}
    my_idx.sort(key=lambda i: len(prompts[i]))

    model.eval()
    inner = model.module
    inner.config.use_cache = True
    local = []
    with torch.no_grad(), FSDP.summon_full_params(model, writeback=False, recurse=True):
        for s in range(0, len(my_idx), EVAL_BATCH):
            chunk = my_idx[s:s + EVAL_BATCH]
            w = max(len(prompts[i]) for i in chunk)
            ids = torch.tensor([[PAD_ID] * (w - len(prompts[i])) + prompts[i]
                                for i in chunk], device=DEVICE)
            mask = torch.tensor([[0] * (w - len(prompts[i])) + [1] * len(prompts[i])
                                 for i in chunk], device=DEVICE)
            out = inner.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=8,
                do_sample=False, pad_token_id=PAD_ID,
                eos_token_id=tokenizer.eos_token_id, use_cache=True)
            for j, i in enumerate(chunk):
                text = tokenizer.decode(out[j, w:], skip_special_tokens=True)
                words = re.sub(r"[^a-z]+", " ", text.lower()).split()
                pred, best = "unknown", 10 ** 9
                for lab in labels:
                    if lab.lower() in words and words.index(lab.lower()) < best:
                        pred, best = lab, words.index(lab.lower())
                local.append((i, pred))
            if IS_MAIN and (s // EVAL_BATCH) % 20 == 0:
                print(f"[{split_name}] {s}/{len(my_idx)}", flush=True)
    inner.config.use_cache = False

    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    metrics = None
    if IS_MAIN:
        pm = dict(x for part in gathered for x in part)
        refs = [get_target(rows[i]) for i in range(len(rows))]
        preds = [pm[i] for i in range(len(rows))]
        metrics = {
            "accuracy": accuracy_score(refs, preds),
            "macro_f1": f1_score(refs, preds, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(refs, preds, labels=labels, average="weighted", zero_division=0),
            "unknown_predictions": sum(p == "unknown" for p in preds),
        }
        print(split_name, "metrics:", json.dumps(metrics, indent=2), flush=True)
    barrier()
    return metrics

def save_final_model(model, ds):
    final_dir = OUTPUT_ROOT / ds / "final_model"
    if IS_MAIN:
        final_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    sd = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if IS_MAIN:
        print("Saving final model...", flush=True)
        save_file({k: v.contiguous() for k, v in sd.items()},
                  str(final_dir / "model.safetensors"), metadata={"format": "pt"})
        model.module.config.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print("Final model saved:", final_dir, flush=True)
    del sd
    gc.collect()
    barrier()
    return final_dir

# ------------------------------------------------------------
# TRAIN ONE DATASET
# ------------------------------------------------------------
def run_dataset(ds):
    log("\n" + "=" * 70 + f"\nDATASET: {ds.upper()} (NO DESCRIPTION)\n" + "=" * 70)

    prog = get_progress(ds)
    if prog and prog.get("completed"):
        log(ds, "already completed - skipping.")
        return {"dataset": ds, "skipped": True}

    d = data_dir(ds)
    train_ds = PromptDataset(d / "train.json")
    valid_ds = PromptDataset(d / "valid.json")
    test_ds = PromptDataset(d / "test.json")
    log("Train/Valid/Test:", len(train_ds), len(valid_ds), len(test_ds))
    log("Sample keys:", list(train_ds.raw_rows[0].keys()))
    log("Labels:", sorted({get_target(r) for r in train_ds.raw_rows}))

    sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK,
                                 shuffle=True, seed=SEED, drop_last=False)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = build_model()
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATES[ds],
                                    betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    total_updates = math.ceil(len(loader) / GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = build_scheduler(optimizer, total_updates)
    scaler = ShardedGradScaler(init_scale=2 ** 14)

    resume_epoch = resume_batch = update_count = 0
    if prog and ckpt_dir(ds).is_dir():
        load_checkpoint(model, ds)
        resume_epoch = int(prog["epoch"])
        resume_batch = int(prog["next_batch"])
        update_count = int(prog["update_count"])
        for _ in range(update_count):
            scheduler.step()
        log(f"RESUMED epoch={resume_epoch} batch={resume_batch} update={update_count}"
            " (optimizer moments restart fresh)")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    last_loss = float("nan")

    for epoch in range(resume_epoch, EPOCHS):
        sampler.set_epoch(epoch)
        log(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for bi, batch in enumerate(loader):
            if epoch == resume_epoch and bi < resume_batch:
                continue
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            last_loss = out.loss.item()
            scaler.scale(out.loss / GRAD_ACCUM_STEPS).backward()
            del out

            if (bi + 1) % GRAD_ACCUM_STEPS == 0 or bi == len(loader) - 1:
                scaler.unscale_(optimizer)
                model.clip_grad_norm_(1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1

                if IS_MAIN and update_count % 10 == 0:
                    print(f"epoch={epoch + 1} batch={bi + 1}/{len(loader)} "
                          f"update={update_count}/{total_updates} loss={last_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"scale={scaler.get_scale():.0f} "
                          f"elapsed={(time.time() - t0) / 60:.1f}m", flush=True)

                if update_count % CHECKPOINT_EVERY_UPDATES == 0:
                    save_checkpoint(model, ds, epoch, bi + 1, update_count)

        save_checkpoint(model, ds, epoch + 1, 0, update_count)

    log("\nTraining finished:", ds)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler, loader
    gc.collect()
    torch.cuda.empty_cache()

    valid_m = evaluate(model, valid_ds, "VALID")
    test_m = evaluate(model, test_ds, "TEST")

    barrier()
    if IS_MAIN:
        shutil.rmtree(ckpt_dir(ds), ignore_errors=True)   # free disk before final save
    barrier()
    final_dir = save_final_model(model, ds)
    uploaded = upload_final(ds, final_dir)
    barrier()

    result = None
    if IS_MAIN:
        result = {"dataset": ds, "setting": "no_audio_description",
                  "valid": valid_m, "test": test_m,
                  "training_updates": update_count,
                  "final_model": str(final_dir) if not (uploaded and DELETE_LOCAL_AFTER_UPLOAD)
                                 else FINAL_HANDLE.format(ds=ds) + " (Kaggle)"}
        sf = OUTPUT_ROOT / "summary.json"
        try:
            existing = json.loads(sf.read_text())
        except Exception:
            existing = []
        existing = [x for x in existing if x.get("dataset") != ds] + [result]
        sf.write_text(json.dumps(existing, indent=2))
        set_progress(ds, epoch=EPOCHS, next_batch=0,
                     update_count=update_count, completed=True)
        if uploaded and DELETE_LOCAL_AFTER_UPLOAD:
            shutil.rmtree(final_dir, ignore_errors=True)
            print("Local final model deleted (safe on Kaggle).", flush=True)
        print("\n" + "=" * 70 + f"\n{ds.upper()} COMPLETE\n" + "=" * 70)
        print(json.dumps(result, indent=2), flush=True)
    barrier()

    del model, train_ds, valid_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()
    barrier()
    return result

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
results = []
for name in DATASETS:
    r = run_dataset(name)
    if IS_MAIN:
        results.append(r)

log("\nALL DONE FOR:", DATASETS)
log(json.dumps(results, indent=2))
barrier()
dist.destroy_process_group()
'''

SCRIPT_PATH.write_text(TRAINING_SCRIPT, encoding="utf-8")
subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT_PATH)], check=True)
print(" Script written and syntax OK:", SCRIPT_PATH)

# ------------------------------------------------------------
# LAUNCH (one fresh process per dataset so GPU memory is clean)
# ------------------------------------------------------------
for ds, max_len in (("meld", "512"), ("iemocap", "384")):
    print(f"\n===== Launching {ds} NO-DESC (MAX_LEN={max_len}) =====", flush=True)
    env = dict(os.environ, RUN_DATASETS=ds, MAX_LEN=max_len)
    process = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--standalone", f"--nproc_per_node={NUM_GPUS}", str(SCRIPT_PATH)],
        cwd="/kaggle/working", env=env)
    if process.returncode != 0:
        raise RuntimeError(f"{ds} failed - read the first rank error above.")

print(" ALL FINISHED. Outputs: /kaggle/working/phi3_speechcuelm_nodesc_outputs")