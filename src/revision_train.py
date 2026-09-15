"""Five V11 branches re-trained with protected outer test partitions."""

from pathlib import Path
import contextlib, gc, json, math, os, random, shutil, time
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from revision_core import *

DEFAULT_CONFIG = {
    "seed": 42,
    "fusion_seeds": [42, 43, 44],
    "epochs_text": 10,
    "epochs_audio": 10,
    "epochs_visual": 8,
    "patience": 3,
    "batch_text": 4,
    "batch_visual": 1,
    "grad_accum_text": 6,
    "grad_accum_visual": 8,
    "max_len_text": 256,
    "max_len_audio": 192,
    "lr_body": 8e-6,
    "lr_head": 5e-4,
    "lr_audio": 8e-6,
    "lr_visual": 5e-5,
    "frames": 4,
    "image_size": 380,
    "amp_text": True,
    "swa": True,
    "swa_start_epoch": 7,
    "swa_lr": 5e-6,
    "augmentation_probability": 0.25,
    "bootstrap": 2000,
    "text_models": [
        "indobenchmark/indobert-base-p1",
        "indolem/indobertweet-base-uncased",
        "google-bert/bert-base-multilingual-cased",
    ],
    "gradient_checkpointing": True,
    "workers": 0,
    "require_gpu": True,
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def augment(text, prob=0.25):
    if random.random() > prob:
        return text
    words = text.split()
    if len(words) < 4:
        return text
    op = random.choice(["delete", "swap", "repeat", "shuffle_local"])
    if op == "delete":
        for _ in range(random.randint(1, min(3, len(words) // 4))):
            if len(words) > 3:
                words.pop(random.randrange(len(words)))
    elif op == "swap":
        for _ in range(random.randint(1, min(2, len(words) // 3))):
            i = random.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
    elif op == "repeat":
        i = random.randrange(len(words))
        words.insert(i + 1, words[i])
    else:
        width = min(4, len(words))
        i = random.randrange(len(words) - width + 1)
        w = words[i : i + width]
        random.shuffle(w)
        words[i : i + width] = w
    return " ".join(words)


class TextData(Dataset):
    def __init__(self, df, ids, tokenizer, field, length, aug=False, prob=0.25):
        self.texts = df.iloc[ids][field].astype(str).tolist()
        self.y = df.iloc[ids].label.to_numpy()
        self.tok = tokenizer
        self.length = length
        self.aug = aug
        self.prob = prob

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        text = augment(self.texts[i], self.prob) if self.aug else self.texts[i]
        x = self.tok(
            text,
            truncation=True,
            max_length=self.length,
            padding="max_length",
            return_tensors="pt",
        )
        return {"ids": x["input_ids"][0], "mask": x["attention_mask"][0]}, torch.tensor(
            self.y[i], dtype=torch.float32
        )


class TextModel(nn.Module):
    def __init__(self, name, k=5, gradient_checkpointing=True, revision=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name, revision=revision)
        h = self.bert.config.hidden_size
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.norm = nn.LayerNorm(h)
        self.cls = nn.Linear(h, 1)
        self.drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(k)])
        self.feature_dim = h

    def forward(self, ids, mask):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        x = self.norm(
            (
                h[:, 0]
                + (h * mask.unsqueeze(-1)).sum(1)
                / mask.sum(1).clamp_min(1).unsqueeze(-1)
            )
            / 2
        )
        logits = torch.stack([self.cls(d(x)) for d in self.drops]).mean(0).squeeze(-1)
        return logits, x


class VisualModel(nn.Module):
    def __init__(self, nframes=4, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

        bb = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.feat = bb.features
        self.pool = bb.avgpool
        self.nframes = nframes
        self.feature_dim = bb.classifier[1].in_features * nframes
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, frames):
        b, n, c, h, w = frames.shape
        x = self.pool(self.feat(frames.reshape(b * n, c, h, w))).reshape(b, -1)
        return self.head(x).squeeze(-1), x


def decode_video(path, nframes=4, size=380):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if not cap.isOpened() or count <= 0:
            return None, 0, "unreadable_container"
        for index in np.linspace(0, count - 1, nframes, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (size, size)))
    finally:
        cap.release()
    decoded = len(frames)
    if not decoded:
        return None, 0, "zero_decoded_frames"
    while len(frames) < nframes:
        frames.append(frames[-1].copy())
    return (
        np.stack(frames[:nframes]),
        decoded,
        "decoded" if decoded == nframes else "decoded_padded_repeat_last",
    )


def media_preflight(df, video_dir, cache_dir, out, config):
    video_dir = Path(video_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out)
    if not video_dir.is_dir():
        raise FileNotFoundError(
            f"VIDEO_DIR tidak ada: {video_dir}. Arahkan ke folder v2/scraped_dataset/videos."
        )
    index = {}
    for p in video_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".mp4",
            ".mkv",
            ".webm",
            ".avi",
            ".mov",
            ".m4v",
        }:
            index.setdefault(p.stem, []).append(p)
    rows = []
    for i, row in df.iterrows():
        # EXACT original ID matching only: no guessing comment IDs into video filenames.
        candidates = index.get(str(row.video_id), [])
        entry = {
            "sample_id": row.sample_id,
            "nominal_audio": bool(row.has_audio),
            "nominal_video": bool(row.has_video),
            "audio_effective": bool(row.audio_effective),
            "file_exists": bool(candidates),
            "visual_effective": False,
            "decoded_frames": 0,
            "reason": "file_missing",
            "media_sha256": None,
        }
        if len(candidates) > 1:
            raise RuntimeError(
                f"Beberapa video untuk ID yang sama {row.video_id}; pilih file sumber secara eksplisit."
            )
        if candidates:
            p = candidates[0]
            mh = digest(p)
            entry["media_sha256"] = mh
            cache = cache_dir / (row.sample_id + ".npz")
            key = stable_hash(
                [mh, config["frames"], config["image_size"], "decode-v12"]
            )
            loaded = False
            if cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as z:
                        if str(z["key"].item()) == key:
                            frames = z["frames"]
                            n = int(z["decoded"])
                            status = str(z["status"].item())
                            loaded = True
                except (ValueError, OSError, KeyError):
                    pass
            if not loaded:
                frames, n, status = decode_video(
                    p, config["frames"], config["image_size"]
                )
                if frames is not None:
                    tmp = cache.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        np.savez_compressed(
                            f, frames=frames, decoded=n, status=status, key=key
                        )
                    os.replace(tmp, cache)
                elif cache.exists():
                    cache.unlink()
            entry.update(
                visual_effective=frames is not None, decoded_frames=n, reason=status
            )
        rows.append(entry)
        if (i + 1) % 1000 == 0:
            print(f"Media checked {i + 1}/{len(df)}", flush=True)
    masks = pd.DataFrame(rows)
    masks.to_csv(out / "modality_masks.csv", index=False)
    vm = masks.visual_effective.to_numpy()
    am = df.audio_effective.to_numpy()
    coverage = {
        "n": len(df),
        "transcript_real": int(am.sum()),
        "visual_real": int(vm.sum()),
        "both_real": int((am & vm).sum()),
        "neither_auxiliary_real": int((~(am | vm)).sum()),
        "video_files_matched": int(masks.file_exists.sum()),
        "video_directory_files": sum(map(len, index.values())),
        "audio_status": "existing transcript CSV; ASR provenance/quality not independently verified",
        "visual_status": "actual decoding in this run, not copied from paper",
    }
    atomic_json(out / "coverage.json", coverage)
    if masks.file_exists.sum() == 0:
        raise RuntimeError(
            "Tidak satu pun nama file cocok dengan ID dataset. Perbaiki VIDEO_DIR sebelum training; tidak dilanjutkan sebagai visual 0.5 semua."
        )
    media_key = stable_hash(rows)
    lock = out / "media_signature.json"
    if lock.exists() and json.loads(lock.read_text())["signature"] != media_key:
        raise RuntimeError(
            "Media/hasil decoding berubah. Gunakan RUN_NAME baru; hasil tidak boleh bercampur."
        )
    atomic_json(lock, {"signature": media_key})
    return vm


class VisualData(Dataset):
    def __init__(self, df, ids, cache_dir, aug=False):
        from torchvision import transforms as T

        self.ids = df.iloc[ids].sample_id.tolist()
        self.y = df.iloc[ids].label.to_numpy()
        self.cache = Path(cache_dir)
        self.aug = aug
        self.tf = (
            T.Compose(
                [
                    T.ToPILImage(),
                    T.RandomResizedCrop(380, scale=(0.75, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.2, 0.2, 0.15),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if aug
            else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with np.load(self.cache / (self.ids[i] + ".npz"), allow_pickle=False) as z:
            frames = z["frames"]
        if self.aug:
            arr = torch.stack([self.tf(f) for f in frames])
        else:
            arr = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255
            arr = (
                arr - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return {"frames": arr}, torch.tensor(self.y[i], dtype=torch.float32)


def focal(logits, y, alpha):
    import torch.nn.functional as F

    logits = logits.float()
    y = y.float()
    smooth = y * 0.95 + 0.025
    bce = F.binary_cross_entropy_with_logits(logits, smooth, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    weight = alpha * y + (1 - alpha) * (1 - y)
    return (weight * (1 - pt).square() * bce).mean()


def rdrop(a, b):
    import torch.nn.functional as F

    p = torch.stack([1 - torch.sigmoid(a.float()), torch.sigmoid(a.float())], -1).clamp(
        1e-6, 1
    )
    q = torch.stack([1 - torch.sigmoid(b.float()), torch.sigmoid(b.float())], -1).clamp(
        1e-6, 1
    )
    return 0.25 * (
        F.kl_div(p.log(), q, reduction="batchmean")
        + F.kl_div(q.log(), p, reduction="batchmean")
    )


def inference(model, loader, device, amp=False):
    model.eval()
    probs = []
    features = []
    truth = []
    with torch.inference_mode():
        for inputs, y in loader:
            with torch.autocast("cuda", enabled=amp):
                logits, feat = model(**{k: v.to(device) for k, v in inputs.items()})
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            features.append(feat.float().cpu().numpy())
            truth.append(y.numpy())
    if not probs:
        return np.empty(0), np.empty((0, model.feature_dim)), np.empty(0)
    p = np.concatenate(probs)
    x = np.concatenate(features)
    y = np.concatenate(truth)
    check_prob(p, len(y))
    if not np.isfinite(x).all():
        raise FloatingPointError("Embedding nonfinite.")
    return p, x, y


def fit_network(model, train_ds, stop_ds, config, branch, folder, seed, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    visual = branch == "visual"
    audio = branch == "transcript"
    amp = bool(config["amp_text"] and not visual and device.type == "cuda")
    batch = config["batch_visual"] if visual else config["batch_text"]
    accum = config["grad_accum_visual"] if visual else config["grad_accum_text"]
    epochs = (
        config["epochs_visual"]
        if visual
        else config["epochs_audio"]
        if audio
        else config["epochs_text"]
    )
    # Filtered visual training: missing frames contribute neither labels nor BatchNorm updates.
    if len(train_ds) == 0:
        raise ValueError(
            "Tidak ada video nyata pada fit split; eksperimen visual tidak dapat diestimasi."
        )
    if len(np.unique(train_ds.y)) < 2:
        raise ValueError(
            "Fit split cabang hanya memiliki satu kelas; jangan melaporkan classifier ini sebagai eksperimen lengkap."
        )
    train = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
    )
    val = DataLoader(
        stop_ds, batch_size=batch, shuffle=False, num_workers=config["workers"]
    )
    model.to(device)
    if visual or audio:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_visual"] if visual else config["lr_audio"],
            weight_decay=0.01,
        )
    else:
        opt = torch.optim.AdamW(
            [
                {
                    "params": model.bert.parameters(),
                    "lr": config["lr_body"],
                    "weight_decay": 0.01,
                },
                {
                    "params": list(model.norm.parameters())
                    + list(model.cls.parameters()),
                    "lr": config["lr_head"],
                    "weight_decay": 0,
                },
            ]
        )
    steps = math.ceil(len(train) / accum) * epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    swa = None
    swa_sch = None
    if config["swa"] and not visual and not audio:
        from torch.optim.swa_utils import AveragedModel, SWALR

        swa = AveragedModel(model)
        swa_sch = SWALR(opt, swa_lr=config["swa_lr"], anneal_epochs=2)
    alpha = float(1 - np.mean(train_ds.y))
    start = 0
    best = -float("inf")
    pat = 0
    history = []
    best_state = None
    current = folder / "resume.pt"
    bestpath = folder / "best.pt"
    if current.exists():
        state = torch.load(current, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sch.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best = state["best"]
        pat = state["patience"]
        history = state["history"]
        if swa is not None:
            swa.load_state_dict(state["swa"])
            swa_sch.load_state_dict(state["swa_scheduler"])
    for ep in range(start, epochs):
        if pat >= config["patience"]:
            break
        seed_all(seed + ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        seen = 0
        start_time = time.time()
        _label = f"{folder.parent.name}/{branch} E{ep+1}"
        _iter = (
            _tqdm(enumerate(train), total=len(train), desc=_label,
                  unit="batch", leave=False,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")
            if _tqdm is not None
            else enumerate(train)
        )
        if _tqdm is not None:
            _iter.set_postfix(loss="?.????")
        for step, (inputs, y) in _iter:
            y = y.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            window = min(accum, len(train) - (step // accum) * accum)
            with torch.autocast("cuda", enabled=amp):
                l1, _ = model(**inputs)
                if visual:
                    loss = focal(l1, y, alpha)
                else:
                    l2, _ = model(**inputs)
                    loss = (focal(l1, y, alpha) + focal(l2, y, alpha)) / 2 + rdrop(
                        l1, l2
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss nonfinite {branch}, epoch {ep + 1}; checkpoint epoch sebelumnya aman."
                )
            scaler.scale(loss / window).backward()
            total += float(loss.detach()) * len(y)
            seen += len(y)
            if _tqdm is not None and seen > 0 and (step + 1) % max(1, len(train) // 20) == 0:
                _iter.set_postfix(loss=f"{total/seen:.4f}")
            if (step + 1) % accum == 0 or step + 1 == len(train):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=not amp
                )
                previous_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                optimizer_updated = not amp or scaler.get_scale() >= previous_scale
                if optimizer_updated and (
                    swa is None or ep + 1 < config["swa_start_epoch"]
                ):
                    sch.step()
        eval_model = model
        if swa is not None and ep + 1 >= config["swa_start_epoch"]:
            swa.update_parameters(model)
            swa_sch.step()
            eval_model = swa.module
        vp, _, vy = inference(eval_model, val, device, amp)
        vf = (
            f1_score(vy, vp >= 0.5, average="macro", labels=[0, 1], zero_division=0)
            if len(vy)
            else -total / seen
        )
        entry = {
            "epoch": ep + 1,
            "train_loss": total / seen,
            "early_stop_macro_f1": float(vf) if len(vy) else None,
            "selection_value": float(vf),
            "seconds": time.time() - start_time,
            "fit_n": len(train_ds),
            "stop_n": len(stop_ds),
            "alpha_fit_only": alpha,
        }
        history.append(entry)
        if vf > best:
            best = float(vf)
            pat = 0
            best_state = {
                "model": {
                    k: v.detach().cpu() for k, v in eval_model.state_dict().items()
                },
                "epoch": ep + 1,
                "selection_value": best,
                "branch": branch,
                "config": config,
            }
            atomic_torch(bestpath, best_state)
        else:
            pat += 1
        atomic_torch(
            current,
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sch.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "best": best,
                "patience": pat,
                "history": history,
                "swa": swa.state_dict() if swa is not None else None,
                "swa_scheduler": swa_sch.state_dict() if swa_sch else None,
            },
        )
        atomic_json(folder / "history.json", history)
        print(
            f"{folder.parent.name}/{branch} E{ep + 1}: loss={total / seen:.5f} inner_F1={vf:.5f}",
            flush=True,
        )
    if best_state is not None:
        state = best_state
    elif bestpath.exists():
        state = torch.load(bestpath, map_location="cpu", weights_only=False)
    else:
        state = torch.load(current, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    # Keep latest optimizer state until predictions have been committed successfully.
    return model


def run_branch(df, roles, branch, config, folder, cache_dir, visual_mask, seed, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "predictions.npz"
    role_names = ["meta_fit", "calibration", "test"]
    ids = np.concatenate([roles[k] for k in role_names])
    bounds = np.cumsum([0] + [len(roles[k]) for k in role_names])
    if output.exists():
        with np.load(output, allow_pickle=False) as z:
            if not np.array_equal(z["ids"], ids):
                raise ValueError("Cached prediction IDs tidak sesuai split.")
            p = z["prob"]
            x = z["features"]
            check_prob(p, len(ids))
            if x.shape[0] != len(ids) or not np.isfinite(x).all():
                raise ValueError("Embedding cache rusak/nonfinite.")
        return p, x, bounds
    seed_all(seed)
    visual = branch == "visual"
    audio = branch == "transcript"
    if visual:
        select = lambda r: [i for i in roles[r] if visual_mask[i]]
        train_ds = VisualData(df, select("fit"), cache_dir, True)
        stop_ds = VisualData(df, select("early_stop"), cache_dir)
        selected = ids[np.asarray(visual_mask)[ids]]
        pred_ds = VisualData(df, selected, cache_dir)
        model = VisualModel(config["frames"])
    else:
        j = 1 if audio else BRANCHES.index(branch)
        name = config["text_models"][j]
        revision = config.get("model_revisions", [None] * 3)[j]
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        tok.save_pretrained(folder / "tokenizer")
        field = "transcript_input" if audio else "text"
        length = config["max_len_audio"] if audio else config["max_len_text"]
        train_ds = TextData(
            df,
            roles["fit"],
            tok,
            field,
            length,
            True,
            config["augmentation_probability"],
        )
        stop_ds = TextData(df, roles["early_stop"], tok, field, length)
        pred_ds = TextData(df, ids, tok, field, length)
        model = TextModel(
            name, 3 if audio else 5, config["gradient_checkpointing"], revision
        )
        atomic_json(folder / "pretrained_config.json", model.bert.config.to_dict())
    model = fit_network(model, train_ds, stop_ds, config, branch, folder, seed, device)
    batch = config["batch_visual"] if visual else config["batch_text"]
    p, x, _ = inference(
        model,
        DataLoader(pred_ds, batch_size=batch, num_workers=0),
        device,
        bool(config["amp_text"] and not visual and device.type == "cuda"),
    )
    if visual:
        prob = np.full(len(ids), 0.5, dtype=np.float32)
        features = np.zeros((len(ids), model.feature_dim), dtype=np.float32)
        mask = np.asarray(visual_mask)[ids]
        prob[mask] = p
        features[mask] = x
        p, x = prob, features
    check_prob(p, len(ids))
    # Drive FUSE kadang gagal membuat folder saat mkdir() di atas,
    # pastikan parent directory benar-benar ada sebelum menulis.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        np.savez_compressed(
            f,
            ids=ids,
            prob=p.astype(np.float32),
            features=x.astype(np.float32),
            bounds=bounds,
        )
    if (folder / "resume.pt").exists():
        (folder / "resume.pt").unlink()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, x, bounds


def run_experiments(df, splits, out, cache_dir, visual_mask, config, only=None):
    out = Path(out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["require_gpu"] and device.type != "cuda":
        raise RuntimeError(
            "Aktifkan GPU di Runtime > Change runtime type sebelum training penuh."
        )
    all_results = []
    for exp, roles in splits.items():
        if only is not None and exp not in only:
            continue
        folder = out / exp
        folder.mkdir(parents=True, exist_ok=True)
        probabilities = []
        for j, branch in enumerate(BRANCHES):
            p, _, bounds = run_branch(
                df,
                roles,
                branch,
                config,
                folder / branch,
                cache_dir,
                visual_mask,
                config["seed"] + 100 * j + list(splits).index(exp) * 1000,
                device,
            )
            probabilities.append(p)
        X = np.stack(probabilities, 1)
        a, b, c, d = bounds
        outputs = fusion_outputs(
            df.iloc[roles["meta_fit"]].label.to_numpy(),
            X[a:b],
            df.iloc[roles["calibration"]].label.to_numpy(),
            X[b:c],
            X[c:d],
            folder / "fusion",
            config["fusion_seeds"],
        )
        for role, lo, hi in zip(
            ["meta_fit", "calibration", "test"], bounds[:-1], bounds[1:]
        ):
            t = pd.DataFrame(X[lo:hi], columns=BRANCHES)
            t.insert(0, "sample_id", df.iloc[roles[role]].sample_id.to_numpy())
            t.to_csv(folder / ("base_probabilities_" + role + ".csv"), index=False)
        pred = prediction_frame(df, roles["test"], outputs, exp, visual_mask)
        pred.to_csv(folder / "outer_predictions.csv", index=False)
        all_results.append(pred)
    ready = [out / exp / "outer_predictions.csv" for exp in splits]
    if all(p.exists() for p in ready):
        total = pd.concat([pd.read_csv(p) for p in ready], ignore_index=True)
        total.to_csv(out / "outer_predictions.csv", index=False)
        regenerate(total, out / "reports", config["bootstrap"])
        atomic_json(
            out / "TRAINING_COMPLETE.json",
            {
                "status": "all partitions finished",
                "experiments": list(splits),
                "n": len(total),
            },
        )
        return total
    print(
        "Run parsial tersimpan. Lanjutkan fold yang belum selesai; laporan akhir belum dibuat.",
        flush=True,
    )
    return pd.concat(all_results, ignore_index=True) if all_results else None


def prepare_model_locks(config, out):
    from huggingface_hub import model_info

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "model_locks.json"
    if lock.exists():
        data = json.loads(lock.read_text())
        if data["models"] != config["text_models"]:
            raise ValueError("Model IDs changed; choose a new run.")
    else:
        data = {
            "models": config["text_models"],
            "revisions": [
                model_info(m).sha if not Path(m).is_dir() else None
                for m in config["text_models"]
            ],
        }
        atomic_json(lock, data)
    config = dict(config)
    config["model_revisions"] = data["revisions"]
    return config


def environment_report(out):
    import importlib.metadata as md
    import cv2, torchvision

    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "opencv": cv2.__version__,
        "packages": {
            p: md.version(p)
            for p in [
                "transformers",
                "scikit-learn",
                "numpy",
                "pandas",
                "scipy",
                "matplotlib",
                "torchvision",
                "einops",
            ]
        },
    }
    data["packages"]["torchvision"] = torchvision.__version__
    atomic_json(Path(out) / "environment.json", data)
    return data
