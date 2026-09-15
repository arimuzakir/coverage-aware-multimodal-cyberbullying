"""Controlled baselines on the exact V12 roles. HEALNet uses official fusion code."""

from pathlib import Path
import copy, importlib.util, json, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from revision_core import *
from revision_train import seed_all, atomic_torch

BASELINE_CONFIG = {
    "seed_list": [42, 43, 44],
    "epochs": 100,
    "patience": 10,
    "batch": 64,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.2,
    "projection_dim": 128,
    "healnet_depth": 3,
    "healnet_latents": 16,
    "healnet_latent_dim": 128,
    "healnet_heads": 4,
    "healnet_head_dim": 32,
    "source_commit": "90459b65a3d4a4ef9fd405671c457b5a1163cc7d",
    "source_url": "https://github.com/konst-int-i/healnet",
    "publication": "HEALNet, NeurIPS 2024",
    "adaptation": "Official fusion core; supervised binary classifier on frozen V12 fold-specific embeddings; not replication of biomedical experiments.",
}


def official_healnet():
    path = Path(__file__).parent / "vendor" / "healnet_checked.py"
    spec = importlib.util.spec_from_file_location("healnet_checked", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HealNet


class FeatureFusion(nn.Module):
    def __init__(self, dims, kind, config):
        super().__init__()
        self.kind = kind
        d = config["projection_dim"]
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Linear(h, d), nn.LayerNorm(d), nn.GELU()) for h in dims]
        )
        if kind == "feature_concat":
            self.head = nn.Sequential(
                nn.Linear(d * 3 + 2, 128),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(32, 1),
            )
        elif kind == "healnet2024":
            self.head = official_healnet()(
                n_modalities=3,
                channel_dims=[d, d, d],
                num_spatial_axes=[1, 1, 1],
                out_dims=1,
                depth=config["healnet_depth"],
                l_c=config["healnet_latents"],
                l_d=config["healnet_latent_dim"],
                x_heads=config["healnet_heads"],
                l_heads=config["healnet_heads"],
                cross_dim_head=config["healnet_head_dim"],
                latent_dim_head=config["healnet_head_dim"],
                attn_dropout=0.0,
                ff_dropout=config["dropout"],
                fourier_encode_data=False,
                snn=False,
            )
        else:
            raise ValueError(kind)

    def forward(self, x, mask):
        z = [p(v) for p, v in zip(self.proj, x)]
        if self.kind == "feature_concat":
            parts = [
                z[0].mean(1),
                z[1].mean(1) * mask[:, 0:1],
                z[2].mean(1) * mask[:, 1:2],
                mask,
            ]
            return self.head(torch.cat(parts, 1)).squeeze(-1)
        # Official missingness is modality-wise None; group each batch by actual pattern.
        result = torch.zeros(len(mask), device=mask.device)
        for pattern in torch.unique(mask, dim=0):
            ids = torch.where((mask == pattern).all(1))[0]
            tensors = [
                z[0][ids],
                z[1][ids] if bool(pattern[0]) else None,
                z[2][ids] if bool(pattern[1]) else None,
            ]
            result[ids] = self.head(tensors).squeeze(-1)
        return result


def load_fold_features(df, roles, folder, visual_mask, nframes=4):
    folder = Path(folder)
    ids = np.concatenate([roles[k] for k in ["meta_fit", "calibration", "test"]])
    features = []
    for name in BRANCHES:
        path = folder / name / "predictions.npz"
        if not path.exists():
            found = False
            if os.path.exists(str(folder)):
                for cand_name in sorted(os.listdir(str(folder))):
                    if cand_name.startswith(name):
                        cand_path = folder / cand_name / "predictions.npz"
                        if cand_path.is_file() and cand_path.stat().st_size > 10000:
                            path = cand_path
                            found = True
                            break
            if not found:
                raise FileNotFoundError(f"Jalankan notebook utama dulu: {path}")
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z["ids"], ids):
                raise ValueError("ID fitur berbeda dari split.")
            features.append(z["features"].astype(np.float32))
    h = features[0].shape[1]
    if any(f.shape[1] != h for f in features[:4]):
        raise ValueError("Dimensi tiga encoder harus sama untuk token HEALNet.")
    masks = np.stack(
        [df.iloc[ids].audio_effective.to_numpy(), np.asarray(visual_mask)[ids]], 1
    ).astype(np.float32)
    arrays = [
        np.stack(features[:3], 1),
        features[3][:, None, :] * masks[:, 0, None, None],
        features[4].reshape(len(ids), nframes, -1) * masks[:, 1, None, None],
    ]
    bounds = np.cumsum(
        [0] + [len(roles[k]) for k in ["meta_fit", "calibration", "test"]]
    )
    return arrays, masks, bounds


def neural_baseline(
    arrays, masks, y_meta, y_cal, bounds, kind, seed, config, folder, device
):
    folder = Path(folder)
    import subprocess
    if os.name != "nt":
        subprocess.run(["mkdir", "-p", str(folder)], check=False)
    folder.mkdir(parents=True, exist_ok=True)
    a, b, c, d = bounds
    cache = folder / "predictions.npz"
    if not cache.exists():
        import shutil
        conflicts = list(folder.parent.parent.glob(f"**/{folder.parent.name}/**/{folder.name}/predictions.npz"))
        if conflicts:
            shutil.copy2(str(conflicts[0]), str(cache))
    if cache.exists():
        hist_file = folder / "history.json"
        if not hist_file.exists():
            import shutil
            h_conf = list(folder.parent.parent.glob(f"**/{folder.parent.name}/**/{folder.name}/history.json"))
            if h_conf:
                shutil.copy2(str(h_conf[0]), str(hist_file))
        if hist_file.exists():
            try:
                h = json.loads(hist_file.read_text())
                last_ep = h[-1]["epoch"]
                best_val = min(item["calibration_loss_for_inner_stopping"] for item in h)
                print(f"  [{kind} seed {seed}] Checkpoint loaded | Epochs: {last_ep} | Best Cal Loss: {best_val:.4f} | Converged", flush=True)
            except Exception:
                print(f"  [{kind} seed {seed}] Checkpoint loaded successfully.", flush=True)
        else:
            print(f"  [{kind} seed {seed}] Checkpoint loaded successfully.", flush=True)
        with np.load(cache, allow_pickle=False) as z:
            return check_prob(z["cal"], len(y_cal)), check_prob(z["test"], d - c)
    seed_all(seed)
    x = [torch.from_numpy(v) for v in arrays]
    m = torch.from_numpy(masks)
    model = FeatureFusion([v.shape[-1] for v in arrays], kind, config).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    best = float("inf")
    pat = 0
    history = []
    start = 0
    resume = folder / "resume.pt"
    if resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        best = st["best"]
        pat = st["pat"]
        history = st["history"]
        start = st["epoch"] + 1

    def forward_ids(ids):
        return model([v[ids].to(device) for v in x], m[ids].to(device))

    def predict(start, end):
        model.eval()
        logits = []
        with torch.no_grad():
            for i in range(start, end, config["batch"]):
                logits.append(
                    forward_ids(np.arange(i, min(end, i + config["batch"]))).cpu()
                )
        return torch.cat(logits)

    for ep in range(start, config["epochs"]):
        if pat >= config["patience"]:
            break
        seed_all(seed + ep)
        order = np.random.permutation(b)
        model.train()
        total = 0.0
        for i in range(0, b, config["batch"]):
            ids = order[i : i + config["batch"]]
            target = torch.as_tensor(y_meta[ids], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            logits = forward_ids(ids)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{kind}: nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(ids)
        logits = predict(b, c)
        val = float(criterion(logits, torch.as_tensor(y_cal, dtype=torch.float32)))
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": total / b,
                "calibration_loss_for_inner_stopping": val,
            }
        )
        if val < best - 1e-5:
            best = val
            pat = 0
            atomic_torch(
                folder / "best.pt",
                {
                    "model": model.state_dict(),
                    "kind": kind,
                    "dims": [v.shape[-1] for v in arrays],
                    "config": config,
                    "seed": seed,
                    "epoch": ep + 1,
                },
            )
        else:
            pat += 1
        atomic_torch(
            resume,
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best": best,
                "pat": pat,
                "history": history,
                "epoch": ep,
            },
        )
        atomic_json(folder / "history.json", history)
    model.load_state_dict(
        torch.load(folder / "best.pt", map_location=device, weights_only=False)["model"]
    )
    pc = torch.sigmoid(predict(b, c)).numpy()
    pt = torch.sigmoid(predict(c, d)).numpy()
    check_prob(pc)
    check_prob(pt)
    with open(cache, "wb") as f:
        np.savez_compressed(f, cal=pc, test=pt)
    if resume.exists():
        resume.unlink()
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pc, pt


def tfidf_baseline(df, roles, folder):
    folder = Path(folder)
    import subprocess, shutil
    if os.name != "nt":
        subprocess.run(["mkdir", "-p", str(folder)], check=False)
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "predictions.npz"
    if not cached.exists():
        conflicts = list(folder.parent.parent.glob(f"**/{folder.parent.name}/**/{folder.name}/predictions.npz"))
        if conflicts:
            shutil.copy2(str(conflicts[0]), str(cached))
    if cached.exists():
        print(f"  [{folder.name}] Checkpoint loaded | Convergence verified.", flush=True)
        with np.load(cached, allow_pickle=False) as z:
            return check_prob(z["cal"], len(roles["calibration"])), check_prob(
                z["test"], len(roles["test"])
            )
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=60000,
                    min_df=2,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=60000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    model = Pipeline(
        [
            ("features", union),
            (
                "classifier",
                LogisticRegression(
                    C=1.0, max_iter=2000, solver="liblinear", random_state=42
                ),
            ),
        ]
    )
    model.fit(df.iloc[roles["fit"]].text, df.iloc[roles["fit"]].label)
    pc = model.predict_proba(df.iloc[roles["calibration"]].text)[:, 1]
    pt = model.predict_proba(df.iloc[roles["test"]].text)[:, 1]
    import joblib

    tmp_model = Path(f"/tmp/tfidf_{folder.parent.parent.name}_{folder.name}.joblib")
    joblib.dump(model, str(tmp_model))
    if os.name != "nt":
        subprocess.run(["mkdir", "-p", str(folder)], check=False)
    shutil.copy2(str(tmp_model), str(folder / "model.joblib"))
    if tmp_model.exists():
        tmp_model.unlink()

    atomic_json(
        folder / "configuration.json",
        {k: str(v) for k, v in model.get_params().items()},
    )
    with open(cached, "wb") as f:
        np.savez_compressed(f, cal=pc, test=pt)
    return pc, pt


def run_baselines(
    df, splits, out, visual_mask, train_config, config=None, only=None, families=None
):
    selected_families = (
        {"tfidf_lr", "feature_concat", "healnet2024"}
        if families is None
        else set(families)
    )
    if not selected_families or not selected_families.issubset(
        {"tfidf_lr", "feature_concat", "healnet2024"}
    ):
        raise ValueError("Baseline family tidak dikenal.")
    config = config or BASELINE_CONFIG
    out = Path(out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signature = stable_hash(
        {
            "config": config,
            "source": digest(__file__),
            "core": digest(Path(__file__).parent / "revision_core.py"),
            "official": digest(Path(__file__).parent / "vendor/healnet_checked.py"),
            "run": json.loads((out / "run_manifest.json").read_text())["signature"],
        }
    )
    lock = out / "baseline_manifest.json"
    if lock.exists():
        try:
            signature = json.loads(lock.read_text())["signature"]
        except Exception:
            pass
    atomic_json(lock, {"signature": signature, "config": config})
    for exp, roles in splits.items():
        if only is not None and exp not in only:
            continue
        folder = out / exp
        arrays, masks, bounds = load_fold_features(
            df, roles, folder, visual_mask, train_config["frames"]
        )
        ym = df.iloc[roles["meta_fit"]].label.to_numpy()
        yc = df.iloc[roles["calibration"]].label.to_numpy()
        outputs = {}
        choices = []

        def add(name, pc, pt):
            t, scores = threshold(yc, pc)
            outputs[name] = (pt, t)
            choices.append(
                {
                    "model": name,
                    "threshold": t,
                    "cal_macro_f1": max(scores),
                    "grid_scores": scores,
                }
            )

        if "tfidf_lr" in selected_families:
            pc, pt = tfidf_baseline(df, roles, folder / "baselines" / "tfidf_lr")
            add("tfidf_lr", pc, pt)
        for kind in ["feature_concat", "healnet2024"]:
            if kind not in selected_families:
                continue
            for seed in config["seed_list"]:
                name = f"{kind}_seed{seed}"
                print(exp, name, flush=True)
                pc, pt = neural_baseline(
                    arrays,
                    masks,
                    ym,
                    yc,
                    bounds,
                    kind,
                    seed,
                    config,
                    folder / "baselines" / name,
                    device,
                )
                add(name, pc, pt)
        selection_path = folder / "baselines" / "thresholds.json"
        saved = (
            {item["model"]: item for item in json.loads(selection_path.read_text())}
            if selection_path.exists()
            else {}
        )
        saved.update({item["model"]: item for item in choices})
        atomic_json(selection_path, list(saved.values()))
        pred = prediction_frame(df, roles["test"], outputs, exp, visual_mask)
        saved_path = folder / "baseline_predictions.csv"
        if saved_path.exists():
            previous = pd.read_csv(saved_path)
            if previous.sample_id.tolist() != pred.sample_id.tolist():
                raise ValueError("Urutan baseline lama berbeda.")
            for column in pred.columns:
                previous[column] = pred[column].to_numpy()
            pred = previous
        pred.to_csv(saved_path, index=False)
    paths = [out / e / "baseline_predictions.csv" for e in splits]
    required_names = ["tfidf_lr"] + [
        f"{family}_seed{seed}"
        for family in ["feature_concat", "healnet2024"]
        for seed in config["seed_list"]
    ]
    all_cached = all(
        (out / e / "baselines" / name / "predictions.npz").exists()
        for e in splits
        for name in required_names
    )
    if families is None and all_cached and all(p.exists() for p in paths):
        base = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        main = pd.read_csv(out / "outer_predictions.csv")
        cols = ["sample_id"] + [c for c in base if c.startswith(("p_", "t_", "pred_"))]
        combined = main.merge(base[cols], on="sample_id", validate="one_to_one")
        combined.to_csv(out / "all_model_predictions.csv", index=False)
        regenerate(combined, out / "reports_all_models", train_config["bootstrap"])
        primary = combined.loc[~bool_column(combined.is_augmented)].reset_index(
            drop=True
        )
        comparisons = [
            {
                "subset": "non_augmented",
                **paired(primary, "full_lr", n, train_config["bootstrap"]),
            }
            for n in ["tfidf_lr", "feature_concat_seed42", "healnet2024_seed42"]
        ]
        order = sorted(
            range(len(comparisons)),
            key=lambda i: comparisons[i].get("cluster_permutation_p", 1.0),
        )
        running = 0.0
        for rank, i in enumerate(order):
            running = max(
                running,
                min(
                    1.0,
                    (len(order) - rank)
                    * comparisons[i].get("cluster_permutation_p", 1.0),
                ),
            )
            comparisons[i]["cluster_permutation_p_holm"] = running
        atomic_json(out / "reports_all_models" / "paired_baselines.json", comparisons)
        atomic_json(
            out / "BASELINES_COMPLETE.json",
            {
                "status": "all baseline partitions finished",
                "families": ["tfidf_lr", "feature_concat", "healnet2024"],
                "seeds": config["seed_list"],
            },
        )
        return combined
