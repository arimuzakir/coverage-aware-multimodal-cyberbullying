"""Reviewer experiment protocol. No test-label use in training or selection."""

from pathlib import Path
import os, re, json, hashlib, random, sys, platform, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import binomtest

VERSION = "v12-reviewer-1"
BRANCHES = ["indobert", "indobertweet", "mbert", "transcript", "visual"]


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def resolve_file(folder, canonical):
    folder = Path(folder)
    direct = folder / canonical
    if direct.is_file():
        return direct
    stem, suffix = Path(canonical).stem, Path(canonical).suffix
    matches = [
        p
        for p in folder.glob("*" + suffix)
        if re.fullmatch(
            re.escape(stem) + r"(?:\s*\(\d+\))*" + re.escape(suffix), p.name
        )
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{canonical}: ditemukan {len(matches)} kandidat di {folder}; isi DATA_DIR secara eksplisit."
        )
    return matches[0]


def find_data_dir(explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        resolve_file(p, "final_labeled_dataset_v5_clean.csv")
        return p
    bases = [
        Path.cwd(),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Othercomputers"),
    ]
    found = []
    for base in bases:
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base, followlinks=False):
            depth = len(Path(root).relative_to(base).parts)
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    "scraped_dataset",
                    "output",
                    "grafik",
                    "revision_runs",
                    "testdeps",
                    ".git",
                    "node_modules",
                }
                and depth < 7
            ]
            if any(
                re.fullmatch(r"final_labeled_dataset_v5_clean(?:\s*\(\d+\))*\.csv", f)
                for f in files
            ):
                found.append(Path(root))
    found = list(dict.fromkeys(p.resolve() for p in found))
    if len(found) != 1:
        raise RuntimeError(
            "DATA_DIR perlu diisi karena kandidat tidak tunggal: " + str(found)
        )
    return found[0]


def bool_column(s):
    x = s.astype(str).str.lower().str.strip()
    if not x.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("Mask harus boolean/0/1, bukan string bebas.")
    return x.isin(["true", "1", "1.0"]).to_numpy()


def make_groups(df):
    # Connected components: source video/thread, backtranslation parent, exact normalised text.
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    for i, row in df.iterrows():
        vid = str(row.video_id)
        base = vid[3:] if vid.startswith("bt_") else vid
        source = base
        if row.platform == "youtube":
            match = re.match(r"^yt_([A-Za-z0-9_-]{11})(?:_|$)", base)
            if match:
                source = "youtube:" + match.group(1)
            elif re.fullmatch(r"[A-Za-z0-9_-]{11}", base):
                source = "youtube:" + base
        text = re.sub(r"\s+", " ", str(row.text).strip().casefold())
        keys = ["source:" + source, "text:" + text] if text else ["source:" + source]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    return np.array([f"g{root(i):06d}" for i in range(len(df))])


def load_data(data_dir):
    data_dir = Path(data_dir)
    paths = {
        k: resolve_file(data_dir, v)
        for k, v in {
            "raw": "final_labeled_dataset_v5.csv",
            "clean": "final_labeled_dataset_v5_clean.csv",
            "audio": "audio_transcriptions_v10.csv",
        }.items()
    }
    raw = pd.read_csv(paths["raw"])
    df = pd.read_csv(paths["clean"])
    aud = pd.read_csv(paths["audio"])
    required = {
        "id",
        "video_id",
        "text",
        "label",
        "platform",
        "label_source",
        "has_audio",
        "has_video",
        "has_both",
    }
    if not required.issubset(df):
        raise ValueError("Kolom wajib hilang: " + str(required - set(df)))
    if df.id.duplicated().any() or df.video_id.duplicated().any():
        raise ValueError("ID sampel tidak unik.")
    if not set(df.label.unique()) == {0, 1}:
        raise ValueError("Label wajib biner 0/1.")
    if df.text.fillna("").str.strip().eq("").any():
        raise ValueError("Dataset clean masih mempunyai teks kosong.")
    if aud.video_id.duplicated().any():
        raise ValueError(
            "Transkripsi mempunyai ID duplikat; tidak boleh silent overwrite."
        )
    kept = raw[raw.id.isin(df.id)].reset_index(drop=True)
    if not kept.equals(df.reset_index(drop=True)):
        raise ValueError(
            "Dataset clean berbeda selain penghapusan baris; audit ulang sumber."
        )
    df = df.reset_index(drop=True)
    for col in ["has_audio", "has_video", "has_both"]:
        df[col] = bool_column(df[col])
    if not (df.has_both == (df.has_audio & df.has_video)).all():
        raise ValueError("has_both inkonsisten.")
    audio_map = dict(
        zip(aud.video_id.astype(str), aud.audio_text.fillna("").astype(str))
    )
    df["audio_text"] = df.video_id.astype(str).map(audio_map).fillna("").str.strip()
    df["audio_effective"] = df.audio_text.ne("")
    df["transcript_input"] = np.where(df.audio_effective, df.audio_text, df.text)
    df["sample_id"] = ["s" + str(int(v)).zfill(6) for v in df.id]
    df["group_id"] = make_groups(df)
    df["is_augmented"] = df.label_source.eq("backtranslation_augment")
    df["raw_row"] = raw.reset_index().set_index("id").loc[df.id, "index"].to_numpy()
    report = {
        "files": {k: {"name": p.name, "sha256": digest(p)} for k, p in paths.items()},
        "raw_n": len(raw),
        "clean_n": len(df),
        "removed_ids": raw.loc[~raw.id.isin(df.id), "id"].tolist(),
        "raw_class_counts": raw.label.value_counts().to_dict(),
        "clean_class_counts": df.label.value_counts().to_dict(),
        "nominal_raw": {
            "audio": int(raw.has_audio.sum()),
            "video": int(raw.has_video.sum()),
            "both": int((raw.has_audio & raw.has_video).sum()),
            "text_only": int((~(raw.has_audio | raw.has_video)).sum()),
        },
        "audio_effective": int(df.audio_effective.sum()),
        "groups": df.group_id.nunique(),
        "largest_group": int(df.group_id.value_counts().max()),
    }
    return df, report


def valid_roles(df, roles):
    values = list(roles.items())
    for name, ids in values:
        if len(ids) == 0 or df.iloc[ids].label.nunique() < 2:
            raise ValueError(f"Partisi {name} kosong/kelas tunggal.")
    for i, (an, a) in enumerate(values):
        for bn, b in values[i + 1 :]:
            if set(df.iloc[a].group_id) & set(df.iloc[b].group_id):
                raise ValueError(f"Kebocoran kelompok {an}/{bn}.")


def group_holdout(df, ids, frac, seed):
    ids = np.asarray(ids)
    sub = df.iloc[ids]
    choices = []
    # Select on split balance only, never on model performance.
    for tr, va in GroupShuffleSplit(
        n_splits=80, test_size=frac, random_state=seed
    ).split(sub, sub.label, sub.group_id):
        a, b = sub.iloc[tr], sub.iloc[va]
        if a.label.nunique() < 2 or b.label.nunique() < 2:
            continue
        score = abs(len(va) / len(sub) - frac) + abs(b.label.mean() - sub.label.mean())
        choices.append((score, tr, va))
    if not choices:
        raise ValueError("Tidak cukup kelompok/kelas untuk split internal.")
    _, tr, va = min(choices, key=lambda z: z[0])
    return ids[tr], ids[va]


def internal_roles(df, train, test, seed):
    remainder, cal = group_holdout(df, train, 0.15, seed)
    remainder, meta = group_holdout(df, remainder, 0.15 / 0.85, seed + 1)
    fit, stop = group_holdout(df, remainder, 0.10, seed + 2)
    roles = {
        "fit": fit,
        "early_stop": stop,
        "meta_fit": meta,
        "calibration": cal,
        "test": np.asarray(test),
    }
    valid_roles(df, roles)
    return {k: np.sort(v).tolist() for k, v in roles.items()}


def build_splits(df, seed=42, lopo=False):
    out = {}
    if lopo:
        for platform_name in sorted(df.platform.unique()):
            test = np.flatnonzero(df.platform.eq(platform_name))
            test_groups = set(df.iloc[test].group_id)
            train = np.flatnonzero(~df.group_id.isin(test_groups))
            out["lopo_" + platform_name] = internal_roles(df, train, test, seed + 100)
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(cv.split(df, df.label, df.group_id)):
            out[f"fold_{f + 1}"] = internal_roles(df, tr, te, seed + 10 * f)
        tests = np.concatenate([r["test"] for r in out.values()])
        assert sorted(tests) == list(range(len(df)))
    return out


def freeze_run(df, audit, out, config, lopo=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(
        {
            "version": VERSION,
            "source": audit["files"],
            "config": config,
            "lopo": lopo,
            "code": {
                p.name: digest(p) for p in Path(__file__).parent.glob("revision_*.py")
            },
        }
    )
    manifest = out / "run_manifest.json"
    splits_file = out / "splits.json"
    if splits_file.exists():
        splits = json.loads(splits_file.read_text())
        if manifest.exists():
            manifest_data = json.loads(manifest.read_text())
            if manifest_data.get("signature") != signature:
                manifest_data["signature"] = signature
                atomic_json(manifest, manifest_data)
        return splits, signature
    splits = build_splits(df, config["seed"], lopo)
    atomic_json(
        manifest,
        {
            "signature": signature,
            "version": VERSION,
            "config": config,
            "input_audit": audit,
            "protocol": "outer CV with disjoint inner fit/early_stop/meta_fit/calibration; not exhaustive inner k-fold",
            "lopo": lopo,
        },
    )
    atomic_json(out / "splits.json", splits)
    df[
        [
            "sample_id",
            "id",
            "video_id",
            "raw_row",
            "group_id",
            "platform",
            "label",
            "label_source",
            "is_augmented",
            "has_audio",
            "has_video",
            "audio_effective",
        ]
    ].to_csv(out / "sample_manifest_PRIVATE.csv", index=False)
    rows = [
        {
            "experiment": e,
            "role": role,
            "sample_id": df.iloc[i].sample_id,
            "group_id": df.iloc[i].group_id,
        }
        for e, r in splits.items()
        for role, ids in r.items()
        for i in ids
    ]
    pd.DataFrame(rows).to_csv(out / "partition_assignments.csv", index=False)
    return splits, signature


def check_prob(p, n=None):
    p = np.asarray(p, dtype=float)
    if (
        p.ndim != 1
        or (n is not None and len(p) != n)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
    ):
        raise ValueError("Probabilitas invalid; tidak boleh diganti 0.5 diam-diam.")
    return p


def threshold(y, p):
    p = check_prob(p, len(y))
    grid = np.arange(5, 96) / 100
    scores = np.array(
        [
            f1_score(y, p >= t, average="macro", labels=[0, 1], zero_division=0)
            for t in grid
        ]
    )
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    j = min(best, key=lambda j: (abs(grid[j] - 0.5), grid[j]))
    return float(grid[j]), scores.tolist()


def metrics(y, p, t):
    p = check_prob(p, len(y))
    pred = p >= t
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(
            y, pred, average="macro", labels=[0, 1], zero_division=0
        ),
        "macro_recall": recall_score(
            y, pred, average="macro", labels=[0, 1], zero_division=0
        ),
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else None,
        "average_precision": average_precision_score(y, p)
        if len(np.unique(y)) == 2
        else None,
    }


def fusion_outputs(y_meta, X_meta, y_cal, X_cal, X_test, out, seeds=(42, 43, 44)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    selection = []
    cal_outputs = {}

    def add(name, pc, pt, model=None):
        t, scores = threshold(y_cal, pc)
        outputs[name] = (check_prob(pt), t)
        cal_outputs["p_" + name] = check_prob(pc, len(y_cal))
        cal_outputs["t_" + name] = np.full(len(y_cal), t)
        selection.append(
            {
                "model": name,
                "threshold": t,
                "cal_macro_f1": max(scores),
                "threshold_scores": scores,
            }
        )
        if model is not None:
            import joblib

            joblib.dump(model, out / (name + ".joblib"))

    for j, name in enumerate(BRANCHES):
        add(name, X_cal[:, j], X_test[:, j])
    add("text_mean", X_cal[:, :3].mean(1), X_test[:, :3].mean(1))
    definitions = {
        "text_lr": [0, 1, 2],
        "text_speech_lr": [0, 1, 2, 3],
        "text_visual_lr": [0, 1, 2, 4],
        "speech_visual_lr": [3, 4],
        "full_lr": [0, 1, 2, 3, 4],
    }
    # Fixed C: no selecting the winner using outer test results.
    lr_models = {}
    for name, cols in definitions.items():
        m = LogisticRegression(
            C=0.5, max_iter=2000, solver="lbfgs", random_state=42
        ).fit(X_meta[:, cols], y_meta)
        lr_models[name] = m
        add(
            name,
            m.predict_proba(X_cal[:, cols])[:, 1],
            m.predict_proba(X_test[:, cols])[:, 1],
            m,
        )
    for seed in seeds:
        for name, cols, layers in [
            ("text_mlp", [0, 1, 2], (32, 16)),
            ("full_mlp", list(range(5)), (64, 32, 16)),
        ]:
            m = MLPClassifier(
                hidden_layer_sizes=layers,
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=min(200, len(y_meta)),
                learning_rate_init=0.001,
                max_iter=500,
                shuffle=True,
                random_state=seed,
                tol=1e-4,
                early_stopping=False,
                n_iter_no_change=10,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m.fit(X_meta[:, cols], y_meta)
            tag = name + f"_seed{seed}"
            add(
                tag,
                m.predict_proba(X_cal[:, cols])[:, 1],
                m.predict_proba(X_test[:, cols])[:, 1],
                m,
            )
            atomic_json(
                out / (tag + "_training.json"),
                {
                    "params": m.get_params(),
                    "loss_curve": m.loss_curve_,
                    "n_iter": m.n_iter_,
                    "warnings": [str(w.message) for w in caught],
                    "dropout": 0,
                    "initialization": "scikit-learn uniform initialization; saved sklearn version",
                    "stopping": "training loss tolerance, no outer test access",
                },
            )
    pd.DataFrame(cal_outputs).to_csv(out / "calibration_meta_outputs.csv", index=False)
    atomic_json(out / "thresholds.json", selection)
    atomic_json(
        out / "lr_coefficients.json",
        {
            k: {
                "features": [BRANCHES[i] for i in definitions[k]],
                "coef": v.coef_.tolist(),
                "intercept": v.intercept_.tolist(),
            }
            for k, v in lr_models.items()
        },
    )
    return outputs


def prediction_frame(df, ids, outputs, experiment, visual_mask):
    part = df.iloc[ids]
    r = (
        part[
            [
                "sample_id",
                "group_id",
                "platform",
                "label",
                "label_source",
                "is_augmented",
                "audio_effective",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )
    r["visual_effective"] = np.asarray(visual_mask)[ids]
    r["experiment"] = experiment
    for name, (p, t) in outputs.items():
        r["p_" + name] = check_prob(p, len(ids))
        r["t_" + name] = t
        r["pred_" + name] = (p >= t).astype(int)
    return r


def paired(df, a, b, n_boot=2000, seed=42):
    # Cluster bootstrap/permutation protects repeated texts and parent/derived items.
    if len(df) == 0:
        return {"n": 0, "status": "not_estimable_empty_subset"}
    y = df.label.to_numpy()
    pa = df["pred_" + a].to_numpy()
    pb = df["pred_" + b].to_numpy()
    groups = [np.flatnonzero(df.group_id.to_numpy() == g) for g in df.group_id.unique()]

    def mf(v, w):
        return f1_score(v, w, average="macro", labels=[0, 1], zero_division=0)

    obs = mf(y, pb) - mf(y, pa)
    rng = np.random.default_rng(seed)
    delta = []
    sa = []
    sb = []
    per = []
    for _ in range(n_boot):
        ids = np.concatenate(
            [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        )
        if len(np.unique(y[ids])) == 2:
            aa = mf(y[ids], pa[ids])
            bb = mf(y[ids], pb[ids])
            sa.append(aa)
            sb.append(bb)
            delta.append(bb - aa)
        swap = np.zeros(len(df), dtype=bool)
        for g, flag in zip(groups, rng.integers(2, size=len(groups))):
            if flag:
                swap[g] = True
        per.append(mf(y, np.where(swap, pa, pb)) - mf(y, np.where(swap, pb, pa)))
    ca, cb = pa == y, pb == y
    n01 = int((~ca & cb).sum())
    n10 = int((ca & ~cb).sum())
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist() if x else [None, None]
    return {
        "n": len(df),
        "groups": len(groups),
        "a": a,
        "b": b,
        "delta_macro_f1_b_minus_a": obs,
        "delta_ci95": ci(delta),
        "a_ci95": ci(sa),
        "b_ci95": ci(sb),
        "cluster_permutation_p": float(
            (1 + sum(abs(v) >= abs(obs) - 1e-12 for v in per)) / (len(per) + 1)
        ),
        "mcnemar_exact_p_descriptive": float(
            binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
        )
        if n01 + n10
        else 1.0,
        "a_wrong_b_right": n01,
        "a_right_b_wrong": n10,
        "net_correct_b_minus_a": n01 - n10,
        "total_disagreements": int((pa != pb).sum()),
        "bootstrap_valid": len(delta),
        "caveat": "Conditional uncertainty of stored predictions; not full retraining uncertainty. McNemar assumes independent items; cluster permutation is primary.",
    }


def regenerate(predictions, out, n_boot=2000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = (
        pd.read_csv(predictions)
        if not isinstance(predictions, pd.DataFrame)
        else predictions.copy()
    )
    if d.sample_id.duplicated().any():
        raise ValueError("Prediksi outer tidak unik; jangan campurkan CV dan LOPO.")
    names = [c[2:] for c in d if c.startswith("p_")]
    rows = []
    foldrows = []
    subsets = {
        "all_clean": np.ones(len(d), bool),
        "non_augmented": ~bool_column(d.is_augmented),
        "speech_real": bool_column(d.audio_effective),
        "visual_real": bool_column(d.visual_effective),
    }
    subsets.update(
        {
            "platform_" + p: d.platform.eq(p).to_numpy()
            for p in sorted(d.platform.unique())
        }
    )
    for subset, mask in subsets.items():
        s = d.loc[mask]
        if len(s) == 0:
            continue
        for name in names:
            rows.append(
                {
                    "subset": subset,
                    "model": name,
                    **metrics(s.label, s["p_" + name], s["t_" + name]),
                }
            )
    for exp, s in d.groupby("experiment"):
        for name in names:
            foldrows.append(
                {
                    "experiment": exp,
                    "model": name,
                    **metrics(s.label, s["p_" + name], s["t_" + name]),
                }
            )
    pd.DataFrame(rows).to_csv(out / "table2_metrics.csv", index=False)
    pd.DataFrame(foldrows).to_csv(out / "fold_metrics.csv", index=False)
    atomic_json(
        out / "metric_scale.json",
        {
            "all_scores": "0–1",
            "delta": "absolute 0–1; multiply by 100 for percentage points",
            "main_model": "full_lr fixed before running",
        },
    )
    cms = {}
    classrows = []
    for name in names:
        cms[name] = confusion_matrix(d.label, d["pred_" + name], labels=[0, 1]).tolist()
        cr = classification_report(
            d.label,
            d["pred_" + name],
            labels=[0, 1],
            target_names=["Non-bullying", "Bullying"],
            output_dict=True,
            zero_division=0,
        )
        for label in ["Non-bullying", "Bullying", "macro avg"]:
            classrows.append({"model": name, "class": label, **cr[label]})
    atomic_json(out / "confusion_matrices.json", cms)
    pd.DataFrame(classrows).to_csv(out / "table3_per_class.csv", index=False)
    comparisons = []
    for subset, a, b in [
        ("all_clean", "text_lr", "full_lr"),
        ("non_augmented", "text_lr", "full_lr"),
        ("speech_real", "text_lr", "text_speech_lr"),
        ("visual_real", "text_lr", "text_visual_lr"),
    ]:
        if {"pred_" + a, "pred_" + b}.issubset(d):
            comparisons.append(
                {
                    "subset": subset,
                    **paired(
                        d.loc[subsets[subset]].reset_index(drop=True), a, b, n_boot
                    ),
                }
            )
    # Holm familywise adjustment for the pre-specified four comparisons.
    available = [i for i, c in enumerate(comparisons) if "cluster_permutation_p" in c]
    order = sorted(available, key=lambda i: comparisons[i]["cluster_permutation_p"])
    running = 0
    for rank, i in enumerate(order):
        running = max(
            running,
            min(1.0, (len(order) - rank) * comparisons[i]["cluster_permutation_p"]),
        )
        comparisons[i]["cluster_permutation_p_holm"] = running
    atomic_json(out / "paired_uncertainty.json", comparisons)
    # Seed variation is across fusion fits only; never label it repeated base training.
    if rows:
        seed_rows = pd.DataFrame(rows)
        seed_rows = seed_rows[seed_rows.model.str.contains("_seed")]
        if len(seed_rows):
            seed_rows["family"] = seed_rows.model.str.replace(
                r"_seed\d+$", "", regex=True
            )
            seed_rows.groupby(["subset", "family"]).macro_f1.agg(
                ["count", "mean", "std", "min", "max"]
            ).to_csv(out / "fusion_seed_stability.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
    )
    for kind, Display in [
        ("roc", RocCurveDisplay),
        ("precision_recall", PrecisionRecallDisplay),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in ["indobert", "text_lr", "full_lr"]:
            if name in names:
                Display.from_predictions(d.label, d["p_" + name], name=name, ax=ax)
        fig.tight_layout()
        fig.savefig(out / (kind + ".png"), dpi=300)
        plt.close(fig)
    if "full_lr" in cms:
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            np.array(cms["full_lr"]), display_labels=["Non-bullying", "Bullying"]
        ).plot(ax=ax, colorbar=False)
        fig.tight_layout()
        fig.savefig(out / "confusion_full_lr.png", dpi=300)
        plt.close(fig)
    return pd.DataFrame(rows)
