"""
ENGINE INFERENSI REAL-TIME: DETEKSI CYBERBULLYING MULTIMODAL PADA ANAK
Model: full_mlp_seed44 (Multimodal Late Fusion MLP - Model Terbaik Tesis)
Fallback: text_mlp_seed44 / indobert (Untuk input teks murni tanpa media)

Penggunaan Cepat:
    from predict_realtime import CyberbullyingInferenceEngine
    
    engine = CyberbullyingInferenceEngine()
    result = engine.predict(
        text="kamu jelek banget gak guna mati aja", 
        audio_transcript=None
    )
    print(result)
    # Output: {
    #     'label': 'BULLYING',
    #     'probability': 0.9421,
    #     'confidence': 94.21,
    #     'threshold_used': 0.53,
    #     'model_used': 'text_mlp_seed44',
    #     'modality_breakdown': {'indobert': 0.92, ...}
    # }
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
FUSION_DIR = BASE_DIR / "fusion"
ENCODERS_DIR = BASE_DIR / "encoders"


class TextEncoder(nn.Module):
    """Arsitektur PyTorch Text Encoder (IndoBERT / IndoBERTweet / mBERT)"""
    def __init__(self, pretrained_model_path, num_drops=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        h = self.bert.config.hidden_size
        self.norm = nn.LayerNorm(h)
        self.cls = nn.Linear(h, 1)
        self.drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(num_drops)])

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.norm(
            (
                h[:, 0]
                + (h * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(1).clamp_min(1).unsqueeze(-1)
            )
            / 2
        )
        logits = torch.stack([self.cls(d(x)) for d in self.drops]).mean(0).squeeze(-1)
        return torch.sigmoid(logits), x


class CyberbullyingInferenceEngine:
    """Engine inferensi terintegrasi untuk Web Realtime."""
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[Engine] Menginisialisasi inference engine pada device: {self.device}")
        self._load_thresholds()
        self._load_models()
        print("[Engine] Inisialisasi selesai. Siap melayani inferensi real-time!")

    def _load_thresholds(self):
        t_path = FUSION_DIR / "thresholds.json"
        self.thresholds = {}
        if t_path.exists():
            data = json.loads(t_path.read_text(encoding="utf-8"))
            for item in data:
                self.thresholds[item["model"]] = float(item["threshold"])
        # Nilai default kalibrasi jika tidak ditemukan
        self.thresh_full_mlp = self.thresholds.get("full_mlp_seed44", 0.68)
        self.thresh_text_mlp = self.thresholds.get("text_mlp_seed44", 0.53)
        self.thresh_indobert = self.thresholds.get("indobert", 0.54)

    def _load_models(self):
        # 1. Load Fusion MLP
        self.full_mlp = joblib.load(FUSION_DIR / "full_mlp_seed44.joblib")
        self.text_mlp = joblib.load(FUSION_DIR / "text_mlp_seed44.joblib")

        # 2. Load IndoBERT (Backbone Teks Utama)
        indobert_dir = ENCODERS_DIR / "indobert"
        tok_dir = indobert_dir / "tokenizer"
        self.indobert_tok = AutoTokenizer.from_pretrained(
            str(tok_dir) if tok_dir.exists() else "indobenchmark/indobert-base-p1"
        )
        self.indobert_model = TextEncoder("indobenchmark/indobert-base-p1")
        if (indobert_dir / "best.pt").exists():
            st = torch.load(indobert_dir / "best.pt", map_location="cpu")
            self.indobert_model.load_state_dict(st, strict=False)
        self.indobert_model.to(self.device).eval()

        # 3. Load IndoBERTweet (Model Teks Bahasa Gaul/Sosmed)
        tweet_dir = ENCODERS_DIR / "indobertweet"
        tok_tweet = tweet_dir / "tokenizer"
        self.tweet_tok = AutoTokenizer.from_pretrained(
            str(tok_tweet) if tok_tweet.exists() else "indolem/indobertweet-base-uncased"
        )
        self.tweet_model = TextEncoder("indolem/indobertweet-base-uncased")
        if (tweet_dir / "best.pt").exists():
            st = torch.load(tweet_dir / "best.pt", map_location="cpu")
            self.tweet_model.load_state_dict(st, strict=False)
        self.tweet_model.to(self.device).eval()

        # 4. Load Transcript Model (Untuk Audio Ucapan)
        tr_dir = ENCODERS_DIR / "transcript"
        tok_tr = tr_dir / "tokenizer"
        self.tr_tok = AutoTokenizer.from_pretrained(
            str(tok_tr) if tok_tr.exists() else "indobenchmark/indobert-base-p1"
        )
        self.tr_model = TextEncoder("indobenchmark/indobert-base-p1")
        if (tr_dir / "best.pt").exists():
            st = torch.load(tr_dir / "best.pt", map_location="cpu")
            self.tr_model.load_state_dict(st, strict=False)
        self.tr_model.to(self.device).eval()

    def _encode_text(self, text, tokenizer, model, max_length=128):
        inputs = tokenizer(
            str(text),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            prob, _ = model(inputs["input_ids"], inputs["attention_mask"])
        return float(prob.cpu().item())

    def predict(self, text: str, audio_transcript: str = None, visual_feature_score: float = None):
        """
        Menjalankan inferensi cerdas dengan penanganan missing modality otomatis:
        - Jika audio_transcript diberikan: Menjalankan Tri-modal / Bi-modal Late Fusion (full_mlp_seed44).
        - Jika hanya text diberikan: Fallback mulus ke Text Late Fusion (text_mlp_seed44) & IndoBERT.
        """
        if not text or not str(text).strip():
            raise ValueError("Parameter 'text' tidak boleh kosong.")

        clean_text = str(text).strip()
        
        # 1. Hitung skor probabilitas unimodal teks
        p_indobert = self._encode_text(clean_text, self.indobert_tok, self.indobert_model)
        p_indobertweet = self._encode_text(clean_text, self.tweet_tok, self.tweet_model)
        p_mbert = (p_indobert + p_indobertweet) / 2.0  # aproksimasi ensemble

        has_audio = audio_transcript is not None and len(str(audio_transcript).strip()) > 0
        has_visual = visual_feature_score is not None

        # 2. Evaluasi berdasarkan ketersediaan modalitas
        if has_audio:
            p_transcript = self._encode_text(str(audio_transcript).strip(), self.tr_tok, self.tr_model)
            p_visual = float(visual_feature_score) if has_visual else 0.50

            # 5 features input untuk full_mlp: [indobert, indobertweet, mbert, transcript, visual]
            features = np.array([[p_indobert, p_indobertweet, p_mbert, p_transcript, p_visual]])
            final_prob = float(self.full_mlp.predict_proba(features)[0, 1])
            model_used = "full_mlp_seed44"
            threshold = self.thresh_full_mlp
        else:
            p_transcript = None
            p_visual = None
            # 3 features input untuk text_mlp: [indobert, indobertweet, mbert]
            features = np.array([[p_indobert, p_indobertweet, p_mbert]])
            final_prob = float(self.text_mlp.predict_proba(features)[0, 1])
            model_used = "text_mlp_seed44"
            threshold = self.thresh_text_mlp

        is_bullying = bool(final_prob >= threshold)
        label = "BULLYING" if is_bullying else "NON_BULLYING"
        confidence = float(final_prob if is_bullying else (1.0 - final_prob)) * 100.0

        return {
            "label": label,
            "is_bullying": is_bullying,
            "probability": round(final_prob, 4),
            "confidence_percent": round(confidence, 2),
            "threshold_used": round(threshold, 4),
            "model_used": model_used,
            "modality_scores": {
                "indobert_text": round(p_indobert, 4),
                "indobertweet_slang": round(p_indobertweet, 4),
                "audio_transcript": round(p_transcript, 4) if p_transcript is not None else "N/A",
                "visual_score": round(p_visual, 4) if p_visual is not None else "N/A",
            },
        }


if __name__ == "__main__":
    import sys
    engine = CyberbullyingInferenceEngine()
    
    sample_text = sys.argv[1] if len(sys.argv) > 1 else "dasar bocah tolol jelek banget lu gak guna"
    print("\n--- TEST SAMPLE INFERENCE ---")
    print(f"Input Text: {sample_text}")
    res = engine.predict(text=sample_text)
    print("Result:")
    print(json.dumps(res, indent=2))
