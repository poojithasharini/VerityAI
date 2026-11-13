from pathlib import Path
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model_artifacts" / "complete_hybrid_model_bundle.pt"
DEVICE = torch.device("cpu")

# These definitions intentionally match the architectures used in the Colab notebook.
class FeatureAttentionNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, input_dim)
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, embedding_dim), nn.ReLU()
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        embedding = self.encoder(x * weights)
        logits = self.classifier(embedding).squeeze(1)
        return logits, embedding, weights


class ProfileAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 24), nn.ReLU(), nn.Linear(24, input_dim))

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        return adjacency @ self.linear(x)


class ProfileGCN(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, 32)
        self.gc2 = GraphConvolution(32, embedding_dim)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x, adjacency):
        h = F.relu(self.gc1(x, adjacency))
        h = self.dropout(h)
        embedding = F.relu(self.gc2(h, adjacency))
        return self.classifier(embedding).squeeze(1), embedding


class HybridFusionClassifier(nn.Module):
    def __init__(self, attention_dim: int = 16, ae_dim: int = 8, gnn_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(attention_dim + ae_dim + 1 + gnn_dim, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, attention_emb, ae_emb, anomaly, gnn_emb):
        combined = torch.cat([attention_emb, ae_emb, anomaly.unsqueeze(1), gnn_emb], dim=1)
        return self.network(combined).squeeze(1)


app = FastAPI(title="VerityAI Profile Intelligence API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

BASELINES = [
    {"model":"Logistic Regression","accuracy":.9333,"precision":.9194,"recall":.9500,"f1":.9344,"auc":.9853},
    {"model":"Random Forest","accuracy":.9500,"precision":.9355,"recall":.9667,"f1":.9508,"auc":.9928},
    {"model":"SVM","accuracy":.9083,"precision":.9298,"recall":.8833,"f1":.9060,"auc":.9750},
    {"model":"XGBoost","accuracy":.9500,"precision":.9355,"recall":.9667,"f1":.9508,"auc":.9928},
]
ABLATION = [
    {"combination":"Attention","accuracy":.8750,"f1":.8718,"auc":.9439},
    {"combination":"Autoencoder","accuracy":.8917,"f1":.8926,"auc":.9500},
    {"combination":"GCN","accuracy":.9000,"f1":.8983,"auc":.9747},
    {"combination":"Attention + Autoencoder","accuracy":.8917,"f1":.8908,"auc":.9600},
    {"combination":"Attention + GCN","accuracy":.8917,"f1":.8889,"auc":.9678},
    {"combination":"Autoencoder + GCN","accuracy":.9167,"f1":.9167,"auc":.9769},
    {"combination":"All three branches","accuracy":.8917,"f1":.8908,"auc":.9700},
]

class Profile(BaseModel):
    profile_pic: int = Field(1, ge=0, le=1)
    username_digit_ratio: float = Field(.08, ge=0, le=1)
    fullname_words: int = Field(2, ge=0, le=20)
    fullname_digit_ratio: float = Field(0, ge=0, le=1)
    name_equals_username: int = Field(0, ge=0, le=1)
    description_length: int = Field(80, ge=0, le=5000)
    external_url: int = Field(0, ge=0, le=1)
    private: int = Field(0, ge=0, le=1)
    posts: int = Field(20, ge=0, le=1000000)
    followers: int = Field(100, ge=0, le=100000000)
    follows: int = Field(150, ge=0, le=1000000)


def _load_bundle():
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"Model bundle not found: {BUNDLE_PATH}")
    # This is a trusted local artifact exported by the project's own notebook.
    try:
        bundle = torch.load(BUNDLE_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        bundle = torch.load(BUNDLE_PATH, map_location=DEVICE)
    required = {"attention_model", "autoencoder", "gcn_model", "hybrid_model", "feature_columns",
                "scaler_mean", "scaler_scale", "X_all", "adjacency_normalized", "anomaly_threshold"}
    missing = required.difference(bundle.keys())
    if missing:
        raise RuntimeError(f"Bundle is missing required keys: {sorted(missing)}")
    dim = int(bundle.get("input_dim", len(bundle["feature_columns"])))
    if dim != 16:
        raise RuntimeError(f"Expected 16 model features, bundle says {dim}")

    attention = FeatureAttentionNet(dim).to(DEVICE)
    autoencoder = ProfileAutoencoder(dim).to(DEVICE)
    gcn = ProfileGCN(dim).to(DEVICE)
    fusion = HybridFusionClassifier().to(DEVICE)
    attention.load_state_dict(bundle["attention_model"])
    autoencoder.load_state_dict(bundle["autoencoder"])
    gcn.load_state_dict(bundle["gcn_model"])
    fusion.load_state_dict(bundle["hybrid_model"])
    for model in (attention, autoencoder, gcn, fusion):
        model.eval()

    return {
        "bundle": bundle, "attention": attention, "autoencoder": autoencoder,
        "gcn": gcn, "fusion": fusion,
        "feature_columns": bundle["feature_columns"],
        "mean": np.asarray(bundle["scaler_mean"], dtype=np.float32),
        "scale": np.asarray(bundle["scaler_scale"], dtype=np.float32),
        "x_all": np.asarray(bundle["X_all"], dtype=np.float32),
        "adj_norm": np.asarray(bundle["adjacency_normalized"], dtype=np.float32),
        "threshold": float(bundle["anomaly_threshold"]),
    }

try:
    MODELS = _load_bundle()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    MODELS = None
    MODEL_LOAD_ERROR = str(exc)


def _feature_row(p: Profile) -> np.ndarray:
    followers = float(p.followers)
    follows = float(p.follows)
    posts = float(p.posts)
    values = {
        "profile pic": float(p.profile_pic),
        "nums/length username": float(p.username_digit_ratio),
        "fullname words": float(p.fullname_words),
        "nums/length fullname": float(p.fullname_digit_ratio),
        "name==username": float(p.name_equals_username),
        "description length": float(p.description_length),
        "external URL": float(p.external_url),
        "private": float(p.private),
        "#posts": posts,
        "#followers": followers,
        "#follows": follows,
        "follower_following_ratio": followers / (follows + 1.0),
        "log_followers": float(np.log1p(max(followers, 0.0))),
        "log_following": float(np.log1p(max(follows, 0.0))),
        "log_posts": float(np.log1p(max(posts, 0.0))),
        "profile_completeness": float(np.mean([
            p.profile_pic, int(p.fullname_words > 0), int(p.description_length > 0), p.external_url
        ])),
    }
    cols = MODELS["feature_columns"]
    return np.asarray([values[c] for c in cols], dtype=np.float32).reshape(1, -1)


def _query_graph(x_query_scaled: np.ndarray):
    """Append query node to saved graph; connect it to its 5 nearest saved nodes.

    This is a serving-time transductive-style adaptation. It is not identical to
    the notebook's original fixed graph evaluation and should be validated separately.
    """
    x_ref = MODELS["x_all"]
    x_aug = np.vstack([x_ref, x_query_scaled]).astype(np.float32)
    n = len(x_ref)
    # Recover the original binary graph support from its normalized adjacency.
    adj = (MODELS["adj_norm"] != 0).astype(np.float32)
    adj_aug = np.zeros((n + 1, n + 1), dtype=np.float32)
    adj_aug[:n, :n] = adj
    nn_search = NearestNeighbors(n_neighbors=min(5, n), metric="euclidean")
    nn_search.fit(x_ref)
    _, inds = nn_search.kneighbors(x_query_scaled)
    for j in inds[0]:
        adj_aug[n, j] = 1.0
        adj_aug[j, n] = 1.0
    adj_aug[n, n] = 1.0
    degree = adj_aug.sum(axis=1)
    inv = np.power(np.maximum(degree, 1.0), -0.5)
    norm = inv[:, None] * adj_aug * inv[None, :]
    return torch.tensor(x_aug, dtype=torch.float32), torch.tensor(norm, dtype=torch.float32), inds[0]


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "trained_hybrid_model" if MODELS else "model_not_loaded",
            "model_loaded": MODELS is not None, "error": MODEL_LOAD_ERROR}


@app.get("/api/summary")
def summary():
    return {
        "dataset": {"total":696,"train":576,"test":120,"genuine":348,"fake":348},
        "baseline": BASELINES,
        "hybrid": {"accuracy":.8917,"precision":.8983,"recall":.8833,"f1":.8908,"auc":.9700},
        "ablation": ABLATION, "features":16,
        "branches":["Feature Attention","Autoencoder","Graph Convolution Network"],
        "note":"Recorded notebook metrics. Live profile analysis uses the exported trained hybrid model when model_loaded is true.",
        "model_loaded": MODELS is not None,
    }


@app.post("/api/predict")
def predict(p: Profile):
    if MODELS is None:
        raise HTTPException(503, f"Trained model bundle could not be loaded: {MODEL_LOAD_ERROR}")
    raw = _feature_row(p)
    scaled = ((raw - MODELS["mean"]) / np.where(MODELS["scale"] == 0, 1.0, MODELS["scale"])).astype(np.float32)
    x_one = torch.tensor(scaled, dtype=torch.float32, device=DEVICE)
    x_graph, a_graph, neighbor_ids = _query_graph(scaled)
    x_graph, a_graph = x_graph.to(DEVICE), a_graph.to(DEVICE)

    with torch.inference_mode():
        att_logits, att_emb, att_weights = MODELS["attention"](x_one)
        reconstruction, ae_emb = MODELS["autoencoder"](x_one)
        anomaly = torch.mean((x_one - reconstruction) ** 2, dim=1)
        gcn_logits_all, gcn_emb_all = MODELS["gcn"](x_graph, a_graph)
        gcn_logit = gcn_logits_all[-1:]
        gcn_emb = gcn_emb_all[-1:]
        fused_logit = MODELS["fusion"](att_emb, ae_emb, anomaly, gcn_emb)
        hybrid_prob = torch.sigmoid(fused_logit).item()
        attention_prob = torch.sigmoid(att_logits).item()
        gcn_prob = torch.sigmoid(gcn_logit).item()
        anomaly_score = anomaly.item()
        attention_values = att_weights[0].cpu().numpy()

    branch_probs = np.asarray([attention_prob, gcn_prob, hybrid_prob], dtype=np.float32)
    disagreement = float(np.std(branch_probs))
    top_idx = np.argsort(attention_values)[::-1][:5]
    top_features = [{"feature": MODELS["feature_columns"][int(i)], "weight": round(float(attention_values[i]), 4)} for i in top_idx]
    label = "Fake-like pattern" if hybrid_prob >= .5 else "Genuine-like pattern"
    risk = "High" if hybrid_prob >= .8 else ("Moderate" if hybrid_prob >= .5 else "Lower")
    evidence = []
    evidence.append("Autoencoder reconstruction error exceeds the genuine-calibration threshold." if anomaly_score >= MODELS["threshold"] else "Autoencoder reconstruction error is below the genuine-calibration threshold.")
    evidence.append("Model branches disagree; manual review is recommended." if disagreement >= .20 else "Attention, GCN and fusion outputs are relatively consistent.")
    evidence.append("Nearest profiles are feature-space neighbours from the saved dataset, not real Instagram connections.")
    return {
        "score": round(hybrid_prob, 4), "label": label, "risk_band": risk,
        "mode": "TRAINED HYBRID MODEL",
        "branches": {"attention_probability": round(attention_prob,4), "gcn_probability": round(gcn_prob,4),
                     "hybrid_probability": round(hybrid_prob,4), "anomaly_score": round(anomaly_score,6),
                     "anomaly_threshold": round(MODELS["threshold"],6), "anomaly_flag": bool(anomaly_score >= MODELS["threshold"])},
        "disagreement": round(disagreement,4), "manual_review": disagreement >= .20,
        "top_attention_features": top_features,
        "similar_training_profiles": [{"reference_index": int(i)} for i in neighbor_ids],
        "evidence": evidence,
        "disclaimer": "Experimental model output, not proof that an account is fake. The GCN query-time graph is an inference adaptation and requires separate validation. Do not use this score alone to accuse, restrict or report an account."
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")
    if df.empty:
        raise HTTPException(400, "CSV is empty.")
    counts = {}
    if "fake" in df.columns:
        counts = {str(k): int(v) for k,v in df.fake.value_counts(dropna=False).to_dict().items()}
    preview_df = df.head(8)
    return {"filename":file.filename,"rows":len(df),"columns":list(df.columns),
            "missing_values":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),
            "target_counts":counts,"preview":preview_df.astype(object).where(pd.notna(preview_df),None).to_dict("records"),
            "message":"CSV inspected in memory; it is not permanently stored."}
