"""
=============================================================================
TRAIN GNN MODEL — Dự báo rủi ro thanh lý Aave V2/V3

Các bước:
  0. Data audit + loại bỏ "ẩn crisis" trong nhãn normal
  1. Feature engineering: mã hóa categorical, log transform, scale
  2. Xử lý class imbalance: class_weight + SMOTE (cho MLP)
  3. Xây graph (PyTorch Geometric): node = vị thế, edge = shared collateral
  4. Train GCN 2 lớp
  5. Train MLP baseline (so sánh H2)
  6. Đánh giá: AUC-ROC, Precision, Recall, F1
  7. KernelSHAP: giải thích GCN (không so sánh với MLP — theo yêu cầu)
  8. Visualize toàn bộ kết quả
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                              roc_curve, precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import shap

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH  = r"D:\University\Ki 2 - Nam 3\PPNCKH\aave_model_ready_analysis_v2.csv"
OUTPUT_DIR = r"D:\University\Ki 2 - Nam 3\PPNCKH\model\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 0: ĐỌC DỮ LIỆU + DATA AUDIT + LOẠI "ẨN CRISIS"
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("BƯỚC 0: ĐỌC DỮ LIỆU & LÀM SẠCH NHÃN SNAPSHOT")
print("=" * 65)

df = pd.read_csv(DATA_PATH, low_memory=False)
df["observation_date"] = pd.to_datetime(
    df["observation_date"],
    dayfirst=True
    )
print(f"  Tổng rows ban đầu: {len(df):,}")

# -- Phát hiện các ngày "ẩn crisis" trong nhãn normal (liq rate bất thường) --
normal_mask = df["snapshot"] == "normal"
daily_stats = (df[normal_mask]
               .groupby("observation_date")["liquidated_next_3d"]
               .agg(["sum", "count"]))
daily_stats["rate"] = daily_stats["sum"] / daily_stats["count"]

# Ngưỡng: ngày có >= 30 quan sát VÀ liq rate > 10% được coi là "ẩn crisis"
# (10% là ngưỡng cao hơn rất nhiều so với baseline ~1.9% của toàn bộ normal)
hidden_crisis_dates = daily_stats[
    (daily_stats["count"] >= 30) & (daily_stats["rate"] > 0.10)
].index.tolist()

print(f"  Phát hiện {len(hidden_crisis_dates)} ngày 'ẩn crisis' trong nhãn normal:")
for d in sorted(hidden_crisis_dates)[:15]:
    r = daily_stats.loc[d]
    print(f"    {d.date()}  | liq_rate={r['rate']*100:.1f}% | n={int(r['count'])}")
if len(hidden_crisis_dates) > 15:
    print(f"    ... và {len(hidden_crisis_dates)-15} ngày khác")

# Loại các ngày này khỏi normal (không gán crisis, vì ta không chủ động chọn các mốc này)
before = len(df)
df = df[~((df["snapshot"] == "normal") &
          (df["observation_date"].isin(hidden_crisis_dates)))].copy()
print(f"  Loại bỏ {before - len(df):,} rows (ẩn crisis) → còn {len(df):,} rows")

print(f"\n  Snapshot distribution sau khi làm sạch:")
print(df["snapshot"].value_counts().to_string())
for s in ["normal", "crisis_1", "crisis_2"]:
    sub = df[df["snapshot"] == s]
    if len(sub) == 0:
        continue
    liq_pct = sub["liquidated_next_3d"].mean() * 100
    print(f"    {s}: n={len(sub):,} | liq={int(sub['liquidated_next_3d'].sum())} ({liq_pct:.3f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Trích chọn đặc trưng cho mô hình
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 1: FEATURE ENGINEERING")
print("=" * 65)

# -- 1a. Map address → symbol cho collateral/debt (để mã hóa categorical) --
TOKEN_SYMBOL = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": "wstETH",
    "0xae78736cd615f374d3085123a210448e74fc6393": "rETH",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a220612309bda2f40e6db16":  "USDT",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
    "0x514910771af9ca656af840dff83e8264ecf986ca": "LINK",
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": "AAVE",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "UNI",
    "0xd533a949740bb3306d119cc777fa900ba034cd52": "CRV",
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": "MKR",
}
df["collateral_symbol"] = df["collateral_asset_primary"].map(TOKEN_SYMBOL).fillna("OTHER")
df["debt_symbol"]       = df["debt_asset_primary"].map(TOKEN_SYMBOL).fillna("OTHER")

print(f"  Collateral symbol distribution:")
print(df["collateral_symbol"].value_counts().head(8).to_string())

# -- 1b. Mã hóa categorical --
le_col  = LabelEncoder()
le_dbt  = LabelEncoder()
le_prot = LabelEncoder()
df["collateral_enc"] = le_col.fit_transform(df["collateral_symbol"])
df["debt_enc"]        = le_dbt.fit_transform(df["debt_symbol"])
df["protocol_enc"]    = le_prot.fit_transform(df["protocol"])

# -- 1c. Clip outlier cho health_factor và distance_to_liquidation --
df["health_factor"] = df["health_factor"].clip(0, 10)
df["distance_to_liquidation_pct"] = df["distance_to_liquidation_pct"].clip(-100, 900)

# -- 1d. Danh sách feature cuối cùng --
RAW_FEATURES = [
    "health_factor", "liquidation_threshold", "distance_to_liquidation_pct",
    "log_collateral_usd", "log_debt_usd", "n_collateral_types",
    "tx_count_7d", "tx_count_30d", "inactive_days", "inactivity_flag_30d",
    "position_age_days",
    "fear_greed_index", "extreme_fear_flag",
    "collateral_enc", "debt_enc", "protocol_enc",
]
FEATURE_NAMES_VN = [
    "Health Factor", "Liquidation Threshold", "Distance to Liq (%)",
    "Log Collateral (USD)", "Log Debt (USD)", "N Collateral Types",
    "Tx Count 7d", "Tx Count 30d", "Inactive Days", "Inactivity Flag (>30d)",
    "Position Age (days)",
    "Fear & Greed Index", "Extreme Fear Flag",
    "Collateral Type", "Debt Asset Type", "Protocol (V2/V3)",
]
N_FEATURES = len(RAW_FEATURES)
print(f"\n  Tổng {N_FEATURES} features được dùng cho model.")

# -- 1e. Điền null --
for c in RAW_FEATURES:
    df[c] = df[c].fillna(df[c].median())

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 2: TRAIN / TEST SPLIT (theo thời gian) + SCALE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 2: TRAIN/TEST SPLIT & STANDARDSCALER")
print("=" * 65)

# Train = normal (đã làm sạch) + crisis_1 | Test = crisis_2
train_df = df[df["snapshot"].isin(["normal", "crisis_1"])].copy()
test_df  = df[df["snapshot"] == "crisis_2"].copy()

print(f"  Train: {len(train_df):,} rows | liq={int(train_df['liquidated_next_3d'].sum())} "
      f"({train_df['liquidated_next_3d'].mean()*100:.3f}%)")
print(f"  Test : {len(test_df):,} rows  | liq={int(test_df['liquidated_next_3d'].sum())} "
      f"({test_df['liquidated_next_3d'].mean()*100:.3f}%)")

# Subsample train nếu quá lớn (816k normal rows sẽ làm graph quá lớn cho 1 máy)
# Giữ tỷ lệ imbalance THỰC TẾ, không lấy hết toàn bộ positive nếu nó quá nhiều
MAX_TRAIN_TOTAL = 15000
TARGET_POS_RATIO = 0.05   # mô phỏng đúng tỷ lệ thực tế ~5% (gần với crisis_2 thật)

if len(train_df[train_df["snapshot"]=="normal"]) > 0:
    normal_part  = train_df[train_df["snapshot"]=="normal"]
    crisis1_part = train_df[train_df["snapshot"]=="crisis_1"]

    pos_all = normal_part[normal_part["liquidated_next_3d"]==1]
    neg_all = normal_part[normal_part["liquidated_next_3d"]==0]

    n_pos_target = min(len(pos_all), int(MAX_TRAIN_TOTAL * TARGET_POS_RATIO))
    n_neg_target = MAX_TRAIN_TOTAL - n_pos_target

    pos_sampled = pos_all.sample(n=n_pos_target, random_state=SEED)
    neg_sampled = neg_all.sample(n=min(n_neg_target, len(neg_all)), random_state=SEED)

    normal_sampled = pd.concat([pos_sampled, neg_sampled])
    train_df = pd.concat([normal_sampled, crisis1_part]).reset_index(drop=True)
    print(f"  → Subsample normal: giữ {n_pos_target:,} positive + {len(neg_sampled):,} negative "
          f"(tỷ lệ mục tiêu {TARGET_POS_RATIO*100:.0f}%)")
    print(f"  → Train sau subsample: {len(train_df):,} rows | "
          f"liq={int(train_df['liquidated_next_3d'].sum())} "
          f"({train_df['liquidated_next_3d'].mean()*100:.3f}%)")

test_df = test_df.reset_index(drop=True)

X_train_raw = train_df[RAW_FEATURES].values.astype(np.float32)
X_test_raw  = test_df[RAW_FEATURES].values.astype(np.float32)
y_train     = train_df["liquidated_next_3d"].values.astype(np.float32)
y_test      = test_df["liquidated_next_3d"].values.astype(np.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)

print(f"\n  Scaler fit trên {len(X_train):,} train rows, transform {len(X_test):,} test rows")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 3: XỬ LÝ CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 3: XỬ LÝ CLASS IMBALANCE")
print("=" * 65)

classes = np.array([0, 1])
cw = compute_class_weight("balanced", classes=classes, y=y_train)
pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32)
print(f"  class_weight = {{0: {cw[0]:.3f}, 1: {cw[1]:.3f}}} → pos_weight = {cw[1]/cw[0]:.2f}x")

# SMOTE cho MLP
n_pos = int(y_train.sum())
k_nb  = min(5, max(1, n_pos - 1))
sm = SMOTE(random_state=SEED, k_neighbors=k_nb, sampling_strategy=0.25)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"  SMOTE: {len(X_train):,} → {len(X_train_sm):,} rows "
      f"(liq: {y_train.mean()*100:.2f}% → {y_train_sm.mean()*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 4: XÂY GRAPH (PyTorch Geometric)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 4: XÂY GRAPH — Shared Collateral Edge")
print("=" * 65)

def build_edges(sub_df, max_pairs_per_group=400, max_users_per_group=60):
    """Tạo edge_index dựa trên cùng collateral_symbol (đồng thời cùng debt_symbol để giảm density)."""
    src, dst = [], []
    sub_df = sub_df.reset_index(drop=True)
    for (col_sym), grp in sub_df.groupby("collateral_symbol"):
        idxs = grp.index.tolist()[:max_users_per_group]
        cnt = 0
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                if cnt >= max_pairs_per_group:
                    break
                src.extend([idxs[i], idxs[j]])
                dst.extend([idxs[j], idxs[i]])
                cnt += 1
            if cnt >= max_pairs_per_group:
                break
    if not src:
        return torch.zeros((2,0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)

edge_index_train = build_edges(train_df)
edge_index_test  = build_edges(test_df)

g_train = Data(
    x=torch.tensor(X_train, dtype=torch.float32),
    edge_index=edge_index_train,
    y=torch.tensor(y_train, dtype=torch.float32),
)
g_test = Data(
    x=torch.tensor(X_test, dtype=torch.float32),
    edge_index=edge_index_test,
    y=torch.tensor(y_test, dtype=torch.float32),
)

print(f"  Train graph: {g_train.num_nodes:,} nodes | {g_train.num_edges:,} edges")
print(f"  Test  graph: {g_test.num_nodes:,} nodes | {g_test.num_edges:,} edges")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 5: ĐỊNH NGHĨA MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 5: ĐỊNH NGHĨA GCN VÀ MLP")
print("=" * 65)

class GCN(nn.Module):
    def __init__(self, in_ch, hidden=128, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, ei))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)

class MLP(nn.Module):
    def __init__(self, in_ch, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

print("  GCN : in=16 → GCNConv(128) → GCNConv(64) → Linear(32) → 1")
print("  MLP : in=16 → Linear(128) → Linear(64) → Linear(32) → 1")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 6: TRAIN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"BƯỚC 6: HUẤN LUYỆN (device={device})")
print("=" * 65)

def train_model(model, optimizer, data, pos_w, n_epochs=200, is_gcn=True):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    model.train()
    losses = []
    for ep in range(n_epochs):
        optimizer.zero_grad()
        logits = model(data.to(device)) if is_gcn else model(data.x.to(device))
        loss = criterion(logits, data.y.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (ep+1) % 50 == 0:
            print(f"    Epoch {ep+1:>3}/{n_epochs} | loss={loss.item():.4f}")
    return losses

def evaluate(model, data, is_gcn=True):
    model.eval()
    with torch.no_grad():
        logits = model(data.to(device)) if is_gcn else model(data.x.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        labels = data.y.numpy()
    auc = roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0
    ap  = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    rep = classification_report(labels, preds, output_dict=True, zero_division=0)
    fpr, tpr, _ = roc_curve(labels, probs)
    pc, rc, _   = precision_recall_curve(labels, probs)
    return dict(auc=auc, ap=ap, report=rep, probs=probs, preds=preds, labels=labels,
                fpr=fpr, tpr=tpr, prec_curve=pc, rec_curve=rc)

print("\n  --- GCN ---")
gcn = GCN(N_FEATURES).to(device)
gcn_opt = torch.optim.Adam(gcn.parameters(), lr=0.005, weight_decay=1e-4)
gcn_losses = train_model(gcn, gcn_opt, g_train, pos_weight, n_epochs=200, is_gcn=True)
gcn_train_res = evaluate(gcn, g_train, is_gcn=True)
gcn_test_res  = evaluate(gcn, g_test,  is_gcn=True)
print(f"  GCN: Train AUC={gcn_train_res['auc']:.4f} | Test AUC={gcn_test_res['auc']:.4f}")

print("\n  --- MLP (baseline, dùng SMOTE) ---")
X_tr_sm_t = torch.tensor(X_train_sm, dtype=torch.float32)
y_tr_sm_t = torch.tensor(y_train_sm, dtype=torch.float32)
g_mlp_train = Data(x=X_tr_sm_t, y=y_tr_sm_t, edge_index=torch.zeros((2,0),dtype=torch.long))
mlp = MLP(N_FEATURES).to(device)
mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.005, weight_decay=1e-4)
mlp_losses = train_model(mlp, mlp_opt, g_mlp_train, pos_weight, n_epochs=200, is_gcn=False)

g_test_mlp = Data(x=g_test.x, y=g_test.y, edge_index=torch.zeros((2,0),dtype=torch.long))
mlp_test_res = evaluate(mlp, g_test_mlp, is_gcn=False)
print(f"  MLP: Test AUC={mlp_test_res['auc']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 7: SO SÁNH GCN vs MLP (H2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 7: SO SÁNH GCN vs MLP (kiểm định H2)")
print("=" * 65)

gcn_r1 = gcn_test_res["report"].get("1", gcn_test_res["report"].get("1.0", {}))
mlp_r1 = mlp_test_res["report"].get("1", mlp_test_res["report"].get("1.0", {}))

print(f"\n  {'Model':10s} | {'AUC':6s} | {'AP':6s} | {'Prec':6s} | {'Recall':6s} | {'F1':6s}")
print("  " + "-"*55)
print(f"  {'GCN':10s} | {gcn_test_res['auc']:.4f} | {gcn_test_res['ap']:.4f} | "
      f"{gcn_r1.get('precision',0):.3f}  | {gcn_r1.get('recall',0):.3f}  | {gcn_r1.get('f1-score',0):.3f}")
print(f"  {'MLP':10s} | {mlp_test_res['auc']:.4f} | {mlp_test_res['ap']:.4f} | "
      f"{mlp_r1.get('precision',0):.3f}  | {mlp_r1.get('recall',0):.3f}  | {mlp_r1.get('f1-score',0):.3f}")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(gcn_losses, color="#534AB7", label="GCN", lw=2)
axes[0].plot(mlp_losses, color="#E24B4A", label="MLP", lw=2, ls="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(gcn_test_res["fpr"], gcn_test_res["tpr"], color="#534AB7", lw=2,
             label=f"GCN (AUC={gcn_test_res['auc']:.3f})")
axes[1].plot(mlp_test_res["fpr"], mlp_test_res["tpr"], color="#E24B4A", lw=2, ls="--",
             label=f"MLP (AUC={mlp_test_res['auc']:.3f})")
axes[1].plot([0,1],[0,1],"k--",lw=1,alpha=.4)
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curve (Test: crisis_2)"); axes[1].legend(); axes[1].grid(alpha=.3)

axes[2].plot(gcn_test_res["rec_curve"], gcn_test_res["prec_curve"], color="#534AB7", lw=2,
             label=f"GCN (AP={gcn_test_res['ap']:.3f})")
axes[2].plot(mlp_test_res["rec_curve"], mlp_test_res["prec_curve"], color="#E24B4A", lw=2, ls="--",
             label=f"MLP (AP={mlp_test_res['ap']:.3f})")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].set_title("Precision-Recall Curve"); axes[2].legend(); axes[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_training_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → Saved: 01_training_results.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, res) in zip(axes, [("GCN", gcn_test_res), ("MLP", mlp_test_res)]):
    cm = confusion_matrix(res["labels"], res["preds"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
    ax.set_title(f"{name} | AUC={res['auc']:.3f}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: 02_confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 8: KERNELSHAP — CHỈ GIẢI THÍCH GCN (không so sánh MLP)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 8: KERNELSHAP — GIẢI THÍCH MÔ HÌNH GCN")
print("=" * 65)

gcn.eval()

def gcn_predict_proba(X_np):
    """Wrapper model-agnostic cho KernelSHAP: nhận numpy, trả xác suất."""
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    n = X_t.shape[0]
    # Nếu kích thước khớp test graph, dùng đúng edge_index (giữ graph structure)
    if n == g_test.num_nodes:
        ei = g_test.edge_index.to(device)
    else:
        ei = torch.zeros((2,0), dtype=torch.long).to(device)
    with torch.no_grad():
        data_tmp = Data(x=X_t, edge_index=ei)
        probs = torch.sigmoid(gcn(data_tmp)).cpu().numpy()
    return probs

X_test_np = g_test.x.numpy()
n_bg = min(100, len(X_test_np))
bg_idx = np.random.choice(len(X_test_np), n_bg, replace=False)
X_bg = X_test_np[bg_idx]

N_EXPLAIN = min(300, len(X_test_np))
X_explain = X_test_np[:N_EXPLAIN]

print(f"  Background: {n_bg} samples | Explain: {N_EXPLAIN} samples")
print("  Đang tính KernelSHAP (có thể mất vài phút)...")

explainer   = shap.KernelExplainer(gcn_predict_proba, X_bg, link="identity")
shap_values = explainer.shap_values(X_explain, nsamples=80, silent=True)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
print(f"  SHAP values shape: {shap_values.shape}")

# -- Plot: feature importance bar --
mean_abs = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh([FEATURE_NAMES_VN[i] for i in order[::-1]],
               mean_abs[order[::-1]], color="#534AB7", alpha=.85, edgecolor="white")
for bar, v in zip(bars, mean_abs[order[::-1]]):
    ax.text(v + max(mean_abs)*0.01, bar.get_y()+bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP Feature Importance — GCN", fontweight="bold")
ax.grid(axis="x", alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_shap_importance_gcn.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: 03_shap_importance_gcn.png")

# -- Plot: beeswarm --
X_exp_df = pd.DataFrame(X_explain, columns=FEATURE_NAMES_VN)
plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_exp_df, feature_names=FEATURE_NAMES_VN,
                   show=False, plot_size=None)
plt.title("SHAP Beeswarm — GCN: mỗi điểm là 1 vị thế vay", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_shap_beeswarm_gcn.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: 04_shap_beeswarm_gcn.png")

# -- Plot: waterfall cho 1 high-risk và 1 low-risk node --
exp_probs  = gcn_predict_proba(X_explain)
high_i     = int(np.argmax(exp_probs))
low_i      = int(np.argmin(exp_probs))

base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(np.array(base_val).mean())

shap_exp = shap.Explanation(
    values=shap_values, base_values=np.full(len(shap_values), base_val),
    data=X_explain, feature_names=FEATURE_NAMES_VN,
)

for idx, label, fname in [
    (high_i, "Vị thế rủi ro cao", "05_shap_waterfall_highrisk.png"),
    (low_i,  "Vị thế an toàn",    "06_shap_waterfall_lowrisk.png"),
]:
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_exp[idx], show=False, max_display=12)
    plt.title(f"SHAP Waterfall — {label} (xác suất={exp_probs[idx]:.3f})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {fname}")

# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 9: LƯU KẾT QUẢ TỔNG HỢP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 9: LƯU KẾT QUẢ TỔNG HỢP")
print("=" * 65)

shap_df = pd.DataFrame({
    "Feature": FEATURE_NAMES_VN,
    "SHAP_mean_abs": mean_abs.round(6),
    "Rank": pd.Series(mean_abs).rank(ascending=False).astype(int).values,
}).sort_values("SHAP_mean_abs", ascending=False)
shap_df.to_csv(f"{OUTPUT_DIR}/shap_importance_gcn.csv", index=False)

metrics_df = pd.DataFrame([
    {"Model":"GCN", "AUC_ROC":gcn_test_res["auc"], "Avg_Precision":gcn_test_res["ap"],
     "Precision":gcn_r1.get("precision",0), "Recall":gcn_r1.get("recall",0), "F1":gcn_r1.get("f1-score",0)},
    {"Model":"MLP", "AUC_ROC":mlp_test_res["auc"], "Avg_Precision":mlp_test_res["ap"],
     "Precision":mlp_r1.get("precision",0), "Recall":mlp_r1.get("recall",0), "F1":mlp_r1.get("f1-score",0)},
]).round(4)
metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics_summary.csv", index=False)

print("\n  Files đã lưu:")
for f in ["01_training_results.png","02_confusion_matrix.png",
          "03_shap_importance_gcn.png","04_shap_beeswarm_gcn.png",
          "05_shap_waterfall_highrisk.png","06_shap_waterfall_lowrisk.png",
          "shap_importance_gcn.csv","model_metrics_summary.csv"]:
    print(f"    {f}")

print("\n" + "=" * 65)
print("TỔNG KẾT")
print("=" * 65)
print(metrics_df.to_string(index=False))
print("\n  Top 5 features (SHAP — GCN):")
for _, row in shap_df.head(5).iterrows():
    print(f"    #{row['Rank']:>2}  {row['Feature']:<25} SHAP={row['SHAP_mean_abs']:.4f}")
print("\n✓ Hoàn chỉnh.\n")
