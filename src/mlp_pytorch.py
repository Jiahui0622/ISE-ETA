"""Model M2: small neural network used alongside the event tree.
辅助神经网络（模型 M2），配合事件树使用。

ISE548 Risk and Crisis Management, Group 1.

Q1, severe harm: 1,271 incidents, 11 inputs, label = any death or
hospitalization. Ranks incidents after the fact; does not forecast leaks.
问题一（严重伤亡）：1,271 起事故，11 个输入，标签为是否有人死亡或住院。
只用于事后给事故排序，不预测哪里会泄漏。

Q2, ignition: 38 inputs known before ignition, no release volume. The 72
fire-first incidents are dropped, leaving 1,199.
问题二（是否点火）：38 个点火前就已知的输入，不用泄漏量；
去掉外部火灾先损坏管道的 72 起，剩 1,199 起。

Q1 inputs (Form F 7100.1) / 问题一的输入字段:
    UNINTENTIONAL_RELEASE (A7)  release volume, MCF, log(1 + x)        泄漏量
    ACCIDENT_PSIG (E1)          pressure at failure point, log(1 + x)  失效点压力
    INCIDENT_AREA_TYPE (B3)     underground / aboveground; transition area = reference
                                埋地 / 地上（过渡区作为参照类）
    CAUSE (Part G)              7 indicators; Other Incident Cause = reference
                                7 个原因指示变量（“其他原因”作为参照类）

Procedure / 流程:
    - 5 x stratified 10-fold CV = 50 models
      分层 10 折交叉验证，重复 5 次，共 50 个模型
    - each training part split 70/15/15 into fit / validation / calibration
      每折的训练部分按 70/15/15 分成拟合、验证、校准
    - scaling statistics from the fit part only
      标准化参数只用拟合部分计算
    - 11 -> 16 -> 8 -> 1 (385 parameters), full-batch Adam,
      BCE loss with pos_weight = N_neg / N_pos
      网络 11→16→8→1（385 个参数），全批量 Adam；严重事故在损失里加权
    - early stopping: patience 20, max 300 epochs, best weights restored
      提前停止：验证损失连续 20 轮不降就停（最多 300 轮），恢复最佳一轮的权重
    - Platt scaling on the calibration part
      在校准部分做 Platt 校准
    - baselines on the same folds: cause-only rate, logistic regression
      同一折里的对照：只按原因的比例、逻辑回归
    - final model: same procedure once on all incidents
      最终模型：对全部样本按同样流程训练一次

Seeds are fixed, so a rerun gives the same files. Final weights:
results/mlp_pytorch_checkpoint.pt (Q1), results/mlp_ignition_checkpoint.pt (Q2).
固定随机种子，重新运行结果相同。
两个最终模型的权重存在 results/mlp_pytorch_checkpoint.pt 和 mlp_ignition_checkpoint.pt。

Run from the folder above src/ / 在 src/ 的上一级目录（03_数据与代码）下运行:
    python -B src/mlp_pytorch.py
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# File locations. 文件位置。
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATA = BASE_DIR / "data" / "incident_gas_distribution_jan2010_present.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The 11 inputs, in the column order of the feature matrix X.
# 11 个输入特征，顺序就是特征矩阵 X 的列顺序。
FEATURE_NAMES = [
    "log_release_mcf",       # log(1 + UNINTENTIONAL_RELEASE)
    "log_accident_psig",     # log(1 + ACCIDENT_PSIG)
    "area_underground",      # INCIDENT_AREA_TYPE == UNDERGROUND
    "area_aboveground",      # INCIDENT_AREA_TYPE == ABOVEGROUND
    "cause_excavation",      # EXCAVATION DAMAGE
    "cause_outside_force",   # OTHER OUTSIDE FORCE DAMAGE
    "cause_incorrect_op",    # INCORRECT OPERATION
    "cause_pipe_weld",       # PIPE, WELD, OR JOINT FAILURE
    "cause_natural_force",   # NATURAL FORCE DAMAGE
    "cause_equipment",       # EQUIPMENT FAILURE
    "cause_corrosion",       # CORROSION FAILURE
]

# PHMSA's eight cause categories as written in the data file.
# 数据文件里 PHMSA 八类事故原因的原文。
RAW_CAUSE_CATEGORIES = [
    "EXCAVATION DAMAGE",
    "OTHER OUTSIDE FORCE DAMAGE",
    "INCORRECT OPERATION",
    "PIPE, WELD, OR JOINT FAILURE",
    "NATURAL FORCE DAMAGE",
    "OTHER INCIDENT CAUSE",
    "EQUIPMENT FAILURE",
    "CORROSION FAILURE",
]

# Evaluation and training settings, fixed before any results were seen.
# 评价与训练设置，在看到结果之前就已固定，不按结果回头调整。
N_FOLDS = 10
N_REPEATS = 5
BASE_SEED = 42
INNER_FRACTIONS = (0.70, 0.15, 0.15)   # fitting / validation / calibration  拟合 / 验证 / 校准
MAX_EPOCHS = 300
PATIENCE = 20                          # epochs without validation improvement  验证损失连续多少轮不降就停
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
LR_RIDGE = 1e-2                        # small ridge for the logistic-regression baseline  逻辑回归对照的岭惩罚
T_CRIT_95_DF49 = 2.0096                # two-sided 95% t quantile, 49 degrees of freedom  自由度 49 的 t 分位数


class GasRiskMLP(nn.Module):
    """Two hidden layers (16, 8), each with BatchNorm, ReLU and dropout.
    Small on purpose: a fit part has only about 180 severe incidents.
    两个隐藏层（16、8），每层后接 BatchNorm、ReLU 和 Dropout。
    每份拟合数据只有约 180 起严重事故，所以网络做得很小。

    Parameter count / 参数个数:
        Linear(11, 16) + BatchNorm1d(16): 11*16 + 16 + 32 = 224
        Linear(16, 8)  + BatchNorm1d(8):  16*8  + 8  + 16 = 152
        Linear(8, 1):                      8*1   + 1       = 9
        total                                                385
    """

    def __init__(self, input_dim: int = 11, hidden1: int = 16, hidden2: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden2, 1)

        # He init for the ReLU layers, Xavier for the output layer.
        # 对 ReLU 层用 He 初始化，对输出层用 Xavier 初始化。
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw logit z; sigmoid(z) is the uncalibrated score.
        # 输出原始 logit z；sigmoid(z) 是未校准的得分。
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        logits = self.fc3(out)
        return logits


def parse_float(val: Optional[str]) -> Optional[float]:
    """Text field -> number; empty or invalid gives None. 文本转数字，空值返回 None。"""
    if val is None:
        return None
    s = val.replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def clean_str(val: Optional[str]) -> str:
    """Trim and upper-case. 去空格并转大写。"""
    return val.strip().upper() if val else ""


def has_severe_harm(r: dict) -> bool:
    """Label: at least one death or hospitalization. 标签：至少一人死亡或住院。"""
    fat = parse_float(r.get("FATAL")) or 0.0
    inj = parse_float(r.get("INJURE")) or 0.0
    return fat > 0.0 or inj > 0.0


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC as the Mann-Whitney statistic; tied scores count 0.5.
    随机抽一起严重事故和一起非严重事故，前者得分更高的概率就是 ROC-AUC；得分相同记 0.5。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_mask = (y_true == 1.0)
    neg_mask = (y_true == 0.0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos = y_score[pos_mask][:, None]
    neg = y_score[neg_mask][None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[dict]]:
    """Build the 11 inputs and the severe-harm label for the primary sample.

    Same selection as the event tree: 2010-2024, natural gas, release volume
    > 0, and no IGNITE_IND = NO / EXPLODE_IND = YES conflict.
    为主样本构造 11 个输入和“严重伤亡”标签。筛选规则与事件树完全相同。
    """
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"PHMSA data file not found: {SOURCE_DATA}")

    with SOURCE_DATA.open("r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        raw_rows = list(reader)

    # Same four filters as eta_pipeline.py. 与 eta_pipeline.py 相同的四条筛选规则。
    clean_rows = [
        r for r in raw_rows
        if 2010 <= int(r["IYEAR"]) <= 2024
        and clean_str(r.get("COMMODITY_RELEASED_TYPE")) == "NATURAL GAS"
        and (parse_float(r.get("UNINTENTIONAL_RELEASE")) or 0.0) > 0.0
        and not (clean_str(r.get("IGNITE_IND")) == "NO" and clean_str(r.get("EXPLODE_IND")) == "YES")
    ]

    features = []
    labels = []

    for r in clean_rows:
        # Volume and pressure span several orders of magnitude, so use log(1 + x).
        # 泄漏量和压力跨越好几个数量级，取 log(1 + x) 压缩尺度。
        vol_mcf = max(0.0, parse_float(r.get("UNINTENTIONAL_RELEASE")) or 0.0)
        log_vol = math.log1p(vol_mcf)

        psig = max(0.0, parse_float(r.get("ACCIDENT_PSIG")) or 0.0)
        log_psig = math.log1p(psig)

        # Location as two 0/1 indicators; transition area is the reference (both 0).
        # 敷设位置用两个 0/1 变量表示；两个都为 0 即“过渡区”。
        area_type = clean_str(r.get("INCIDENT_AREA_TYPE"))
        is_underground = 1.0 if area_type == "UNDERGROUND" else 0.0
        is_aboveground = 1.0 if area_type == "ABOVEGROUND" else 0.0

        # Cause as seven 0/1 indicators; Other Incident Cause is the reference (all 0).
        # 原因用 7 个 0/1 变量表示；7 个都为 0 即“其他原因”。
        cause = clean_str(r.get("CAUSE"))
        if cause not in RAW_CAUSE_CATEGORIES:
            cause = "OTHER INCIDENT CAUSE"

        c_excavation = 1.0 if cause == "EXCAVATION DAMAGE" else 0.0
        c_outside_force = 1.0 if cause == "OTHER OUTSIDE FORCE DAMAGE" else 0.0
        c_incorrect_op = 1.0 if cause == "INCORRECT OPERATION" else 0.0
        c_pipe_weld = 1.0 if cause == "PIPE, WELD, OR JOINT FAILURE" else 0.0
        c_natural_force = 1.0 if cause == "NATURAL FORCE DAMAGE" else 0.0
        c_equipment = 1.0 if cause == "EQUIPMENT FAILURE" else 0.0
        c_corrosion = 1.0 if cause == "CORROSION FAILURE" else 0.0

        row_feat = [
            log_vol,
            log_psig,
            is_underground,
            is_aboveground,
            c_excavation,
            c_outside_force,
            c_incorrect_op,
            c_pipe_weld,
            c_natural_force,
            c_equipment,
            c_corrosion,
        ]
        features.append(row_feat)
        labels.append(1.0 if has_severe_harm(r) else 0.0)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # Consistency checks against the event tree counts.
    # 与事件树的计数逐项核对，确保两个模型用的是同一批事故。
    assert X.shape == (1271, 11), f"Feature matrix shape should be (1271, 11), got {X.shape}"
    assert y.shape == (1271,), f"Label vector shape should be (1271,), got {y.shape}"
    assert int(np.sum(y == 1.0)) == 287, f"Expected 287 severe incidents, got {int(np.sum(y == 1.0))}"
    assert int(np.sum(X[:, 4] == 1.0)) == 479, f"Expected 479 excavation incidents, got {int(np.sum(X[:, 4] == 1.0))}"
    assert int(np.sum(X[:, 5] == 1.0)) == 407, f"Expected 407 outside force incidents, got {int(np.sum(X[:, 5] == 1.0))}"
    assert int(np.sum(X[:, 6] == 1.0)) == 86, f"Expected 86 incorrect operation incidents, got {int(np.sum(X[:, 6] == 1.0))}"
    assert int(np.sum(X[:, 7] == 1.0)) == 83, f"Expected 83 pipe/weld/joint incidents, got {int(np.sum(X[:, 7] == 1.0))}"
    assert int(np.sum(X[:, 8] == 1.0)) == 82, f"Expected 82 natural force incidents, got {int(np.sum(X[:, 8] == 1.0))}"
    assert int(np.sum(X[:, 9] == 1.0)) == 48, f"Expected 48 equipment incidents, got {int(np.sum(X[:, 9] == 1.0))}"
    assert int(np.sum(X[:, 10] == 1.0)) == 28, f"Expected 28 corrosion incidents, got {int(np.sum(X[:, 10] == 1.0))}"
    ref_cause_count = int(np.sum(X[:, 4:11].sum(axis=1) == 0.0))
    assert ref_cause_count == 58, f"Expected 58 Other Incident Cause records, got {ref_cause_count}"
    col_stds = X.std(axis=0)
    assert np.all(col_stds > 0.0), f"Constant feature column(s): {np.where(col_stds == 0.0)[0]}"

    return X, y, FEATURE_NAMES, clean_rows


def fit_platt_calibration(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Platt scaling, P(y=1 | z) = 1 / (1 + exp(-(A*z + B))).

    Smoothed targets (Platt, 2000): (N+ + 1)/(N+ + 2) and 1/(N- + 2).
    Newton's method; the step is halved until the loss drops enough
    (Lin, Lin and Weng, 2007). Plain Newton diverged on some folds.
    用 Platt 方法校准：用两个参数 A、B 把网络输出 z 映射成概率。目标值做了平滑处理，
    避免在小样本上拟合得过于极端；用牛顿法迭代求解，每一步都检查交叉熵是否确实下降，
    不下降就把步长减半（Lin、Lin 和 Weng 2007 年的改进），防止迭代发散。
    """
    n_pos = np.sum(y_true == 1.0)
    n_neg = np.sum(y_true == 0.0)
    hi_target = (n_pos + 1.0) / (n_pos + 2.0)
    lo_target = 1.0 / (n_neg + 2.0)
    t = np.where(y_true == 1.0, hi_target, lo_target)

    def objective(a: float, b: float) -> float:
        # Cross-entropy against the smoothed targets: sum(log(1 + e^f) - t*f), f = a*z + b.
        # 对平滑目标的交叉熵，写成数值稳定的形式。
        f = logits * a + b
        return float(np.sum(np.logaddexp(0.0, f) - t * f))

    A = 1.0
    B = math.log((n_neg + 1.0) / (n_pos + 1.0)) if n_pos > 0 else 0.0
    current = objective(A, B)

    # Newton-Raphson iterations with backtracking. 带回溯线搜索的牛顿法迭代。
    for _ in range(100):
        f_ap = logits * A + B
        p = 1.0 / (1.0 + np.exp(-np.clip(f_ap, -30.0, 30.0)))
        d1 = p - t
        d2 = p * (1.0 - p)
        grad_A = np.dot(d1, logits)
        grad_B = np.sum(d1)
        H_AA = np.dot(d2, logits ** 2) + 1e-4
        H_BB = np.sum(d2) + 1e-4
        H_AB = np.dot(d2, logits)
        det = H_AA * H_BB - H_AB ** 2
        if abs(det) < 1e-9:
            break
        step_A = (H_BB * grad_A - H_AB * grad_B) / det
        step_B = (H_AA * grad_B - H_AB * grad_A) / det

        size = 1.0
        while size >= 1e-10:
            new_A, new_B = A - size * step_A, B - size * step_B
            new_value = objective(new_A, new_B)
            if new_value < current - 1e-4 * size * (grad_A * step_A + grad_B * step_B):
                break
            size /= 2.0
        else:
            break  # no step decreases the objective: already at the minimum / 已到最小值
        moved = (size * step_A) ** 2 + (size * step_B) ** 2
        A, B, current = new_A, new_B, new_value
        if moved < 1e-10:
            break

    return float(A), float(B)


def compute_reliability_bins(scores: np.ndarray, y_true: np.ndarray, n_bins: int = 5) -> List[Dict[str, Any]]:
    """Equal-count bins of the sorted scores, for the reliability diagram.
    按得分排序后分成等量的几组，比较每组的平均预测概率和实际严重比例（画可靠性图用）。"""
    groups = np.array_split(np.argsort(scores), n_bins)
    bin_results = []
    for b_idx, g in enumerate(groups, 1):
        g_scores = scores[g]
        g_y = y_true[g]
        bin_results.append({
            "bin_index": b_idx,
            "sample_count": len(g),
            "severe_count": int(np.sum(g_y)),
            "mean_prediction": round(float(np.mean(g_scores)), 4),
            "observed_fraction": round(float(np.mean(g_y)), 4),
            "min_score": round(float(np.min(g_scores)), 4),
            "max_score": round(float(np.max(g_scores)), 4),
        })
    return bin_results


def stratified_folds(y: np.ndarray, n_folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Split all row indices into n_folds parts with the same share of severe incidents.
    把全部样本分成 n_folds 份，每份里严重事故的比例相同（分层）。"""
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for label in (1.0, 0.0):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, n_folds)):
            folds[k].extend(part.tolist())
    return [np.array(sorted(f)) for f in folds]


def stratified_three_way(indices: np.ndarray, y: np.ndarray,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a training part into fitting / validation / calibration (about 70/15/15), stratified.
    把训练部分按约 70/15/15 分层分成拟合、验证、校准三块。"""
    fit, val, cal = [], [], []
    for label in (1.0, 0.0):
        idx = indices[y[indices] == label].copy()
        rng.shuffle(idx)
        n_fit = int(round(len(idx) * INNER_FRACTIONS[0]))
        n_val = int(round(len(idx) * INNER_FRACTIONS[1]))
        fit.extend(idx[:n_fit].tolist())
        val.extend(idx[n_fit:n_fit + n_val].tolist())
        cal.extend(idx[n_fit + n_val:].tolist())
    return np.array(sorted(fit)), np.array(sorted(val)), np.array(sorted(cal))


def train_with_early_stopping(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                              seed: int, log_history: bool = False) -> Dict[str, Any]:
    """Train one network with early stopping on the validation loss.
    训练一个网络；验证损失不再下降时停止，并恢复到验证损失最低那一轮的权重。

    Positives get pos_weight = N_neg / N_pos (about 3.4) so the network does not
    just predict "not severe". Same weight in the validation loss.
    严重事故在拟合部分中约占 22%，所以损失里给它加权 w = N_neg / N_pos，
    否则网络会倾向于一律预测“不严重”。验证损失使用同一个权重。
    """
    torch.manual_seed(seed)
    model = GasRiskMLP(input_dim=X_fit.shape[1], hidden1=16, hidden2=8, dropout_rate=DROPOUT)

    pos_weight_val = float(np.sum(y_fit == 0.0)) / float(np.sum(y_fit == 1.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    # Full-batch training: one gradient step per epoch on the whole fitting part.
    # 全批量训练：每一轮用整个拟合部分算一次梯度、更新一次参数。
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_fit_t), y_fit_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            if log_history:
                fit_logits = model(X_fit_t)
                history.append({
                    "epoch": epoch,
                    "fit_loss": round(criterion(fit_logits, y_fit_t).item(), 6),
                    "val_loss": round(val_loss, 6),
                    "fit_auc": round(compute_roc_auc(y_fit, torch.sigmoid(fit_logits).numpy().flatten()), 6),
                    "val_auc": round(compute_roc_auc(y_val, torch.sigmoid(val_logits).numpy().flatten()), 6),
                })

        # Early stopping on the validation loss. 按验证损失决定是否提前停止。
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "pos_weight": pos_weight_val,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val_loss": best_loss,
        "history": history,
    }


def predict_logits(model: GasRiskMLP, X_scaled: np.ndarray) -> np.ndarray:
    """Raw network outputs z for already standardised inputs. 对已标准化的输入输出原始 logit z。"""
    with torch.no_grad():
        return model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().flatten()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_procedure(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int,
                  log_history: bool = False, n_cont: int = 2) -> Dict[str, Any]:
    """One training part: split, standardise, train, calibrate.
    在一份训练数据上走完整个流程：三分、标准化、训练（提前停止）、Platt 校准。
    Used for every cross-validation fold and, once, for the final model.
    交叉验证的每一折和最终模型都用这同一个函数。"""
    rng = np.random.default_rng(seed)
    fit_idx, val_idx, cal_idx = stratified_three_way(train_idx, y, rng)

    # Standardise the first n_cont (continuous) columns with fitting-part statistics only.
    # Unknown values (NaN, e.g. an unknown installation year) are left out of the
    # statistics and then set to the fitting-part mean.
    # 前 n_cont 列连续变量做标准化，均值和标准差只用拟合部分计算；
    # 未知值（NaN，例如安装年份不详）不参与计算，之后用拟合部分的均值填补。
    mean = np.nanmean(X[fit_idx, :n_cont], axis=0)
    std = np.nanstd(X[fit_idx, :n_cont], axis=0)
    std = np.where(std == 0.0, 1.0, std)

    def scale(rows: np.ndarray) -> np.ndarray:
        Xs = X[rows].copy()
        cont = Xs[:, :n_cont]
        Xs[:, :n_cont] = (np.where(np.isnan(cont), mean, cont) - mean) / std
        return Xs

    trained = train_with_early_stopping(scale(fit_idx), y[fit_idx], scale(val_idx), y[val_idx],
                                        seed=seed, log_history=log_history)
    model = trained["model"]

    platt_A, platt_B = fit_platt_calibration(predict_logits(model, scale(cal_idx)), y[cal_idx])
    fit_auc = compute_roc_auc(y[fit_idx], predict_logits(model, scale(fit_idx)))
    val_auc = compute_roc_auc(y[val_idx], predict_logits(model, scale(val_idx)))

    return {
        **trained,
        "fit_idx": fit_idx, "val_idx": val_idx, "cal_idx": cal_idx,
        "mean": mean, "std": std, "scale": scale,
        "platt_A": platt_A, "platt_B": platt_B,
        "fit_auc": fit_auc, "val_auc": val_auc,
    }


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, ridge: float = LR_RIDGE) -> np.ndarray:
    """Baseline: logistic regression on the same inputs (Newton-Raphson, small ridge).
    对照方法：用同样 11 个输入的普通逻辑回归，牛顿法求解，加很小的岭惩罚防止系数发散。"""
    Xa = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xa.shape[1])
    R = ridge * np.eye(Xa.shape[1])
    R[0, 0] = 0.0  # no penalty on the intercept / 截距不加惩罚
    for _ in range(100):
        p = sigmoid(Xa @ w)
        grad = Xa.T @ (p - y) + R @ w
        hess = Xa.T @ (Xa * (p * (1.0 - p))[:, None]) + R
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def corrected_interval(diffs: np.ndarray, n_train: float, n_test: float) -> Tuple[float, float, float]:
    """Mean and 95% interval over J fold results.

    Folds share training data, so the variance factor is 1/J + n_test/n_train
    instead of 1/J (Nadeau and Bengio, 2003); t quantile with J - 1 = 49 df.
    重复交叉验证中各折的训练数据有重叠，普通标准误会偏小；按 Nadeau 和 Bengio (2003)
    的修正，把方差系数从 1/J 放大为 1/J + n_test/n_train，再用自由度 49 的 t 分位数。
    """
    J = len(diffs)
    mean = float(np.mean(diffs))
    var = float(np.var(diffs, ddof=1))
    half = T_CRIT_95_DF49 * math.sqrt((1.0 / J + n_test / n_train) * var)
    return mean, mean - half, mean + half


def summarise(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "sd": round(float(arr.std(ddof=1)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def run_cross_validation(X: np.ndarray, y: np.ndarray, causes: np.ndarray, report_numbers: List[str],
                         feature_groups: Dict[str, List[int]], n_cont: int = 2) -> Dict[str, Any]:
    """Repeated stratified 10-fold cross-validation of the network and the two baselines.
    分层 10 折交叉验证（重复 5 次），同时计算网络和两个对照方法的结果。"""
    n_samples = len(y)
    rng_perm = np.random.default_rng(1041)
    per_fold = []
    perm_drops: Dict[str, List[float]] = {g: [] for g in feature_groups}
    oof_rows = []
    oof_raw, oof_cal, oof_y = [], [], []

    for repeat in range(N_REPEATS):
        folds = stratified_folds(y, N_FOLDS, np.random.default_rng(BASE_SEED + repeat))
        for k, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
            seed = BASE_SEED + 1000 * (repeat + 1) + k
            proc = fit_procedure(X, y, train_idx, seed, n_cont=n_cont)
            y_test = y[test_idx]

            X_test = proc["scale"](test_idx)
            test_logits = predict_logits(proc["model"], X_test)
            raw = sigmoid(test_logits)
            cal = sigmoid(proc["platt_A"] * test_logits + proc["platt_B"])

            # Baselines are fitted on the whole training part.
            # 两个对照方法用整个训练部分拟合（它们不需要验证和校准数据）。
            prior = float(y[train_idx].mean())
            rates = {c: float(y[train_idx][causes[train_idx] == c].mean()) for c in set(causes[train_idx])}
            cause_scores = np.array([rates.get(c, prior) for c in causes[test_idx]])
            lr_w = fit_logistic_regression(proc["scale"](train_idx), y[train_idx])
            lr_prob = sigmoid(np.c_[np.ones(len(test_idx)), X_test] @ lr_w)

            mlp_auc = compute_roc_auc(y_test, raw)
            per_fold.append({
                "repeat": repeat + 1,
                "fold": k + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_severe": int(y_test.sum()),
                "n_fit": len(proc["fit_idx"]),
                "n_val": len(proc["val_idx"]),
                "n_cal": len(proc["cal_idx"]),
                "pos_weight": round(proc["pos_weight"], 6),
                "best_epoch": proc["best_epoch"],
                "epochs_run": proc["epochs_run"],
                "platt_A": round(proc["platt_A"], 6),
                "platt_B": round(proc["platt_B"], 6),
                "fit_auc": round(proc["fit_auc"], 6),
                "mlp_auc": round(mlp_auc, 6),
                "cause_only_auc": round(compute_roc_auc(y_test, cause_scores), 6),
                "logistic_auc": round(compute_roc_auc(y_test, lr_prob), 6),
                "brier_raw": round(float(np.mean((raw - y_test) ** 2)), 6),
                "brier_calibrated": round(float(np.mean((cal - y_test) ** 2)), 6),
                "brier_prior": round(float(np.mean((prior - y_test) ** 2)), 6),
                "brier_logistic": round(float(np.mean((lr_prob - y_test) ** 2)), 6),
                "mean_raw_score": round(float(raw.mean()), 6),
                "mean_calibrated_prob": round(float(cal.mean()), 6),
                "test_prevalence": round(float(y_test.mean()), 6),
            })

            # Permutation importance: shuffle one input group 10 times, mean AUC drop.
            # 置换重要性：在测试折上把一组输入打乱 10 次，记录 AUC 的平均下降。
            for g_name, cols in feature_groups.items():
                drops = []
                for _ in range(10):
                    X_p = X_test.copy()
                    X_p[:, cols] = X_p[rng_perm.permutation(len(X_p))][:, cols]
                    drops.append(mlp_auc - compute_roc_auc(y_test, predict_logits(proc["model"], X_p)))
                perm_drops[g_name].append(float(np.mean(drops)))

            for i, idx in enumerate(test_idx):
                oof_rows.append([report_numbers[idx], repeat + 1, k + 1, int(y[idx]),
                                 f"{raw[i]:.6f}", f"{cal[i]:.6f}", f"{cause_scores[i]:.6f}", f"{lr_prob[i]:.6f}"])
            oof_raw.extend(raw.tolist())
            oof_cal.extend(cal.tolist())
            oof_y.extend(y_test.tolist())
        print(f"  repeat {repeat + 1}/{N_REPEATS} done")

    # Summaries over the 50 folds. Differences are paired (same test fold).
    # 汇总 50 折的结果。差值是配对的：网络和对照方法在同一测试折上比较。
    col = lambda key: [f[key] for f in per_fold]  # noqa: E731
    n_train_mean = float(np.mean(col("n_train")))
    n_test_mean = float(np.mean(col("n_test")))

    def paired(a: str, b: str) -> Dict[str, Any]:
        d = np.array(col(a)) - np.array(col(b))
        mean, lo, hi = corrected_interval(d, n_train_mean, n_test_mean)
        return {"mean": round(mean, 6), "ci95": [round(lo, 6), round(hi, 6)],
                "folds_positive": int(np.sum(d > 0)), "folds_total": len(d)}

    mlp_mean, mlp_lo, mlp_hi = corrected_interval(np.array(col("mlp_auc")), n_train_mean, n_test_mean)
    repeat_means = [round(float(np.mean([f["mlp_auc"] for f in per_fold if f["repeat"] == r])), 6)
                    for r in range(1, N_REPEATS + 1)]
    summary = {
        "mlp_auc": {**summarise(col("mlp_auc")), "ci95": [round(mlp_lo, 6), round(mlp_hi, 6)]},
        "mlp_auc_by_repeat": repeat_means,
        "cause_only_auc": summarise(col("cause_only_auc")),
        "logistic_auc": summarise(col("logistic_auc")),
        "fit_auc": summarise(col("fit_auc")),
        "auc_gain_over_cause_only": paired("mlp_auc", "cause_only_auc"),
        "auc_gain_over_logistic": paired("mlp_auc", "logistic_auc"),
        "brier_raw": summarise(col("brier_raw")),
        "brier_calibrated": summarise(col("brier_calibrated")),
        "brier_prior": summarise(col("brier_prior")),
        "brier_logistic": summarise(col("brier_logistic")),
        "brier_calibrated_minus_prior": paired("brier_calibrated", "brier_prior"),
        "mean_raw_score": summarise(col("mean_raw_score")),
        "mean_calibrated_prob": summarise(col("mean_calibrated_prob")),
        "test_prevalence": summarise(col("test_prevalence")),
        "best_epoch": summarise(col("best_epoch")),
        "epochs_run": summarise(col("epochs_run")),
        "platt_A": summarise(col("platt_A")),
        "platt_B": summarise(col("platt_B")),
        "pos_weight": summarise(col("pos_weight")),
    }

    permutation_results = {}
    for g_name, drops in perm_drops.items():
        arr = np.array(drops)
        permutation_results[g_name] = {
            "mean_auc_drop": round(float(arr.mean()), 6),
            "sd_across_folds": round(float(arr.std(ddof=1)), 6),
            "min_drop": round(float(arr.min()), 6),
            "max_drop": round(float(arr.max()), 6),
            "note": "Mean over 50 test folds of the AUC drop averaged over 10 shuffles per fold.",
        }

    return {
        "per_fold": per_fold,
        "summary": summary,
        "oof_rows": oof_rows,
        "oof_raw": np.array(oof_raw),
        "oof_cal": np.array(oof_cal),
        "oof_y": np.array(oof_y),
        "permutation": permutation_results,
        "n_train_mean": n_train_mean,
        "n_test_mean": n_test_mean,
    }


def write_cv_predictions(path: Path, oof_rows: List[list]) -> None:
    """All out-of-fold predictions, one row per incident per repeat. 每次重复中每起事故的测试预测。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["REPORT_NUMBER", "REPEAT", "FOLD", "ACTUAL_LABEL", "RAW_SCORE",
                         "CALIBRATED_PROB", "CAUSE_ONLY_RATE", "LOGISTIC_PROB"])
        writer.writerows(oof_rows)
    print(f"Saved {path}")


def final_model_outputs(X: np.ndarray, y: np.ndarray, report_numbers: List[str], path: Path,
                        n_cont: int = 2) -> Tuple[Dict[str, Any], np.ndarray]:
    """Train the final model on all rows and save its predictions for every row.
    用全部样本训练最终模型，并保存它对每一条记录的预测。"""
    n_samples = len(y)
    final = fit_procedure(X, y, np.arange(n_samples), BASE_SEED, log_history=True, n_cont=n_cont)
    final_logits = predict_logits(final["model"], final["scale"](np.arange(n_samples)))
    final_raw = sigmoid(final_logits)
    final_cal = sigmoid(final["platt_A"] * final_logits + final["platt_B"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["REPORT_NUMBER", "PART", "ACTUAL_LABEL", "LOGIT", "RAW_SCORE", "CALIBRATED_PROB"])
        part = np.empty(n_samples, dtype=object)
        part[final["fit_idx"]], part[final["val_idx"]], part[final["cal_idx"]] = "FIT", "VALIDATION", "CALIBRATION"
        for i in range(n_samples):
            writer.writerow([report_numbers[i], part[i], int(y[i]),
                             f"{final_logits[i]:.6f}", f"{final_raw[i]:.6f}", f"{final_cal[i]:.6f}"])
    print(f"Saved {path}")
    return final, final_cal


def final_model_record(final: Dict[str, Any], y: np.ndarray, final_cal: np.ndarray) -> Dict[str, Any]:
    """Sizes, stopping epoch, scaling and Platt parameters of the final model. 最终模型的关键参数。"""
    return {
        "seed": BASE_SEED,
        "fit_size": len(final["fit_idx"]),
        "validation_size": len(final["val_idx"]),
        "calibration_size": len(final["cal_idx"]),
        "fit_severe": int(y[final["fit_idx"]].sum()),
        "validation_severe": int(y[final["val_idx"]].sum()),
        "calibration_severe": int(y[final["cal_idx"]].sum()),
        "pos_weight": round(final["pos_weight"], 6),
        "best_epoch": final["best_epoch"],
        "epochs_run": final["epochs_run"],
        "fit_auc": round(final["fit_auc"], 6),
        "validation_auc": round(final["val_auc"], 6),
        "scaler_mean": final["mean"].tolist(),
        "scaler_std": final["std"].tolist(),
        "platt_A": final["platt_A"],
        "platt_B": final["platt_B"],
        "mean_calibrated_prob_all": round(float(final_cal.mean()), 6),
        "note": "No separate test set; the cross-validation summary is the performance estimate for this procedure.",
    }


def model_metadata(input_dim: int, feature_names: List[str], n_params: int, purpose: str) -> Dict[str, Any]:
    """Model description and the software environment of this run. 模型说明和本次运行的软件环境。"""
    with SOURCE_DATA.open("rb") as f_data:
        raw_data_hash = hashlib.sha256(f_data.read()).hexdigest()
    return {
        "model_type": "PyTorch Multilayer Perceptron (MLP)",
        "purpose": purpose,
        "input_dimension": input_dim,
        "architecture": "Dense(16) -> BN -> ReLU -> Dropout(0.1) -> Dense(8) -> BN -> ReLU -> Dropout(0.1) -> Dense(1)",
        "total_trainable_parameters": n_params,
        "feature_names": feature_names,
        "optimiser": f"Adam, learning rate {LEARNING_RATE}, weight decay {WEIGHT_DECAY}, full batch",
        "early_stopping": f"validation loss, patience {PATIENCE}, at most {MAX_EPOCHS} epochs, best-epoch weights restored",
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "os": platform.platform(),
            "raw_dataset_sha256": raw_data_hash,
        },
    }


LOSS_FORMULATION = {
    "loss_name": "nn.BCEWithLogitsLoss(pos_weight=w)",
    "pos_weight_formula": "w = N_neg / N_pos of the fitting part",
    "loss_mathematical_formula": "L(z, y) = (1 - y) * softplus(z) + w * y * softplus(-z)",
    "softplus_definition": "softplus(t) = log(1 + exp(t)) = max(t, 0) + log1p(exp(-abs(t)))",
}


def cv_protocol(n_samples: int, n_positive: int, cv: Dict[str, Any], label: str, n_inputs: int) -> Dict[str, Any]:
    """Description of the cross-validation set-up stored with the results. 随结果保存的交叉验证设置说明。"""
    return {
        "total_sample": n_samples,
        "total_severe" if label == "severe" else "total_positive": n_positive,
        "folds": N_FOLDS,
        "repeats": N_REPEATS,
        "fold_split_seeds": [BASE_SEED + r for r in range(N_REPEATS)],
        "model_seed_rule": "BASE_SEED + 1000 * repeat + (fold - 1), repeat counted from 1",
        "inner_split": "fitting / validation / calibration = 70 / 15 / 15 within each training part, stratified",
        "stratification": ("Stratified by the severe-harm label (FATAL > 0 or INJURE > 0)" if label == "severe"
                           else "Stratified by the ignition label (IGNITE_IND = YES)"),
        "baselines": {
            "cause_only": f"{'severe' if label == 'severe' else 'ignition'} rate of each cause in the training part",
            "logistic_regression": f"same {n_inputs} inputs, unweighted, ridge {LR_RIDGE} (intercept unpenalised), fitted on the training part",
            "prior": "training-part prevalence given to every test incident (Brier only)",
        },
        "interval_method": "corrected repeated k-fold t interval (Nadeau and Bengio, 2003), 49 df",
        "mean_train_size": round(cv["n_train_mean"], 1),
        "mean_test_size": round(cv["n_test_mean"], 1),
    }


def train_and_evaluate_mlp() -> dict:
    """Question 1, severe harm: cross-validate, train the final model and save all outputs.
    问题一（是否严重伤亡）：交叉验证、训练最终模型，并保存全部结果。"""
    torch.set_num_threads(1)  # same results on every run / 单线程保证每次结果一致
    X, y, feature_names, rows = load_dataset()
    causes = np.array([clean_str(r.get("CAUSE")) for r in rows])
    report_numbers = [r["REPORT_NUMBER"].strip() for r in rows]
    n_samples = len(y)

    # Input groups for permutation importance. The 7 cause indicators move together;
    # shuffling one alone gives rows with two causes.
    # 置换重要性的输入分组。7 个原因变量必须一起打乱；单独打乱一个会出现“一起事故有两个原因”。
    feature_groups = {
        "all_causes_joint_group": [4, 5, 6, 7, 8, 9, 10],
        "log_accident_psig": [1],
        "log_release_mcf": [0],
        "location_type_group": [2, 3],
    }

    # 1. Repeated stratified 10-fold cross-validation.
    #    第 1 步：分层 10 折交叉验证，重复 5 次。
    print("Severe-harm model: cross-validation")
    cv = run_cross_validation(X, y, causes, report_numbers, feature_groups)

    # 2. Final model on all 1,271 incidents, with its predictions.
    #    第 2 步：用全部 1,271 起事故训练最终模型，并保存预测。
    final, final_cal = final_model_outputs(X, y, report_numbers, RESULTS_DIR / "final_model_predictions.csv")

    # 3. Reliability bins from all out-of-fold predictions (5 repeats x 1,271).
    #    第 3 步：用全部交叉验证测试预测（5 次 × 1,271 条）做可靠性分箱。
    raw_bins = compute_reliability_bins(cv["oof_raw"], cv["oof_y"], n_bins=10)
    cal_bins = compute_reliability_bins(cv["oof_cal"], cv["oof_y"], n_bins=10)

    # How many two-cause rows appear if the excavation column is shuffled alone.
    # 记录单独打乱“开挖”一列会产生多少条不合理数据，作为原因整组打乱的依据。
    X_copy = X.copy()
    X_copy[:, 4] = X_copy[np.random.default_rng(1041).permutation(n_samples), 4]
    invalid_single_cause_rows = int(np.sum(X_copy[:, 4:].sum(axis=1) > 1.0))

    # 4. Save outputs. 第 4 步：保存结果。
    checkpoint_path = RESULTS_DIR / "mlp_pytorch_checkpoint.pt"
    torch.save(final["model"].state_dict(), checkpoint_path)
    print(f"Saved {checkpoint_path}")
    write_cv_predictions(RESULTS_DIR / "cv_predictions.csv", cv["oof_rows"])

    cal_bins_path = RESULTS_DIR / "calibration_bins.json"
    with cal_bins_path.open("w", encoding="utf-8") as f:
        json.dump({
            "source": "all out-of-fold predictions, 5 repeats x 1,271 incidents",
            "n_predictions": len(cv["oof_y"]),
            "prevalence": round(float(cv["oof_y"].mean()), 6),
            "n_bins": 10,
            "raw_bins": raw_bins,
            "calibrated_bins": cal_bins,
            "pooled_brier": {
                "raw": round(float(np.mean((cv["oof_raw"] - cv["oof_y"]) ** 2)), 6),
                "calibrated": round(float(np.mean((cv["oof_cal"] - cv["oof_y"]) ** 2)), 6),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {cal_bins_path}")

    results_payload = {
        "model_metadata": model_metadata(11, FEATURE_NAMES, sum(p.numel() for p in final["model"].parameters()),
                                         "Auxiliary post-incident severity ranking for reported incidents"),
        "loss_formulation": LOSS_FORMULATION,
        "cross_validation": {
            "protocol": cv_protocol(n_samples, int(y.sum()), cv, "severe", 11),
            "summary": cv["summary"],
            "per_fold": cv["per_fold"],
        },
        "final_model": final_model_record(final, y, final_cal),
        "permutation_feature_importance": cv["permutation"],
        "permutation_methodological_notes": {
            "invalid_single_excavation_permutation_rows": invalid_single_cause_rows,
            "explanation": "Shuffling one cause indicator on its own creates rows with two causes, so the joint cause group is the measure used for cause importance.",
        },
        "training_history": final["history"],
    }

    json_path = RESULTS_DIR / "mlp_pytorch_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_path}")

    return results_payload


# Question 2, ignition. Only facts that exist before ignition: place, pipe, system,
# cause. No release volume: it is estimated afterwards and depends on whether the gas burned.
# 问题二（泄漏后是否点火）。输入只用点火之前就已存在的事实：位置、管道、系统和原因。
# 泄漏量不用，因为它是事后估算的，而且受气体是否燃烧的影响。
EXTERNAL_FIRE_DETAIL = "NEARBY INDUSTRIAL, MAN-MADE, OR OTHER FIRE/EXPLOSION AS PRIMARY CAUSE"

# (input name, PHMSA field, values counted as 1). Records with none of the listed
# values are the reference category of that field.
# （输入名、PHMSA 字段、记为 1 的取值）。不属于任何一项的记录就是该字段的参照类。
IGNITION_CATEGORICAL = [
    ("loc_private_property", "LOCATION_TYPE", {"PRIVATE PROPERTY"}),
    ("loc_public_property", "LOCATION_TYPE", {"PUBLIC PROPERTY"}),
    ("loc_utility_row", "LOCATION_TYPE", {"UTILITY RIGHT-OF-WAY / EASEMENT"}),
    ("sub_under_soil", "INCIDENT_AREA_SUBTYPE", {"UNDER SOIL"}),
    ("sub_aboveground_piping", "INCIDENT_AREA_SUBTYPE",
     {"TYPICAL ABOVEGROUND FACILITY PIPING OR APPURTENANCE (E.G. VALUE OR REGULATOR STATION, OUTDOOR METER SET)"}),
    ("sub_under_pavement", "INCIDENT_AREA_SUBTYPE", {"UNDER PAVEMENT"}),
    ("sub_exposed_by_excavation", "INCIDENT_AREA_SUBTYPE", {"EXPOSED DUE TO EXCAVATION"}),
    ("sub_in_or_under_building", "INCIDENT_AREA_SUBTYPE", {"INSIDE A BUILDING", "UNDER A BUILDING"}),
    ("part_main", "SYSTEM_PART_INVOLVED", {"MAIN", "MAIN VALVE"}),
    ("part_service", "SYSTEM_PART_INVOLVED", {"SERVICE"}),
    ("part_service_riser", "SYSTEM_PART_INVOLVED", {"SERVICE RISER"}),
    ("part_meter_set", "SYSTEM_PART_INVOLVED",
     {"OUTSIDE METER/REGULATOR SET", "INSIDE METER/REGULATOR SET", "FARM TAP METER/REGULATOR SET"}),
    ("part_district_station", "SYSTEM_PART_INVOLVED", {"DISTRICT REGULATOR/METERING STATION"}),
    ("mat_steel", "MATERIAL_INVOLVED", {"STEEL"}),
    ("mat_plastic", "MATERIAL_INVOLVED", {"PLASTIC"}),
    ("mat_iron", "MATERIAL_INVOLVED", {"CAST/WROUGHT IRON", "DUCTILE IRON"}),
    ("rel_mechanical_puncture", "RELEASE_TYPE", {"MECHANICAL PUNCTURE"}),
    ("rel_leak", "RELEASE_TYPE", {"LEAK"}),
    ("rel_rupture", "RELEASE_TYPE", {"RUPTURE"}),
    ("scada_in_place", "SCADA_IN_PLACE_IND", {"YES"}),
    ("crossing", "CROSSING", {"YES"}),
    ("owner_municipal", "PIPE_FACILITY_TYPE", {"MUNICIPALLY OWNED"}),
    ("owner_investor", "PIPE_FACILITY_TYPE", {"INVESTOR OWNED"}),
]
IGNITION_FEATURE_NAMES = (
    ["log_accident_psig", "class_location", "log_pipe_age"]          # continuous / 连续变量
    + ["area_underground", "area_aboveground"] + FEATURE_NAMES[4:]   # as in question 1 / 与问题一相同
    + [name for name, _, _ in IGNITION_CATEGORICAL]
    + ["pipe_age_unknown", "night_22_to_6", "winter_dec_to_feb"]
)
IGNITION_N_CONT = 3

# Input groups for permutation importance (column positions in IGNITION_FEATURE_NAMES).
# 置换重要性的输入分组（对应 IGNITION_FEATURE_NAMES 中的列位置）。
IGNITION_GROUPS = {
    "pressure": [0],
    "class_location": [1],
    "pipe_age": [2, 35],
    "area_type": [3, 4],
    "cause": list(range(5, 12)),
    "location_type": [12, 13, 14],
    "area_subtype": list(range(15, 20)),
    "system_part": list(range(20, 25)),
    "material": [25, 26, 27],
    "release_type": [28, 29, 30],
    "scada": [31],
    "crossing": [32],
    "owner": [33, 34],
    "time_of_event": [36, 37],
}


def build_ignition_inputs(X11: np.ndarray, rows: List[dict]) -> np.ndarray:
    """The 38 pre-ignition inputs for every primary-sample incident. 为主样本每起事故构造 38 个点火前输入。"""
    out = []
    for i, r in enumerate(rows):
        cls = clean_str(r.get("CLASS_LOCATION_TYPE"))
        class_no = float(cls.split()[1]) if cls.startswith("CLASS") else np.nan

        # Pipe age in years at the incident; unknown installation years stay NaN.
        # 事故发生时的管龄（年）；安装年份不详的记为 NaN，训练时用拟合部分均值填补。
        year = r.get("INSTALLATION_YEAR", "").strip()
        known = year.isdigit() and 1850 < int(year) <= int(r["IYEAR"])
        log_age = math.log1p(int(r["IYEAR"]) - int(year)) if known else np.nan

        date_part, time_part = r["LOCAL_DATETIME"].split()
        hour = int(time_part.split(":")[0])
        month = int(date_part.split("/")[0])

        row = [X11[i, 1], class_no, log_age] + X11[i, 2:].tolist()
        row += [1.0 if clean_str(r.get(field)) in values else 0.0 for _, field, values in IGNITION_CATEGORICAL]
        row += [0.0 if known else 1.0, 1.0 if (hour >= 22 or hour < 6) else 0.0, 1.0 if month in (12, 1, 2) else 0.0]
        out.append(row)
    X = np.array(out, dtype=np.float32)
    assert X.shape[1] == len(IGNITION_FEATURE_NAMES)
    # Each listed category must match at least one record (catches typos).
    # 每个列出的类别至少要匹配到一条记录，防止取值拼写错误导致整列为 0。
    empty = [IGNITION_FEATURE_NAMES[j] for j in range(12, 35) if X[:, j].sum() == 0]
    assert not empty, f"Categories with no records: {empty}"
    return X


def adjusted_probabilities(final: Dict[str, Any], X: np.ndarray, cols: List[int],
                           settings: Dict[str, List[float]]) -> Dict[str, float]:
    """Mean calibrated probability with one input group set to the same value for
    every incident. Other inputs stay as recorded.
    把某一组输入对所有事故统一设成某个取值，其余输入保持原样，求平均校准概率；
    这样比较时其他因素保持不变。
    """
    out = {}
    for label, values in settings.items():
        X_set = X.copy()
        X_set[:, cols] = np.array(values, dtype=np.float32)
        Xs = X_set.copy()
        cont = Xs[:, :IGNITION_N_CONT]
        Xs[:, :IGNITION_N_CONT] = (np.where(np.isnan(cont), final["mean"], cont) - final["mean"]) / final["std"]
        z = predict_logits(final["model"], Xs)
        out[label] = round(float(sigmoid(final["platt_A"] * z + final["platt_B"]).mean()), 6)
    return out


def train_and_evaluate_ignition() -> dict:
    """Question 2, ignition: which pre-ignition conditions go with ignition.
    问题二（是否点火）：哪些点火前的条件让泄漏更容易点火。

    The 72 fire-first incidents (nearby fire damaged the pipe) are left out.
    外部火灾或爆炸先损坏管道的 72 起事故不进入主分析，因为这些事故是先着火、后泄漏。
    """
    torch.set_num_threads(1)
    X11, _, _, all_rows = load_dataset()
    X_all = build_ignition_inputs(X11, all_rows)
    ign_all = np.array([1.0 if clean_str(r.get("IGNITE_IND")) == "YES" else 0.0 for r in all_rows], dtype=np.float32)
    fire_first = np.array([EXTERNAL_FIRE_DETAIL in clean_str(r.get("CAUSE_DETAILS")) for r in all_rows])
    causes_all = np.array([clean_str(r.get("CAUSE")) for r in all_rows])
    numbers_all = [r["REPORT_NUMBER"].strip() for r in all_rows]

    keep = np.flatnonzero(~fire_first)
    X, y, causes = X_all[keep], ign_all[keep], causes_all[keep]
    report_numbers = [numbers_all[i] for i in keep]
    assert int(fire_first.sum()) == 72, f"Expected 72 fire-first incidents, got {int(fire_first.sum())}"

    # 1. Cross-validation on the 1,199 incidents where the release came first.
    #    第 1 步：在先泄漏的 1,199 起事故上做交叉验证。
    print("Ignition model: cross-validation")
    cv = run_cross_validation(X, y, causes, report_numbers, IGNITION_GROUPS, n_cont=IGNITION_N_CONT)

    # 2. Sensitivity check: all 1,271 incidents, fire-first cases included.
    #    第 2 步：敏感性检验，把先着火的 72 起也放进来。
    print("Ignition model: sensitivity check with all 1,271 incidents")
    cv_all = run_cross_validation(X_all, ign_all, causes_all, numbers_all, {}, n_cont=IGNITION_N_CONT)

    # 3. Final model; adjusted probabilities for system part, location type,
    #    material and pressure (other inputs as recorded).
    #    第 3 步：最终模型和“其他因素不变”的点火概率：把管段类型、位置类型、管材、压力分别设成
    #    不同取值，其余输入保持原样，求平均预测点火概率。
    final, final_cal = final_model_outputs(X, y, report_numbers, RESULTS_DIR / "final_model_predictions_ignition.csv",
                                           n_cont=IGNITION_N_CONT)
    part_cols = IGNITION_GROUPS["system_part"]
    loc_cols = IGNITION_GROUPS["location_type"]
    mat_cols = IGNITION_GROUPS["material"]
    psig_q = np.quantile(X[:, 0], [0.1, 0.5, 0.9])
    adjusted = {
        "system_part": adjusted_probabilities(final, X, part_cols, {
            "main": [1, 0, 0, 0, 0], "service": [0, 1, 0, 0, 0], "service_riser": [0, 0, 1, 0, 0],
            "meter_set": [0, 0, 0, 1, 0], "district_station": [0, 0, 0, 0, 1], "other": [0, 0, 0, 0, 0]}),
        "location_type": adjusted_probabilities(final, X, loc_cols, {
            "private_property": [1, 0, 0], "public_property": [0, 1, 0], "utility_row": [0, 0, 1],
            "operator_controlled": [0, 0, 0]}),
        "material": adjusted_probabilities(final, X, mat_cols, {
            "steel": [1, 0, 0], "plastic": [0, 1, 0], "iron": [0, 0, 1], "other_or_unknown": [0, 0, 0]}),
        "pressure": adjusted_probabilities(final, X, [0], {
            f"p10_{math.expm1(psig_q[0]):.1f}_psig": [psig_q[0]],
            f"p50_{math.expm1(psig_q[1]):.1f}_psig": [psig_q[1]],
            f"p90_{math.expm1(psig_q[2]):.1f}_psig": [psig_q[2]]}),
    }

    # Observed ignition rates for the same groups.
    # 同样分组的实际点火比例，用来和“其他因素不变”的结果对照。
    def observed(cols: List[int], labels: List[str]) -> Dict[str, Dict[str, float]]:
        res = {}
        masks = [X[:, j] == 1.0 for j in cols] + [X[:, cols].sum(axis=1) == 0]  # last label = reference
        for lab, m in zip(labels, masks):
            res[lab] = {"n": int(m.sum()), "ignition_rate": round(float(y[m].mean()), 6)}
        return res

    observed_rates = {
        "system_part": observed(part_cols, ["main", "service", "service_riser", "meter_set", "district_station", "other"]),
        "location_type": observed(loc_cols, ["private_property", "public_property", "utility_row", "operator_controlled"]),
        "material": observed(mat_cols, ["steel", "plastic", "iron", "other_or_unknown"]),
    }

    # 4. Save outputs. 第 4 步：保存结果。
    checkpoint_path = RESULTS_DIR / "mlp_ignition_checkpoint.pt"
    torch.save(final["model"].state_dict(), checkpoint_path)
    print(f"Saved {checkpoint_path}")
    write_cv_predictions(RESULTS_DIR / "cv_predictions_ignition.csv", cv["oof_rows"])

    s_all = cv_all["summary"]
    results_payload = {
        "model_metadata": model_metadata(len(IGNITION_FEATURE_NAMES), IGNITION_FEATURE_NAMES,
                                         sum(p.numel() for p in final["model"].parameters()),
                                         "Which pre-ignition conditions make ignition of a reported release more likely"),
        "loss_formulation": LOSS_FORMULATION,
        "sample": {
            "primary_sample": len(all_rows),
            "fire_first_excluded": int(fire_first.sum()),
            "fire_first_ignited": int(ign_all[fire_first].sum()),
            "analysis_sample": len(keep),
            "ignited": int(y.sum()),
            "ignition_rate": round(float(y.mean()), 6),
            "fire_first_rule": f"CAUSE_DETAILS contains '{EXTERNAL_FIRE_DETAIL}'",
            "release_volume_used": False,
        },
        "cross_validation": {
            "protocol": cv_protocol(len(keep), int(y.sum()), cv, "ignition", len(IGNITION_FEATURE_NAMES)),
            "summary": cv["summary"],
            "per_fold": cv["per_fold"],
        },
        "sensitivity_all_1271": {
            "mlp_auc": s_all["mlp_auc"],
            "cause_only_auc": s_all["cause_only_auc"],
            "logistic_auc": s_all["logistic_auc"],
            "auc_gain_over_cause_only": s_all["auc_gain_over_cause_only"],
        },
        "final_model": final_model_record(final, y, final_cal),
        "permutation_feature_importance": cv["permutation"],
        "adjusted_ignition_probability": adjusted,
        "observed_ignition_rate": observed_rates,
        "calibration_bins": {
            "raw_bins": compute_reliability_bins(cv["oof_raw"], cv["oof_y"], n_bins=10),
            "calibrated_bins": compute_reliability_bins(cv["oof_cal"], cv["oof_y"], n_bins=10),
        },
        "training_history": final["history"],
    }

    json_path = RESULTS_DIR / "mlp_ignition_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_path}")

    return results_payload


def print_summary(title: str, results: dict) -> None:
    """Print the cross-validation summary of one question. 打印一个问题的交叉验证摘要。"""
    s = results["cross_validation"]["summary"]
    fm = results["final_model"]
    pfi = results["permutation_feature_importance"]

    def ms(d):
        return f"{d['mean']:.4f} ± {d['sd']:.4f}"

    print(f"\n{title} ({N_REPEATS} x {N_FOLDS}-fold cross-validation)")
    print(f"  Inputs / parameters: {results['model_metadata']['input_dimension']} / "
          f"{results['model_metadata']['total_trainable_parameters']}")
    print(f"  Network AUC:         {ms(s['mlp_auc'])}  (95% CI {s['mlp_auc']['ci95'][0]:.4f} to {s['mlp_auc']['ci95'][1]:.4f})")
    print(f"  AUC by repeat:       {', '.join(f'{v:.4f}' for v in s['mlp_auc_by_repeat'])}")
    print(f"  Cause-only AUC:      {ms(s['cause_only_auc'])}")
    print(f"  Logistic AUC:        {ms(s['logistic_auc'])}")
    for key, label in (("auc_gain_over_cause_only", "Gain vs cause-only"), ("auc_gain_over_logistic", "Gain vs logistic")):
        g = s[key]
        print(f"  {label + ':':<20} {g['mean']:+.4f} (95% CI {g['ci95'][0]:+.4f} to {g['ci95'][1]:+.4f}; "
              f"{g['folds_positive']}/{g['folds_total']} folds > 0)")
    print(f"  Fitting-part AUC:    {ms(s['fit_auc'])}")
    print(f"  Brier raw / calibrated / prior / logistic: {s['brier_raw']['mean']:.4f} / {s['brier_calibrated']['mean']:.4f} / "
          f"{s['brier_prior']['mean']:.4f} / {s['brier_logistic']['mean']:.4f}")
    b = s["brier_calibrated_minus_prior"]
    print(f"  Calibrated - prior:  {b['mean']:+.4f} (95% CI {b['ci95'][0]:+.4f} to {b['ci95'][1]:+.4f})")
    print(f"  Mean score raw / calibrated / prevalence: {s['mean_raw_score']['mean']:.4f} / "
          f"{s['mean_calibrated_prob']['mean']:.4f} / {s['test_prevalence']['mean']:.4f}")
    print(f"  Best epoch:          {ms(s['best_epoch'])} (range {s['best_epoch']['min']:.0f} to {s['best_epoch']['max']:.0f})")
    print(f"  Final model:         best epoch {fm['best_epoch']} of {fm['epochs_run']}, fit AUC {fm['fit_auc']:.4f}, "
          f"validation AUC {fm['validation_auc']:.4f}, Platt A {fm['platt_A']:.4f}, B {fm['platt_B']:.4f}")
    print("  Permutation importance (mean AUC drop):")
    for k, v in sorted(pfi.items(), key=lambda kv: -kv[1]["mean_auc_drop"]):
        print(f"    {k + ':':<24} {v['mean_auc_drop']:.4f} ± {v['sd_across_folds']:.4f}")


def main():
    """Run both questions and print their summaries. 运行两个问题并打印摘要。"""
    severe = train_and_evaluate_mlp()
    ignition = train_and_evaluate_ignition()
    print_summary("Model M2, question 1: severe harm", severe)
    print_summary("Model M2, question 2: ignition (fire-first incidents excluded)", ignition)
    sens = ignition["sensitivity_all_1271"]
    print(f"  Sensitivity, all 1,271 incidents: network AUC {sens['mlp_auc']['mean']:.4f}, "
          f"cause-only {sens['cause_only_auc']['mean']:.4f}, logistic {sens['logistic_auc']['mean']:.4f}")
    print("  Adjusted ignition probability (other inputs as recorded):")
    for group, values in ignition["adjusted_ignition_probability"].items():
        print(f"    {group}: " + ", ".join(f"{k} {v:.3f}" for k, v in values.items()))
    print("  Observed ignition rate:")
    for group, values in ignition["observed_ignition_rate"].items():
        print(f"    {group}: " + ", ".join(f"{k} {v['ignition_rate']:.3f} (n={v['n']})" for k, v in values.items()))


if __name__ == "__main__":
    main()
