"""Check that the saved network and the cross-validation results can be reproduced.
复核：用已保存的网络和交叉验证记录重新计算，确认结果与保存值一致。

ISE548 Risk and Crisis Management, Group 1.

Four checks, nothing is written:
    1. Load results/mlp_pytorch_checkpoint.pt, apply the saved scaling and Platt
       parameters to all 1,271 incidents, and compare every logit, raw score and
       calibrated probability with final_model_predictions.csv.
    2. Retrain the first cross-validation fold (repeat 1, fold 1) from scratch and
       compare its AUCs and Brier scores with the saved fold record.
    3. Recompute the cross-validation means from the 50 saved fold records.
    4. Question 2 (ignition): repeat checks 1 and 3 with mlp_ignition_checkpoint.pt,
       final_model_predictions_ignition.csv and mlp_ignition_results.json.
四项检查，不写任何文件：
    1. 加载最终模型权重，用保存的标准化参数和 Platt 参数重算全部 1,271 起事故，
       与 final_model_predictions.csv 逐条比较；
    2. 从头重新训练交叉验证的第 1 次重复第 1 折，与保存的该折结果比较；
    3. 用保存的 50 折记录重算交叉验证平均值；
    4. 问题二（点火）：对点火模型重复第 1 项和第 3 项。
所有差异都小于 1e-5 时输出 PASS。

Run from the 03_数据与代码 folder:
    python -B src/verify_mlp_inference.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from mlp_pytorch import (BASE_SEED, EXTERNAL_FIRE_DETAIL, IGNITION_N_CONT, N_FOLDS, GasRiskMLP,
                         build_ignition_inputs, clean_str, compute_roc_auc, fit_procedure, load_dataset,
                         predict_logits, sigmoid, stratified_folds)

# Largest difference allowed between recomputed and saved values.
# 重算值与保存值之间允许的最大差异（浮点运算在不同电脑上会有极小差别）。
TOLERANCE = 1e-5


def main():
    torch.set_num_threads(1)
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    meta = json.loads((results_dir / "mlp_pytorch_results.json").read_text(encoding="utf-8"))
    fm = meta["final_model"]
    X, y, names, rows = load_dataset()

    # 1. Final model: saved weights, scaling and Platt parameters.
    #    第 1 项：最终模型，使用保存的权重、标准化参数和 Platt 参数。
    Xs = X.copy()
    Xs[:, :2] = (Xs[:, :2] - np.array(fm["scaler_mean"])) / np.array(fm["scaler_std"])
    model = GasRiskMLP(input_dim=11, hidden1=16, hidden2=8, dropout_rate=0.1)
    model.load_state_dict(torch.load(results_dir / "mlp_pytorch_checkpoint.pt", weights_only=True, map_location="cpu"))
    model.eval()
    logits = predict_logits(model, Xs)
    raw = sigmoid(logits)
    cal = sigmoid(fm["platt_A"] * logits + fm["platt_B"])

    with open(results_dir / "final_model_predictions.csv", "r", encoding="utf-8") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == len(y), f"Expected {len(y)} rows in final_model_predictions.csv, got {len(saved)}"
    max_diff = 0.0
    for i, r in enumerate(saved):
        assert r["REPORT_NUMBER"] == rows[i]["REPORT_NUMBER"].strip(), f"Row {i}: report number mismatch"
        assert int(r["ACTUAL_LABEL"]) == int(y[i]), f"Row {i}: label mismatch"
        for key, value in (("LOGIT", logits[i]), ("RAW_SCORE", raw[i]), ("CALIBRATED_PROB", cal[i])):
            max_diff = max(max_diff, abs(float(r[key]) - value))
    print("1. Final model")
    print(f"   Largest difference in final_model_predictions.csv: {max_diff:.2e}")
    assert max_diff < TOLERANCE, "Final model predictions do not match"

    # 2. Retrain repeat 1, fold 1 and compare with the saved record.
    #    第 2 项：重新训练第 1 次重复第 1 折，与保存记录比较。
    saved_fold = meta["cross_validation"]["per_fold"][0]
    test_idx = stratified_folds(y, N_FOLDS, np.random.default_rng(BASE_SEED))[0]
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    proc = fit_procedure(X, y, train_idx, BASE_SEED + 1000)
    t_logits = predict_logits(proc["model"], proc["scale"](test_idx))
    t_raw = sigmoid(t_logits)
    t_cal = sigmoid(proc["platt_A"] * t_logits + proc["platt_B"])
    y_test = y[test_idx]
    causes = np.array([clean_str(r.get("CAUSE")) for r in rows])
    prior = float(y[train_idx].mean())
    rates = {c: float(y[train_idx][causes[train_idx] == c].mean()) for c in set(causes[train_idx])}
    cause_scores = np.array([rates.get(c, prior) for c in causes[test_idx]])

    recomputed = {
        "best_epoch": proc["best_epoch"],
        "mlp_auc": compute_roc_auc(y_test, t_raw),
        "cause_only_auc": compute_roc_auc(y_test, cause_scores),
        "brier_calibrated": float(np.mean((t_cal - y_test) ** 2)),
        "brier_prior": float(np.mean((prior - y_test) ** 2)),
    }
    print("2. Cross-validation fold (repeat 1, fold 1) retrained")
    worst = 0.0
    for key, value in recomputed.items():
        diff = abs(value - saved_fold[key])
        worst = max(worst, diff)
        print(f"   {key:<17} recomputed {value:.6f}   saved {saved_fold[key]:.6f}   diff {diff:.2e}")
    assert worst < TOLERANCE, "Retrained fold does not match the saved record"

    # 3. Cross-validation means from the saved fold records.
    #    第 3 项：用保存的 50 折记录重算平均值。
    per_fold = meta["cross_validation"]["per_fold"]
    summary = meta["cross_validation"]["summary"]
    print("3. Cross-validation means from the 50 fold records")
    worst = 0.0
    for key in ("mlp_auc", "cause_only_auc", "logistic_auc", "brier_calibrated", "brier_prior"):
        mean = float(np.mean([f[key] for f in per_fold]))
        diff = abs(mean - summary[key]["mean"])
        worst = max(worst, diff)
        print(f"   {key:<17} recomputed {mean:.6f}   saved {summary[key]['mean']:.6f}   diff {diff:.2e}")
    assert len(per_fold) == 50, f"Expected 50 fold records, got {len(per_fold)}"
    assert worst < TOLERANCE, "Cross-validation summary does not match the fold records"

    # 4. Ignition model: final predictions and cross-validation means.
    #    第 4 项：点火模型的最终预测和交叉验证平均值。
    ign = json.loads((results_dir / "mlp_ignition_results.json").read_text(encoding="utf-8"))
    fi = ign["final_model"]
    keep = [i for i, r in enumerate(rows) if EXTERNAL_FIRE_DETAIL not in clean_str(r.get("CAUSE_DETAILS"))]
    Xi = build_ignition_inputs(X, rows)[keep]
    mean_i, std_i = np.array(fi["scaler_mean"]), np.array(fi["scaler_std"])
    cont = Xi[:, :IGNITION_N_CONT]
    Xi[:, :IGNITION_N_CONT] = (np.where(np.isnan(cont), mean_i, cont) - mean_i) / std_i
    model_i = GasRiskMLP(input_dim=Xi.shape[1], hidden1=16, hidden2=8, dropout_rate=0.1)
    model_i.load_state_dict(torch.load(results_dir / "mlp_ignition_checkpoint.pt", weights_only=True, map_location="cpu"))
    model_i.eval()
    z = predict_logits(model_i, Xi)
    cal_i = sigmoid(fi["platt_A"] * z + fi["platt_B"])
    with open(results_dir / "final_model_predictions_ignition.csv", "r", encoding="utf-8") as f:
        saved_i = list(csv.DictReader(f))
    assert len(saved_i) == len(keep), f"Expected {len(keep)} rows, got {len(saved_i)}"
    max_diff = 0.0
    for j, r in enumerate(saved_i):
        assert r["REPORT_NUMBER"] == rows[keep[j]]["REPORT_NUMBER"].strip(), f"Ignition row {j}: report number mismatch"
        for key, value in (("LOGIT", z[j]), ("CALIBRATED_PROB", cal_i[j])):
            max_diff = max(max_diff, abs(float(r[key]) - value))
    per_fold_i = ign["cross_validation"]["per_fold"]
    worst = 0.0
    for key in ("mlp_auc", "cause_only_auc", "logistic_auc"):
        worst = max(worst, abs(float(np.mean([f[key] for f in per_fold_i])) - ign["cross_validation"]["summary"][key]["mean"]))
    print("4. Ignition model")
    print(f"   Largest difference in final_model_predictions_ignition.csv: {max_diff:.2e}")
    print(f"   Largest difference in cross-validation means: {worst:.2e}")
    assert max_diff < TOLERANCE, "Ignition model predictions do not match"
    assert worst < TOLERANCE, "Ignition cross-validation summary does not match the fold records"

    print(f"PASS: all differences below {TOLERANCE:g}")


if __name__ == "__main__":
    main()
