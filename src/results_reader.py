"""Read the saved analysis results once and give the common numbers short names.
统一读取 results/ 中保存的分析结果，并给常用数字起简短的名字。

ISE548 Risk and Crisis Management, Group 1.

`results` exposes the event tree and neural network numbers from results/ (read
only, nothing is recomputed). The figure script takes its numbers from here.
`results`：事件树和神经网络的全部数字（只读，不重新计算）。画图脚本从这里取数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"



class SharedResults:
    """Loads the result files and gives the commonly used numbers short names.
    加载结果文件，并给常用的数字起简短的名字。"""

    def __init__(self, results_dir: Optional[Path] = None):
        r_dir = results_dir or RESULTS_DIR
        self.eta_path = r_dir / "eta_audit_results.json"
        self.mlp_path = r_dir / "mlp_pytorch_results.json"
        self.bins_path = r_dir / "calibration_bins.json"
        self.ign_path = r_dir / "mlp_ignition_results.json"

        if not self.eta_path.exists():
            raise FileNotFoundError(f"{self.eta_path} not found. Run src/eta_pipeline.py first.")
        if not self.mlp_path.exists():
            raise FileNotFoundError(f"{self.mlp_path} not found. Run src/mlp_pytorch.py first.")

        with open(self.eta_path, "r", encoding="utf-8") as f:
            self.eta_data = json.load(f)
        with open(self.mlp_path, "r", encoding="utf-8") as f:
            self.mlp_data = json.load(f)
        with open(self.ign_path, "r", encoding="utf-8") as f:
            self.ign_data = json.load(f)

        # Event tree (Model M1). 事件树（模型 M1）。
        self.primary_eta = self.eta_data["primary_window_2010_2024"]
        self.baseline_eta = self.eta_data["baseline_window_2015_2024"]
        self.sensitivity = self.eta_data["sensitivity_analysis"]

        totals = self.primary_eta["casualty_totals"]
        self.total_incidents = self.primary_eta["sample_flow"]["final_clean_sample_n"]
        self.fatalities = totals["fatalities"]
        self.injuries = totals["injuries"]  # inpatient hospitalizations
        self.total_casualties = totals["total_casualties"]
        self.severe_incidents = totals["severe_incidents"]
        self.severe_rate = self.severe_incidents / self.total_incidents

        self.endpoints = self.primary_eta["endpoints"]
        self.causes = self.primary_eta["cause_stratification"]
        self.outside_force_subcauses = self.primary_eta["outside_force_subcauses"]

        # Neural network (Model M2): cross-validation summary and final model.
        # 神经网络（模型 M2）：交叉验证汇总和最终模型。
        self.mlp_meta = self.mlp_data["model_metadata"]
        self.mlp_loss = self.mlp_data["loss_formulation"]
        self.mlp_cv = self.mlp_data["cross_validation"]
        self.mlp_cv_protocol = self.mlp_cv["protocol"]
        self.mlp_cv_summary = s = self.mlp_cv["summary"]
        self.mlp_final = fm = self.mlp_data["final_model"]
        self.mlp_pfi = self.mlp_data["permutation_feature_importance"]

        # Means over the 50 test folds (10 folds x 5 repeats). 50 个测试折（10 折 × 5 次）的平均值。
        self.cv_auc = s["mlp_auc"]["mean"]
        self.cv_auc_sd = s["mlp_auc"]["sd"]
        self.cv_auc_ci = s["mlp_auc"]["ci95"]
        self.cv_auc_by_repeat = s["mlp_auc_by_repeat"]
        self.cause_base_auc = s["cause_only_auc"]["mean"]
        self.logistic_auc = s["logistic_auc"]["mean"]
        self.fit_auc = s["fit_auc"]["mean"]
        self.auc_gain = s["auc_gain_over_cause_only"]["mean"]
        self.auc_gain_ci = s["auc_gain_over_cause_only"]["ci95"]
        self.auc_gain_folds_positive = s["auc_gain_over_cause_only"]["folds_positive"]
        self.auc_gain_lr = s["auc_gain_over_logistic"]["mean"]
        self.auc_gain_lr_ci = s["auc_gain_over_logistic"]["ci95"]
        self.auc_gain_lr_folds_positive = s["auc_gain_over_logistic"]["folds_positive"]

        self.raw_brier = s["brier_raw"]["mean"]
        self.cal_brier = s["brier_calibrated"]["mean"]
        self.prior_brier = s["brier_prior"]["mean"]
        self.logistic_brier = s["brier_logistic"]["mean"]
        self.delta_brier = s["brier_calibrated_minus_prior"]["mean"]
        self.delta_brier_ci = s["brier_calibrated_minus_prior"]["ci95"]
        self.raw_mean_score = s["mean_raw_score"]["mean"]
        self.cal_mean_score = s["mean_calibrated_prob"]["mean"]
        self.prevalence = s["test_prevalence"]["mean"]
        self.best_epoch_mean = s["best_epoch"]["mean"]
        self.best_epoch_range = (int(s["best_epoch"]["min"]), int(s["best_epoch"]["max"]))
        self.cv_train_size = self.mlp_cv_protocol["mean_train_size"]
        self.cv_test_size = self.mlp_cv_protocol["mean_test_size"]

        # Final model trained on all 1,271 incidents. 用全部 1,271 起事故训练的最终模型。
        self.platt_A = fm["platt_A"]
        self.platt_B = fm["platt_B"]
        self.pos_weight = fm["pos_weight"]
        self.final_best_epoch = fm["best_epoch"]
        self.final_epochs_run = fm["epochs_run"]

        # Question 2: ignition from pre-ignition conditions (fire-first incidents excluded).
        # 问题二：用点火前的条件判断是否点火（不含先着火的 72 起）。
        g = self.ign_data
        gs = g["cross_validation"]["summary"]
        self.ign_sample = g["sample"]
        self.ign_n = g["sample"]["analysis_sample"]
        self.ign_inputs = g["model_metadata"]["input_dimension"]
        self.ign_params = g["model_metadata"]["total_trainable_parameters"]
        self.ign_auc = gs["mlp_auc"]["mean"]
        self.ign_auc_sd = gs["mlp_auc"]["sd"]
        self.ign_auc_ci = gs["mlp_auc"]["ci95"]
        self.ign_cause_auc = gs["cause_only_auc"]["mean"]
        self.ign_logistic_auc = gs["logistic_auc"]["mean"]
        self.ign_gain = gs["auc_gain_over_cause_only"]["mean"]
        self.ign_gain_ci = gs["auc_gain_over_cause_only"]["ci95"]
        self.ign_gain_folds_positive = gs["auc_gain_over_cause_only"]["folds_positive"]
        self.ign_gain_lr = gs["auc_gain_over_logistic"]["mean"]
        self.ign_gain_lr_ci = gs["auc_gain_over_logistic"]["ci95"]
        self.ign_cal_brier = gs["brier_calibrated"]["mean"]
        self.ign_prior_brier = gs["brier_prior"]["mean"]
        self.ign_fit_auc = gs["fit_auc"]["mean"]
        self.ign_sensitivity = g["sensitivity_all_1271"]
        self.ign_pfi = g["permutation_feature_importance"]
        self.ign_adjusted = g["adjusted_ignition_probability"]
        self.ign_observed = g["observed_ignition_rate"]

    def get_cause_table_rows(self) -> List[Dict[str, Any]]:
        """One row per PHMSA cause category, in order of incident count.
        每类事故原因一行，按事故数从多到少排列（报告表 10、PPT 第 7 页使用）。"""
        order = [
            "EXCAVATION DAMAGE",
            "OTHER OUTSIDE FORCE DAMAGE",
            "INCORRECT OPERATION",
            "PIPE, WELD, OR JOINT FAILURE",
            "NATURAL FORCE DAMAGE",
            "OTHER INCIDENT CAUSE",
            "EQUIPMENT FAILURE",
            "CORROSION FAILURE",
        ]
        rows = []
        for c in order:
            d = self.causes[c]
            n = d["incident_count"]
            rows.append({
                "cause": c,
                "incident_count": n,
                "incident_share": n / self.total_incidents,
                "p_ignition": d["p_ignition"],
                "p_explosion_given_ignition": d["p_explosion_given_ignition"],
                "severe_incident_count": d["severe_incident_count"],
                "p_severe_harm": d["severe_incident_count"] / n if n > 0 else 0.0,
                "fatalities": d["fatalities"],
                "injuries": d["injuries"],
                "total_casualties": d["total_casualties"],
                "severe_incident_share_of_all_severe": d["severe_incident_count"] / self.severe_incidents,
                "casualty_share_of_all_casualties": d["total_casualties"] / self.total_casualties,
            })
        return rows


results = SharedResults()
