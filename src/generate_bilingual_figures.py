"""Draw the six report figures in English and Chinese (PNG at 300 dpi and PDF).
生成报告和 PPT 用的 6 张图，中英文各一套（PNG 300 dpi 和 PDF）。

ISE548 Risk and Crisis Management, Group 1.

All numbers come from the saved results through results_reader.py.
图中所有数字都通过 results_reader.py 从保存的结果读取，不在这里手写。

Figures (written to figures/en and figures/zh):
    eta_tree_model_m1         event tree with the six paths
    mlp_training_curves       loss and AUC of the final model, with the early-stopping epoch
    mlp_feature_importance    grouped permutation importance over 50 test folds
    mlp_calibration_curve     reliability diagram of all cross-validation predictions
    mlp_architecture_diagram  network layout
    mlp_ignition_factors      ignition model: input importance and adjusted ignition probability

Run from the 03_数据与代码 folder:
    python -B src/generate_bilingual_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIG_EN_DIR = BASE_DIR / "figures" / "en"
FIG_ZH_DIR = BASE_DIR / "figures" / "zh"
FIG_EN_DIR.mkdir(parents=True, exist_ok=True)
FIG_ZH_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "src"))
from results_reader import results  # noqa: E402

NAVY = "#1A365D"
BLUE = "#2B6CB0"
RED = "#9B2C2C"
GREY = "#4A5568"
TEXT = "#2D3748"


def set_matplotlib_font(lang: str = "en"):
    """Fonts: Arial for English, Microsoft YaHei for Chinese. 英文用 Arial，中文用微软雅黑。"""
    plt.rcParams.clear()
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "SimSun", "Arial"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def save(fig, name: str, lang: str):
    """Save one figure as PNG and PDF into figures/en or figures/zh. 保存为 PNG 和 PDF。"""
    out_dir = FIG_ZH_DIR if lang == "zh" else FIG_EN_DIR
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name} ({lang})")


def label_segment(ax, p0, p1, text, side, color, size=11.0, weight="bold", offset=0.17, frac=0.5):
    """Write text next to the middle of a branch line, above (side=+1) or below (side=-1) it.
    在事件树分支线的中点旁边写标签，放在线的上方或下方，避免文字压在线上。"""
    (x0, y0), (x1, y1) = p0, p1
    xm, ym = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)  # point along the line / 线上的位置
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    nx, ny = -dy / length, dx / length  # unit normal pointing to the left of the direction of travel
    if ny < 0:
        nx, ny = -nx, -ny
    # Anchor the text by the corner nearest the line, so the text grows away from the line.
    # 用离线最近的那个角对齐文字，让文字朝远离线的方向展开，不会压线。
    va = "bottom" if side > 0 else "top"
    if abs(dy) < 0.05 * abs(dx):
        ha = "center"
    else:
        ha = "right" if (side > 0) == (dy > 0) else "left"
    ax.text(xm + side * nx * offset, ym + side * ny * offset, text, ha=ha, va=va,
            fontsize=size, color=color, weight=weight)


def plot_eta_tree(lang: str = "en"):
    """Figure 1: the event tree with branch shares and the six outcome paths.
    图 1：事件树，标出各分支比例和六条结果路径的起数与伤亡。"""
    set_matplotlib_font(lang)
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 6.3)

    nodes = results.primary_eta["node_probabilities"]
    ep = results.endpoints
    n_total = results.total_incidents
    n_ign = nodes["p_ignition"]["count"]
    n_unign = n_total - n_ign
    n_exp = nodes["p_explosion_given_ignition"]["count"]
    n_unexp = n_ign - n_exp
    p_ign = n_ign / n_total
    p_exp = n_exp / n_ign
    sev = [nodes["p_severe_given_ie"]["point_estimate"], nodes["p_severe_given_ine"]["point_estimate"],
           nodes["p_severe_given_ni"]["point_estimate"]]

    title = ("Event tree of reportable gas distribution incidents" if not zh
             else "已上报配气管网事故的事件树")
    subtitle = (f"PHMSA records, 2010–2024, N = {n_total:,}" if not zh
                else f"PHMSA 上报记录，2010–2024 年，共 {n_total:,} 起")
    ax.text(0.15, 6.12, title, ha="left", va="center", fontsize=15, weight="bold", color=NAVY)
    ax.text(0.15, 5.80, subtitle, ha="left", va="center", fontsize=12, color=GREY)

    # Column headers: plain text over the branch they describe, with a thin rule below.
    # 列标题：直接写在所描述的分支上方，下面加一条细线（不用底色方框）。
    root_x, x1, x2, x3 = 0.55, 2.55, 4.35, 6.15
    box_x, box_w = 6.55, 4.15
    headers = [
        (f"Release\nN = {n_total:,}" if not zh else f"泄漏\n{n_total:,} 起", root_x - 0.4, root_x + 0.75),
        ("Ignition?" if not zh else "是否点火", root_x + 0.85, x1 - 0.05),
        ("Explosion?" if not zh else "是否爆炸", x1 + 0.05, x2 - 0.05),
        ("Death or\nhospitalization?" if not zh else "是否有人\n死亡或住院", x2 + 0.05, x3),
        ("Outcome path: incidents (share), casualties" if not zh else "结果路径：起数（占比）、伤亡", box_x, box_x + box_w),
    ]
    for label, xa, xb in headers:
        ax.text((xa + xb) / 2, 5.28, label, ha="center", va="center", fontsize=12, weight="bold", color=NAVY,
                linespacing=1.15)
        ax.plot([xa, xb], [4.98, 4.98], color=NAVY, lw=1.2)

    root_y, y_ign, y_unign = 2.75, 3.85, 1.35
    y_exp, y_unexp = 4.42, 3.2
    y_end = [4.62, 4.02, 3.38, 2.78, 1.72, 0.92]

    def line(p0, p1, lw=2.0):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREY, lw=lw, zorder=1)

    yes, no = ("Yes", "No") if not zh else ("是", "否")
    colon = ": " if not zh else "："
    unit = "" if not zh else " 起"
    line((root_x, root_y), (x1, y_ign))
    line((root_x, root_y), (x1, y_unign))
    label_segment(ax, (root_x, root_y), (x1, y_ign), f"{yes}{colon}{n_ign}{unit} ({p_ign:.1%})", +1, NAVY, size=12, offset=0.04)
    label_segment(ax, (root_x, root_y), (x1, y_unign), f"{no}{colon}{n_unign}{unit} ({1 - p_ign:.1%})", -1, NAVY, size=12, offset=0.04)

    line((x1, y_ign), (x2, y_exp))
    line((x1, y_ign), (x2, y_unexp))
    label_segment(ax, (x1, y_ign), (x2, y_exp), f"{yes}{colon}{n_exp}{unit} ({p_exp:.1%})", +1, NAVY, size=12, offset=0.04)
    label_segment(ax, (x1, y_ign), (x2, y_unexp), f"{no}{colon}{n_unexp}{unit} ({1 - p_exp:.1%})", -1, NAVY, size=12, offset=0.04, frac=0.62)

    for (start, ends, s) in [((x2, y_exp), (y_end[0], y_end[1]), sev[0]),
                             ((x2, y_unexp), (y_end[2], y_end[3]), sev[1]),
                             ((x1, y_unign), (y_end[4], y_end[5]), sev[2])]:
        line(start, (x3, ends[0]), 1.6)
        line(start, (x3, ends[1]), 1.6)
        label_segment(ax, start, (x3, ends[0]), f"{yes}{colon}{s:.1%}", +1, RED, size=11.5, offset=0.04)
        label_segment(ax, start, (x3, ends[1]), f"{no}{colon}{1 - s:.1%}", -1, GREY, size=11.5, weight="normal", offset=0.04)

    ax.scatter([root_x], [root_y], color=NAVY, s=150, zorder=5)
    ax.scatter([x1, x1, x2, x2], [y_ign, y_unign, y_exp, y_unexp], color=BLUE, s=110, zorder=5)
    ax.scatter([x3] * 6, y_end, color=BLUE, s=80, zorder=5)

    titles_en = [
        "Path 1: fire and explosion, severe harm",
        "Path 2: fire and explosion, no severe harm",
        "Path 3: fire, no explosion, severe harm",
        "Path 4: fire, no explosion, no severe harm",
        "Path 5: no ignition, severe harm",
        "Path 6: no ignition, no severe harm",
    ]
    titles_zh = [
        "路径 1：起火并爆炸，有死亡或住院",
        "路径 2：起火并爆炸，无死亡或住院",
        "路径 3：起火未爆炸，有死亡或住院",
        "路径 4：起火未爆炸，无死亡或住院",
        "路径 5：未点火，有死亡或住院",
        "路径 6：未点火，无死亡或住院",
    ]
    keys = ["Path_1_IE_Severe", "Path_2_IE_NoSevere", "Path_3_INE_Severe",
            "Path_4_INE_NoSevere", "Path_5_NI_Severe", "Path_6_NI_NoSevere"]
    for i, key in enumerate(keys):
        e = ep[key]
        severe = i in (0, 2, 4)
        if zh:
            detail = f"{e['incident_count']} 起，占 {e['unconditional_probability']:.1%}；死亡 {e['fatalities']} 人，住院 {e['injuries']} 人"
        else:
            detail = f"{e['incident_count']} ({e['unconditional_probability']:.1%}), {e['fatalities']} deaths, {e['injuries']} hospitalized"
        y = y_end[i]
        # A coloured bar on the left marks the paths with deaths or hospitalizations.
        # 左侧色条标出有死亡或住院的路径。
        ax.add_patch(patches.Rectangle((box_x, y - 0.25), 0.07, 0.5, facecolor=RED if severe else "#A0AEC0",
                                       edgecolor="none"))
        ax.text(box_x + 0.17, y + 0.1, (titles_zh if zh else titles_en)[i], ha="left", va="center", fontsize=11.5,
                weight="bold", color=RED if severe else TEXT)
        ax.text(box_x + 0.17, y - 0.13, detail, ha="left", va="center", fontsize=11, color=TEXT)

    note = (
        f"Shares are of all {n_total:,} incidents; the six paths add up to 100%. All {results.fatalities} deaths and "
        f"{results.injuries} hospitalizations are on Paths 1, 3 and 5."
        if not zh else
        f"占比都以全部 {n_total:,} 起为分母，六条路径合计 100%。{results.fatalities} 名死者和 {results.injuries} 名住院者都在路径 1、3、5。"
    )
    ax.text(0.15, 0.25, note, ha="left", va="center", fontsize=11, color=GREY)

    fig.tight_layout()
    save(fig, "eta_tree_model_m1", lang)


def plot_mlp_training_curves(lang: str = "en"):
    """Figure 3: loss and AUC of the final model; the dotted line marks the epoch kept by early stopping.
    图 3：最终模型的损失和 AUC；点线标出提前停止所保留的那一轮。"""
    set_matplotlib_font(lang)
    zh = lang == "zh"
    history = results.mlp_data["training_history"]
    fm = results.mlp_final
    epochs = [h["epoch"] for h in history]
    best = fm["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.plot(epochs, [h["fit_loss"] for h in history], color=BLUE, lw=2,
             label=f"Fitting part (N={fm['fit_size']})" if not zh else f"拟合部分（N={fm['fit_size']}）")
    ax1.plot(epochs, [h["val_loss"] for h in history], color="#C53030", lw=2, linestyle="--",
             label=f"Validation part (N={fm['validation_size']})" if not zh else f"验证部分（N={fm['validation_size']}）")
    ax1.axvline(best, color=GREY, linestyle=":", lw=1.6,
                label=f"Kept: epoch {best}" if not zh else f"保留第 {best} 轮")
    ax1.set_xlabel("Epoch" if not zh else "训练轮次", fontsize=12)
    ax1.set_ylabel("Weighted BCE loss" if not zh else "加权交叉熵损失", fontsize=12)
    ax1.set_title("Loss" if not zh else "损失", weight="bold", fontsize=13)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=11)

    ax2.plot(epochs, [h["fit_auc"] for h in history], color=BLUE, lw=2,
             label="Fitting part" if not zh else "拟合部分")
    ax2.plot(epochs, [h["val_auc"] for h in history], color="#C53030", lw=2, linestyle="--",
             label="Validation part" if not zh else "验证部分")
    ax2.axvline(best, color=GREY, linestyle=":", lw=1.6)
    ax2.axhline(results.cv_auc, color=NAVY, linestyle="-.", lw=1.4,
                label=f"Cross-validation mean ({results.cv_auc:.3f})" if not zh else f"交叉验证平均（{results.cv_auc:.3f}）")
    ax2.axhline(results.cause_base_auc, color="#718096", linestyle=":", lw=1.6,
                label=f"Cause-only baseline ({results.cause_base_auc:.3f})" if not zh else f"仅用起因的基准（{results.cause_base_auc:.3f}）")
    ax2.set_xlabel("Epoch" if not zh else "训练轮次", fontsize=12)
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("Ranking performance (ROC-AUC)" if not zh else "排序能力（ROC-AUC）", weight="bold", fontsize=13)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=10.5)

    fig.tight_layout()
    save(fig, "mlp_training_curves", lang)


def plot_mlp_feature_importance(lang: str = "en"):
    """Figure 4: permutation importance of the four input groups. 图 4：四组输入的置换重要性。"""
    set_matplotlib_font(lang)
    zh = lang == "zh"
    imp = results.mlp_pfi
    keys = ["all_causes_joint_group", "log_accident_psig", "log_release_mcf", "location_type_group"]
    labels = {
        "all_causes_joint_group": "Cause (7 indicators shuffled together)" if not zh else "起因（7 个指示变量一起打乱）",
        "log_accident_psig": "Pressure: ln(1 + psig)" if not zh else "运行压力：ln(1 + psig)",
        "log_release_mcf": "Release volume: ln(1 + MCF)" if not zh else "泄漏量：ln(1 + MCF)",
        "location_type_group": "Location (underground / aboveground)" if not zh else "敷设位置（埋地 / 地上）",
    }

    order = np.argsort([imp[k]["mean_auc_drop"] for k in keys])
    keys = [keys[i] for i in order]
    means = [imp[k]["mean_auc_drop"] for k in keys]
    stds = [imp[k]["sd_across_folds"] for k in keys]

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    ax.barh([labels[k] for k in keys], means, xerr=stds, color=BLUE, alpha=0.85, capsize=5, edgecolor=NAVY, height=0.55)
    ax.axvline(0, color="#718096", lw=1)
    ax.set_xlabel("Mean drop in test-fold ROC-AUC when the inputs are shuffled" if not zh else "打乱该组输入后测试折 ROC-AUC 的平均下降", fontsize=12)
    ax.set_title("Model M2: permutation importance (50 test folds; bars show ±1 SD)" if not zh
                 else "模型 M2：置换重要性（50 个测试折；误差线为 ±1 个标准差）", weight="bold", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.tick_params(axis="y", labelsize=11.5)
    for i, k in enumerate(keys):
        lo, hi = imp[k]["min_drop"], imp[k]["max_drop"]
        txt = (f"{imp[k]['mean_auc_drop']:.3f}  (folds {lo:.3f} to {hi:.3f})" if not zh
               else f"{imp[k]['mean_auc_drop']:.3f}（各折 {lo:.3f} 至 {hi:.3f}）")
        ax.text(means[i] + stds[i] + 0.003, i, txt, va="center", fontsize=10.5, color=TEXT)
    ax.set_xlim(0, max(m + s for m, s in zip(means, stds)) + 0.07)

    fig.tight_layout()
    save(fig, "mlp_feature_importance", lang)


def plot_mlp_calibration_curve(lang: str = "en"):
    """Figure 5: reliability diagram of all cross-validation predictions, before and after Platt scaling.
    图 5：全部交叉验证预测在校准前后的可靠性图。"""
    set_matplotlib_font(lang)
    zh = lang == "zh"
    with open(RESULTS_DIR / "calibration_bins.json", "r", encoding="utf-8") as f:
        cal = json.load(f)

    raw_bins = cal["raw_bins"]
    cal_bins = cal["calibrated_bins"]
    per_bin = raw_bins[0]["sample_count"]
    top = max(max(b["mean_prediction"], b["observed_fraction"]) for b in raw_bins + cal_bins) + 0.05

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#718096", lw=1.5, label="Perfect calibration" if not zh else "理想校准线 (y = x)")
    ax.plot([b["mean_prediction"] for b in raw_bins], [b["observed_fraction"] for b in raw_bins], marker="s", markersize=7,
            color="#C53030", lw=1.8, linestyle=":",
            label=f"Raw scores (Brier {results.raw_brier:.3f})" if not zh else f"未校准得分（Brier {results.raw_brier:.3f}）")
    ax.plot([b["mean_prediction"] for b in cal_bins], [b["observed_fraction"] for b in cal_bins], marker="o", markersize=8,
            color=BLUE, lw=2.2,
            label=f"After Platt scaling (Brier {results.cal_brier:.3f})" if not zh else f"Platt 校准后（Brier {results.cal_brier:.3f}）")

    ax.set_xlim(-0.02, top)
    ax.set_ylim(-0.02, top)
    ax.set_xlabel("Mean predicted probability in bin" if not zh else "分箱内平均预测概率", fontsize=12)
    ax.set_ylabel("Observed share of severe incidents" if not zh else "分箱内实际严重事故比例", fontsize=12)
    ax.set_title(f"Reliability diagram, cross-validation ({cal['n_bins']} equal-count bins)" if not zh
                 else f"交叉验证可靠性图（{cal['n_bins']} 个等量分箱）", weight="bold", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10.5)

    note = (
        f"Share of severe incidents: {cal['prevalence']:.1%}\n"
        f"Constant-prevalence Brier: {results.prior_brier:.3f}\n"
        f"Platt parameters fitted in each fold\n"
        f"{per_bin} predictions per bin ({cal['n_predictions']:,} in total)"
        if not zh else
        f"严重事故比例：{cal['prevalence']:.1%}\n"
        f"常数比例基准 Brier：{results.prior_brier:.3f}\n"
        f"Platt 参数在每一折单独拟合\n"
        f"每箱 {per_bin} 条预测（共 {cal['n_predictions']:,} 条）"
    )
    ax.text(0.02, top - 0.03, note, fontsize=10, color=TEXT, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EDF2F7", edgecolor="#CBD5E0", lw=0.8))

    fig.tight_layout()
    save(fig, "mlp_calibration_curve", lang)


def plot_mlp_architecture_diagram(lang: str = "en"):
    """Figure 2: structure of the two networks (same layers, different inputs).
    图 2：两个网络的结构（隐藏层相同，输入不同）。"""
    set_matplotlib_font(lang)
    zh = lang == "zh"
    n1, n2 = results.mlp_meta["input_dimension"], results.ign_inputs
    p1, p2 = results.mlp_meta["total_trainable_parameters"], results.ign_params
    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.5)

    def box(x, y, w, h, head, body, edge, head_color=NAVY):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                            facecolor="white", edgecolor=edge, linewidth=1.8))
        ax.text(x + w / 2, y + h - 0.24, head, ha="center", va="center", fontsize=12.5, weight="bold",
                color=head_color)
        ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center", fontsize=11.5, color=TEXT,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=GREY, mutation_scale=14))

    teal = "#0D9488"
    q1_body = (f"{n1} inputs: release volume,\npressure, location (2),\ncause (7)" if not zh else
               f"{n1} 个输入：泄漏量、压力、\n位置（2 个）、原因（7 个）")
    q2_body = (f"{n2} inputs known before ignition:\nlocation, system part, material,\npressure, cause and others" if not zh else
               f"{n2} 个点火前就已知的输入：\n位置、管段、管材、压力、\n原因等")
    box(0.1, 2.6, 2.9, 1.6, "Question 1: severe harm" if not zh else "问题一：严重伤亡", q1_body, teal)
    box(0.1, 0.75, 2.9, 1.6, "Question 2: ignition" if not zh else "问题二：是否点火", q2_body, teal)

    layer_y, layer_h = 1.35, 2.2
    layers = [
        (3.45, "Hidden layer 1" if not zh else "隐藏层 1",
         f"Linear {n1}/{n2} → 16\nBatchNorm\nReLU\nDropout 0.1" if not zh else f"全连接 {n1}/{n2} → 16\n批归一化\nReLU\nDropout 0.1"),
        (5.15, "Hidden layer 2" if not zh else "隐藏层 2",
         "Linear 16 → 8\nBatchNorm\nReLU\nDropout 0.1" if not zh else "全连接 16 → 8\n批归一化\nReLU\nDropout 0.1"),
        (6.85, "Output" if not zh else "输出层", "Linear 8 → 1\nlogit z" if not zh else "全连接 8 → 1\nlogit z"),
    ]
    for x, head, body in layers:
        box(x, layer_y, 1.5, layer_h, head, body, BLUE)
    box(8.55, layer_y, 1.7, layer_h, "Calibration" if not zh else "校准",
        "Platt scaling\n$p = \\sigma(Az + B)$" if not zh else "Platt 校准\n$p = \\sigma(Az + B)$", "#B7791F")

    mid = layer_y + layer_h / 2
    arrow(3.02, 3.4, 3.43, mid + 0.25)
    arrow(3.02, 1.55, 3.43, mid - 0.25)
    for x in (4.97, 6.67, 8.37):
        arrow(x, mid, x + 0.16, mid)

    note = (f"Same layers for both questions. Parameters: {p1} for question 1, {p2} for question 2.\n"
            rf"Training: loss weighted by $w = N_{{\mathrm{{neg}}}}/N_{{\mathrm{{pos}}}}$, Adam, early stopping on the validation loss."
            if not zh else
            f"两个问题用相同的隐藏层。参数个数：问题一 {p1} 个，问题二 {p2} 个。\n"
            rf"训练：损失按 $w = N_{{\mathrm{{neg}}}}/N_{{\mathrm{{pos}}}}$ 加权，Adam 优化，按验证损失提前停止。")
    ax.text(0.1, 0.32, note, ha="left", va="center", fontsize=11, color=GREY, linespacing=1.5)

    # Fill the whole canvas so the boxes keep their size. 让坐标区铺满整张图，方框保持设定的大小。
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(fig, "mlp_architecture_diagram", lang)


def plot_mlp_ignition_factors(lang: str = "en"):
    """Figure 6: ignition model. Left, which input groups matter; right, the predicted ignition
    probability when one factor is changed and every other input is kept as recorded.
    图 6：点火模型。左图：哪些输入组最重要；右图：只改变一个因素、其余输入保持原样时的预测点火概率。"""
    set_matplotlib_font(lang)
    zh = lang == "zh"
    imp = results.ign_pfi
    adj = results.ign_adjusted
    obs = results.ign_observed

    group_labels = {
        "material": ("Pipe material", "管材"),
        "pressure": ("Pressure", "运行压力"),
        "location_type": ("Location type (private,\npublic, right-of-way)", "位置类型（私人、\n公共、通行带）"),
        "system_part": ("System part (main, service,\nriser, meter set, station)", "管段（主管、入户支管、\n立管、表组、调压站）"),
        "area_type": ("Underground / aboveground", "埋地 / 地上"),
        "cause": ("Cause (7 indicators)", "原因（7 个指示变量）"),
        "area_subtype": ("Area detail (soil, pavement,\nbuilding, piping, excavation)", "区域细分（土下、路面下、\n建筑、地上管件、开挖暴露）"),
        "release_type": ("Release type", "泄漏形式"),
    }
    keys = sorted(group_labels, key=lambda k: imp[k]["mean_auc_drop"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax1.barh([group_labels[k][1 if zh else 0] for k in keys], [imp[k]["mean_auc_drop"] for k in keys],
             xerr=[imp[k]["sd_across_folds"] for k in keys], color=BLUE, alpha=0.85, capsize=4, edgecolor=NAVY, height=0.6)
    ax1.axvline(0, color="#718096", lw=1)
    ax1.set_xlabel("Mean drop in test-fold AUC when shuffled" if not zh else "打乱后测试折 AUC 的平均下降", fontsize=12)
    ax1.set_title("Which inputs matter (50 folds, ±1 SD)" if not zh else "哪些输入重要（50 折，±1 个标准差）",
                  weight="bold", fontsize=13)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax1.tick_params(axis="y", labelsize=11.5)
    ax1.tick_params(axis="x", labelsize=11)

    # Rows for the right panel: (group, key in results, English label, Chinese label).
    # 右图的行：（分组、结果中的键、英文标签、中文标签）。
    psig_keys = list(adj["pressure"].keys())
    rows = [
        ("system_part", "service", "Service line", "入户支管"),
        ("system_part", "service_riser", "Service riser", "立管"),
        ("system_part", "meter_set", "Meter / regulator set", "表组 / 调压器"),
        ("system_part", "main", "Main", "主管"),
        ("system_part", "district_station", "District regulator station", "区域调压站"),
        ("location_type", "private_property", "Private property", "私人地产"),
        ("location_type", "public_property", "Public property", "公共地产"),
        ("location_type", "utility_row", "Utility right-of-way", "公用通行带"),
        ("material", "plastic", "Plastic pipe", "塑料管"),
        ("material", "steel", "Steel pipe", "钢管"),
        ("material", "iron", "Cast / ductile iron", "铸铁管"),
    ] + [("pressure", k, f"Pressure {k.split('_')[1]} psig", f"压力 {k.split('_')[1]} psig") for k in psig_keys]
    y_pos = np.arange(len(rows))[::-1]
    colors = {"system_part": "#2B6CB0", "location_type": "#2F855A", "material": "#B7791F", "pressure": "#9B2C2C"}
    for yp, (grp, key, en, cn) in zip(y_pos, rows):
        ax2.barh(yp, adj[grp][key], color=colors[grp], alpha=0.85, height=0.62)
        # Values sit in their own column right of the plot, clear of bars and markers.
        # 数值单独排成一列放在图的右侧，不和柱子、菱形标记重叠。
        ax2.text(1.03, yp, f"{adj[grp][key]:.2f}", va="center", ha="left", fontsize=11, color=TEXT,
                 clip_on=False)
        if grp in obs and key in obs[grp]:
            ax2.plot(obs[grp][key]["ignition_rate"], yp, marker="D", color="black", markersize=6.5, linestyle="none")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([cn if zh else en for _, _, en, cn in rows], fontsize=11.5)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.axvline(results.ign_sample["ignition_rate"], color="#718096", linestyle="--", lw=1.2)
    ax2.set_xlabel("Ignition probability" if not zh else "点火概率", fontsize=12)
    ax2.set_title("Predicted ignition probability" if not zh else "其他输入不变时的预测点火概率",
                  weight="bold", fontsize=13)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax2.plot([], [], marker="D", color="black", linestyle="none",
             label="Observed rate in the data" if not zh else "数据中的实际点火比例")
    ax2.plot([], [], color="#718096", linestyle="--",
             label=f"Overall rate ({results.ign_sample['ignition_rate']:.1%})" if not zh
             else f"总体点火比例（{results.ign_sample['ignition_rate']:.1%}）")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=11, frameon=False)

    fig.tight_layout()
    save(fig, "mlp_ignition_factors", lang)


def main():
    """Draw every figure in both languages. 中英文各画一遍全部图。"""
    for lang in ["en", "zh"]:
        print(f"Figures ({lang})")
        plot_eta_tree(lang)
        plot_mlp_training_curves(lang)
        plot_mlp_feature_importance(lang)
        plot_mlp_calibration_curve(lang)
        plot_mlp_architecture_diagram(lang)
        plot_mlp_ignition_factors(lang)


if __name__ == "__main__":
    main()
