"""Model M1: event tree for PHMSA gas distribution incidents.
事件树分析（模型 M1）：PHMSA 天然气配气管网事故。

ISE548 Risk and Crisis Management, Group 1.

Steps / 步骤:
    Primary sample: 2010-2024, natural gas, release volume > 0, no
    ignite/explode conflict. Each incident goes on one of six paths; casualties
    are counted by path and by cause. Also run: a 2015-2024 window and five
    other sample definitions.
    读取 data/ 中的 PHMSA 事故文件，筛出主样本（2010–2024 年、天然气、
    泄漏量大于 0、点火/爆炸字段不矛盾），把每起事故归入六条事件树路径之一，
    再按路径和按事故原因汇总伤亡。另外计算 2015–2024 年窗口和五种样本口径，
    用来检验结论是否稳健。

Outputs (results/) / 输出:
    eta_audit_results.json  - all counts, probabilities and confidence intervals
                              全部计数、概率和置信区间
    eta_summary_tables.md   - the same numbers as readable tables
                              同样的数字，整理成可读的表格

Run from the folder above src/ / 在 src/ 的上一级目录（03_数据与代码）下运行:
    python -B src/eta_pipeline.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Tuple

# File locations, relative to the folder above src/.
# 文件位置，均相对于 03_数据与代码 目录。
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATA = BASE_DIR / "data" / "incident_gas_distribution_jan2010_present.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sub-cause text marking incidents where a nearby fire damaged the pipe first.
# 表示“附近火灾或爆炸先损坏管道”的子原因文字，这类事故是先起火、后泄漏。
EXTERNAL_FIRE_DETAIL = "NEARBY INDUSTRIAL, MAN-MADE, OR OTHER FIRE/EXPLOSION AS PRIMARY CAUSE"


def parse_float(val: Optional[str]) -> Optional[float]:
    """Text field -> number. Empty or invalid text gives None, never 0.
    把文本字段转成数字；空值或无法解析时返回 None，而不是 0。"""
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
    """Trim spaces and upper-case, so "Yes " and "YES" compare equal.
    去掉首尾空格并转成大写，避免 "Yes " 和 "YES" 被当成不同的值。"""
    return val.strip().upper() if val else ""


def wilson_score_interval(k: int, n: int, z: float = 1.95996) -> Optional[Tuple[float, float]]:
    """95% Wilson score interval for k successes out of n (Wilson, 1927).
    比例 k/n 的 95% Wilson 置信区间（Wilson, 1927）。

    Stays inside [0, 1] and works for small counts. None when n == 0.
    和常见的正态近似不同，它不会超出 [0, 1]，样本小时也可用。
    当 n 为 0 时返回 None，避免把空分支显示成 [0, 0]。
    """
    if n <= 0:
        return None
    p_hat = k / n
    denom = 1.0 + (z**2) / n
    centre = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * (n**2)))) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (round(lower, 4), round(upper, 4))


def has_severe_harm(r: dict) -> bool:
    """Severe harm = at least one death or one inpatient hospitalization.
    “严重伤亡”的定义：至少一人死亡，或至少一人需要住院治疗。"""
    fat = parse_float(r.get("FATAL")) or 0.0
    inj = parse_float(r.get("INJURE")) or 0.0
    return fat > 0.0 or inj > 0.0


def load_raw_rows() -> list:
    """Read the tab-separated PHMSA file; one dict per incident report.
    读取以制表符分隔的 PHMSA 文件，每份事故报告是一个字典。"""
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"PHMSA data file not found: {SOURCE_DATA}")
    with SOURCE_DATA.open("r", encoding="latin-1", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def is_skip_conflict(r: dict) -> bool:
    # The form skips the explosion question when there was no ignition, so
    # IGNITE_IND = NO together with EXPLODE_IND = YES is a recording conflict.
    # 表格规定：未点火时不填“是否爆炸”。所以“未点火却爆炸”是填报矛盾，需要剔除。
    return clean_str(r["IGNITE_IND"]) == "NO" and clean_str(r["EXPLODE_IND"]) == "YES"


def casualty_summary(rows: list) -> dict:
    """Incidents, severe incidents, deaths and hospitalizations for a group of records.
    统计一组记录的事故数、严重事故数、死亡人数和住院人数。"""
    return {
        "incident_count": len(rows),
        "severe_incident_count": sum(has_severe_harm(r) for r in rows),
        "fatalities": sum(int(parse_float(r["FATAL"]) or 0.0) for r in rows),
        "injuries": sum(int(parse_float(r["INJURE"]) or 0.0) for r in rows),
    }


def run_event_tree(start_year: int = 2010, end_year: int = 2024) -> dict:
    """Select the sample for one time window and compute the whole event tree.
    对一个时间窗口筛选样本，并计算整棵事件树（路径、分支概率、按原因统计）。"""
    raw_rows = load_raw_rows()

    # 1. Calendar window.
    #    第 1 步：只保留指定年份的事故（2025–2026 年数据不完整，不使用）。
    period_rows = [r for r in raw_rows if start_year <= int(r["IYEAR"]) <= end_year]

    # 2. Natural gas only (Form F 7100.1 Item A6). Records with a blank
    #    commodity field are excluded because the gas type cannot be confirmed.
    #    第 2 步：只保留天然气（表格 A6 项）。气体类型为空的记录无法确认是天然气，
    #    一并排除，但单独统计，供报告说明和敏感性分析使用。
    gas_rows = [r for r in period_rows if clean_str(r["COMMODITY_RELEASED_TYPE"]) == "NATURAL GAS"]
    non_gas_rows = [r for r in period_rows if clean_str(r["COMMODITY_RELEASED_TYPE"]) != "NATURAL GAS"]
    blank_commodity_rows = [r for r in non_gas_rows if clean_str(r["COMMODITY_RELEASED_TYPE"]) == ""]

    # 3. Release volume (Item A7): missing values are not treated as zero.
    #    第 3 步：泄漏量（A7 项）必须大于 0。没填的记录是“未知”，不能当成 0，
    #    所以和泄漏量为 0 的记录分开计数。
    missing_release_rows = [r for r in gas_rows if parse_float(r["UNINTENTIONAL_RELEASE"]) is None]
    zero_release_rows = [r for r in gas_rows if parse_float(r["UNINTENTIONAL_RELEASE"]) == 0.0]
    positive_release_rows = [r for r in gas_rows if (parse_float(r["UNINTENTIONAL_RELEASE"]) or 0.0) > 0.0]

    # 4. Records that say "no ignition" but "explosion".
    #    第 4 步：找出“未点火却爆炸”的矛盾记录。
    anomaly_rows = [r for r in positive_release_rows if is_skip_conflict(r)]
    anomaly_ids = {r["REPORT_NUMBER"].strip() for r in anomaly_rows}

    # 5. What remains is the primary sample (1,271 incidents for 2010-2024).
    #    第 5 步：剩下的就是主样本（2010–2024 年共 1,271 起）。
    primary_rows = [r for r in positive_release_rows if r["REPORT_NUMBER"].strip() not in anomaly_ids]

    # Six mutually exclusive end states of the tree: ignite? -> explode? -> severe harm?
    # 事件树的六个结局，互不重叠：是否点火 → 是否爆炸 → 是否有严重伤亡。
    # IE = ignited and exploded, INE = ignited but not exploded, NI = not ignited.
    # 其中 IE = 点火并爆炸，INE = 点火未爆炸，NI = 未点火。
    paths = {
        "Path_1_IE_Severe": [],
        "Path_2_IE_NoSevere": [],
        "Path_3_INE_Severe": [],
        "Path_4_INE_NoSevere": [],
        "Path_5_NI_Severe": [],
        "Path_6_NI_NoSevere": [],
    }

    # Count how the explosion field was filled for unignited incidents (NO vs blank).
    # 统计未点火事故的“是否爆炸”字段是填了 NO 还是空着。
    unignited_explosion_blank_cnt = 0
    unignited_explosion_no_cnt = 0

    for r in primary_rows:
        ignite = clean_str(r["IGNITE_IND"]) == "YES"
        explode = clean_str(r["EXPLODE_IND"]) == "YES"
        sev = has_severe_harm(r)

        if ignite and explode:
            if sev:
                paths["Path_1_IE_Severe"].append(r)
            else:
                paths["Path_2_IE_NoSevere"].append(r)
        elif ignite and not explode:
            if sev:
                paths["Path_3_INE_Severe"].append(r)
            else:
                paths["Path_4_INE_NoSevere"].append(r)
        else:
            # Unignited: the explosion field is either NO or left blank by the form's skip logic.
            # 未点火：表格会跳过“是否爆炸”，所以该字段是 NO 或空白，两者都按未爆炸处理。
            exp_val = clean_str(r["EXPLODE_IND"])
            if exp_val == "":
                unignited_explosion_blank_cnt += 1
            else:
                unignited_explosion_no_cnt += 1

            if sev:
                paths["Path_5_NI_Severe"].append(r)
            else:
                paths["Path_6_NI_NoSevere"].append(r)

    # Node counts: how many incidents reach each branch point of the tree.
    # 各节点的事故数：有多少起事故走到事件树的每个分叉点。
    total_n = len(primary_rows)
    ignited_n = len(paths["Path_1_IE_Severe"]) + len(paths["Path_2_IE_NoSevere"]) + len(paths["Path_3_INE_Severe"]) + len(paths["Path_4_INE_NoSevere"])
    unignited_n = total_n - ignited_n
    exploded_n = len(paths["Path_1_IE_Severe"]) + len(paths["Path_2_IE_NoSevere"])
    unexploded_ignited_n = ignited_n - exploded_n

    severe_incidents_total = sum(len(paths[k]) for k in ["Path_1_IE_Severe", "Path_3_INE_Severe", "Path_5_NI_Severe"])
    total_fatalities = sum(int(parse_float(r["FATAL"]) or 0.0) for r in primary_rows)
    total_injuries = sum(int(parse_float(r["INJURE"]) or 0.0) for r in primary_rows)

    # Conditional branch probabilities = incidents taking the branch / incidents at the node.
    # 条件分支概率 = 走这条分支的事故数 / 到达这个节点的事故数。
    p_ignite = ignited_n / total_n if total_n > 0 else 0.0
    p_explode_given_ignite = exploded_n / ignited_n if ignited_n > 0 else 0.0
    p_severe_given_ie = len(paths["Path_1_IE_Severe"]) / exploded_n if exploded_n > 0 else 0.0
    p_severe_given_ine = len(paths["Path_3_INE_Severe"]) / unexploded_ignited_n if unexploded_ignited_n > 0 else 0.0
    p_severe_given_ni = len(paths["Path_5_NI_Severe"]) / unignited_n if unignited_n > 0 else 0.0

    # Path probability = product of the branch probabilities along the path (chain rule).
    # The six products must add up to 1 and equal the observed path shares.
    # 路径概率 = 沿途各分支概率相乘（链式法则）。六条路径相加应等于 1，
    # 且应等于实际观察到的路径占比，用来自检。
    chain_p1 = p_ignite * p_explode_given_ignite * p_severe_given_ie
    chain_p2 = p_ignite * p_explode_given_ignite * (1.0 - p_severe_given_ie)
    chain_p3 = p_ignite * (1.0 - p_explode_given_ignite) * p_severe_given_ine
    chain_p4 = p_ignite * (1.0 - p_explode_given_ignite) * (1.0 - p_severe_given_ine)
    chain_p5 = (1.0 - p_ignite) * p_severe_given_ni
    chain_p6 = (1.0 - p_ignite) * (1.0 - p_severe_given_ni)

    # Readable names used in tables and figures.
    # 表格和图中显示的路径名称。
    endpoint_names = {
        "Path_1_IE_Severe": "Ignition & Explosion with Severe Harm (Fatal / Hospitalized)",
        "Path_2_IE_NoSevere": "Ignition & Explosion without Severe Harm",
        "Path_3_INE_Severe": "Ignition without Explosion with Severe Harm (Fatal / Hospitalized)",
        "Path_4_INE_NoSevere": "Ignition without Explosion without Severe Harm",
        "Path_5_NI_Severe": "Unignited Release with Severe Harm (Fatal / Hospitalized)",
        "Path_6_NI_NoSevere": "Unignited Release without Severe Harm",
    }

    # Share, confidence interval and casualties for each path.
    # 每条路径的占比、置信区间和伤亡人数。
    endpoints_data = {}
    for k, v in paths.items():
        cnt = len(v)
        fat = sum(int(parse_float(r["FATAL"]) or 0.0) for r in v)
        inj = sum(int(parse_float(r["INJURE"]) or 0.0) for r in v)
        endpoints_data[k] = {
            "title": endpoint_names[k],
            "incident_count": cnt,
            "unconditional_probability": round(cnt / total_n, 4) if total_n > 0 else 0.0,
            "wilson_ci_95": wilson_score_interval(cnt, total_n),
            "fatalities": fat,
            "injuries": inj,
            "total_casualties": fat + inj,
        }

    # Repeat the tree for each of the eight PHMSA cause categories.
    # 对 PHMSA 的八类事故原因，分别重复上面的统计（报告第 5 节）。
    by_cause = defaultdict(list)
    for r in primary_rows:
        by_cause[clean_str(r["CAUSE"]) or "OTHER INCIDENT CAUSE"].append(r)

    cause_metrics = {}
    for c_name, r_list in by_cause.items():
        c_n = len(r_list)
        c_ign = sum(clean_str(r["IGNITE_IND"]) == "YES" for r in r_list)
        c_exp = sum(clean_str(r["IGNITE_IND"]) == "YES" and clean_str(r["EXPLODE_IND"]) == "YES" for r in r_list)
        c_sev = sum(has_severe_harm(r) for r in r_list)
        c_fatal = sum(int(parse_float(r["FATAL"]) or 0.0) for r in r_list)
        c_inj = sum(int(parse_float(r["INJURE"]) or 0.0) for r in r_list)
        c_ext_fire = sum(1 for r in r_list if EXTERNAL_FIRE_DETAIL in clean_str(r.get("CAUSE_DETAILS", "")))

        cause_metrics[c_name] = {
            "incident_count": c_n,
            "incident_share": round(c_n / total_n, 4) if total_n > 0 else 0.0,
            "p_ignition": round(c_ign / c_n, 4) if c_n > 0 else 0.0,
            "p_explosion_given_ignition": round(c_exp / c_ign, 4) if c_ign > 0 else 0.0,
            "p_severe_harm": round(c_sev / c_n, 4) if c_n > 0 else 0.0,
            "severe_incident_count": c_sev,
            "severe_incident_share_of_all_severe": round(c_sev / severe_incidents_total, 4) if severe_incidents_total > 0 else 0.0,
            "fatalities": c_fatal,
            "injuries": c_inj,
            "total_casualties": c_fatal + c_inj,
            "external_fire_primary_cause_count": c_ext_fire,
            "wilson_ci_p_severe": wilson_score_interval(c_sev, c_n),
        }

    # Sub-causes inside Other Outside Force Damage (vehicle impact, external fire, ...).
    # 把“其他外力损伤”按子原因拆开（车辆撞击、附近火灾等），作为防撞建议的依据。
    osf_by_detail = defaultdict(list)
    for r in by_cause.get("OTHER OUTSIDE FORCE DAMAGE", []):
        osf_by_detail[clean_str(r.get("CAUSE_DETAILS", "")) or "NOT RECORDED"].append(r)
    outside_force_subcauses = sorted(
        ({"cause_detail": d, **casualty_summary(rows)} for d, rows in osf_by_detail.items()),
        key=lambda x: x["incident_count"],
        reverse=True,
    )

    # Which commodities were excluded in step 2, and how many of each.
    # 第 2 步排除了哪些气体类型、各多少条。
    commodity_counts = Counter(clean_str(r["COMMODITY_RELEASED_TYPE"]) or "(BLANK)" for r in non_gas_rows)

    # main() writes this to results/eta_audit_results.json.
    # 由 main() 把结果写入 results/eta_audit_results.json；报告和 PPT 的数字都从那里读取。
    return {
        "observation_window": f"{start_year}-{end_year}",
        "sample_flow": {
            "raw_total_records": len(raw_rows),
            "period_records": len(period_rows),
            "natural_gas_records": len(gas_rows),
            "missing_release_records_excluded": len(missing_release_rows),
            "zero_release_records_excluded": len(zero_release_rows),
            "positive_release_candidates": len(positive_release_rows),
            "reporting_anomalies_excluded": len(anomaly_rows),
            "final_clean_sample_n": total_n,
            "records_outside_window": len(raw_rows) - len(period_rows),
            "non_natural_gas_excluded": len(non_gas_rows),
            "non_natural_gas_by_commodity": dict(commodity_counts.most_common()),
        },
        "blank_commodity_audit": {
            "total_count": len(blank_commodity_rows),
            **{k: v for k, v in casualty_summary(blank_commodity_rows).items() if k != "incident_count"},
            "years": dict(sorted(Counter(r["IYEAR"] for r in blank_commodity_rows).items())),
            "note": "COMMODITY_RELEASED_TYPE is empty, so the gas type cannot be confirmed. These records are excluded from the primary sample; see sensitivity specification 5.",
        },
        "anomalous_records_audit": [
            {
                "report_number": r["REPORT_NUMBER"].strip(),
                "year": int(r["IYEAR"]),
                "cause": clean_str(r["CAUSE"]),
                "fatalities": int(parse_float(r["FATAL"]) or 0.0),
                "injuries": int(parse_float(r["INJURE"]) or 0.0),
                "release_mcf": parse_float(r["UNINTENTIONAL_RELEASE"]),
                "cause_details": r["CAUSE_DETAILS"].strip()[:80],
            }
            for r in anomaly_rows
        ],
        "missing_release_audit": {
            "total_count": len(missing_release_rows),
            "years": dict(Counter(r["IYEAR"] for r in missing_release_rows)),
            "severe_incident_count": sum(has_severe_harm(r) for r in missing_release_rows),
            "fatalities": sum(int(parse_float(r["FATAL"]) or 0.0) for r in missing_release_rows),
            "injuries": sum(int(parse_float(r["INJURE"]) or 0.0) for r in missing_release_rows),
        },
        "unignited_explosion_field_audit": {
            "total_unignited": unignited_n,
            "blank_explosion_count": unignited_explosion_blank_cnt,
            "no_explosion_count": unignited_explosion_no_cnt,
            "note": "When IGNITE_IND is NO the form skips the explosion question, so EXPLODE_IND may be blank. Both blank and NO are treated as no explosion.",
        },
        "casualty_totals": {
            "total_incidents": total_n,
            "severe_incidents": severe_incidents_total,
            "severe_incident_rate": round(severe_incidents_total / total_n, 4) if total_n > 0 else 0.0,
            "fatalities": total_fatalities,
            "injuries": total_injuries,
            "total_casualties": total_fatalities + total_injuries,
        },
        "node_probabilities": {
            "p_ignition": {
                "point_estimate": round(p_ignite, 4),
                "count": ignited_n,
                "denominator": total_n,
                "wilson_ci_95": wilson_score_interval(ignited_n, total_n),
            },
            "p_explosion_given_ignition": {
                "point_estimate": round(p_explode_given_ignite, 4),
                "count": exploded_n,
                "denominator": ignited_n,
                "wilson_ci_95": wilson_score_interval(exploded_n, ignited_n),
            },
            "p_severe_given_ie": {
                "point_estimate": round(p_severe_given_ie, 4),
                "count": len(paths["Path_1_IE_Severe"]),
                "denominator": exploded_n,
                "wilson_ci_95": wilson_score_interval(len(paths["Path_1_IE_Severe"]), exploded_n),
            },
            "p_severe_given_ine": {
                "point_estimate": round(p_severe_given_ine, 4),
                "count": len(paths["Path_3_INE_Severe"]),
                "denominator": unexploded_ignited_n,
                "wilson_ci_95": wilson_score_interval(len(paths["Path_3_INE_Severe"]), unexploded_ignited_n),
            },
            "p_severe_given_ni": {
                "point_estimate": round(p_severe_given_ni, 4),
                "count": len(paths["Path_5_NI_Severe"]),
                "denominator": unignited_n,
                "wilson_ci_95": wilson_score_interval(len(paths["Path_5_NI_Severe"]), unignited_n),
            },
        },
        "chain_rule_verification": {
            "Path_1": {"telescoping_product": round(chain_p1, 4), "empirical_fraction": round(len(paths["Path_1_IE_Severe"]) / total_n, 4)},
            "Path_2": {"telescoping_product": round(chain_p2, 4), "empirical_fraction": round(len(paths["Path_2_IE_NoSevere"]) / total_n, 4)},
            "Path_3": {"telescoping_product": round(chain_p3, 4), "empirical_fraction": round(len(paths["Path_3_INE_Severe"]) / total_n, 4)},
            "Path_4": {"telescoping_product": round(chain_p4, 4), "empirical_fraction": round(len(paths["Path_4_INE_NoSevere"]) / total_n, 4)},
            "Path_5": {"telescoping_product": round(chain_p5, 4), "empirical_fraction": round(len(paths["Path_5_NI_Severe"]) / total_n, 4)},
            "Path_6": {"telescoping_product": round(chain_p6, 4), "empirical_fraction": round(len(paths["Path_6_NI_NoSevere"]) / total_n, 4)},
            "sum_of_endpoints": round(chain_p1 + chain_p2 + chain_p3 + chain_p4 + chain_p5 + chain_p6, 6),
        },
        "endpoints": endpoints_data,
        "cause_stratification": cause_metrics,
        "outside_force_subcauses": outside_force_subcauses,
        "external_fire_initiating_cases": {
            "total_count": sum(c["external_fire_primary_cause_count"] for c in cause_metrics.values()),
            "explanation": "In these Other Outside Force Damage records a nearby fire or explosion damaged the pipe, so ignition came before the release. The tree therefore classifies recorded outcomes and does not claim a fixed time order for every incident.",
        },
    }


def compute_sensitivity_analysis() -> dict:
    """Repeat the headline rates under five sample definitions.
    用五种不同的样本口径重算主要比例，检验筛选规则会不会改变结论。"""
    all_rows = load_raw_rows()
    in_window = [r for r in all_rows if 2010 <= int(r["IYEAR"]) <= 2024]
    raw_rows = [r for r in in_window if clean_str(r["COMMODITY_RELEASED_TYPE"]) == "NATURAL GAS"]

    def evaluate_subset(rows: list, label: str) -> dict:
        # Main rates for one sample definition. 计算某一种口径下的主要比例。
        n = len(rows)
        sev_cnt = sum(has_severe_harm(r) for r in rows)
        fatal_cnt = sum(int(parse_float(r["FATAL"]) or 0.0) for r in rows)
        inj_cnt = sum(int(parse_float(r["INJURE"]) or 0.0) for r in rows)
        ign_cnt = sum(clean_str(r["IGNITE_IND"]) == "YES" for r in rows)
        exp_cnt = sum(clean_str(r["IGNITE_IND"]) == "YES" and clean_str(r["EXPLODE_IND"]) == "YES" for r in rows)

        return {
            "label": label,
            "sample_size": n,
            "p_ignition": round(ign_cnt / n, 4) if n > 0 else 0.0,
            "p_explosion_given_ignition": round(exp_cnt / ign_cnt, 4) if ign_cnt > 0 else 0.0,
            "severe_incidents": sev_cnt,
            "severe_rate": round(sev_cnt / n, 4) if n > 0 else 0.0,
            "fatalities": fatal_cnt,
            "injuries": inj_cnt,
            "total_casualties": fatal_cnt + inj_cnt,
        }

    # Spec 1: primary sample.
    # 口径 1：主样本。
    pos_rows = [r for r in raw_rows if (parse_float(r["UNINTENTIONAL_RELEASE"]) or 0.0) > 0.0]
    clean_rows = [r for r in pos_rows if not is_skip_conflict(r)]

    # Spec 2: primary sample plus the 12 ignite/explode conflict records (= all positive-release records)
    # Spec 3: primary sample plus records with a missing release volume. Two of the 61
    #         missing-volume records are also conflict records and stay excluded.
    # 口径 2：主样本 + 12 条点火/爆炸矛盾记录。
    # 口径 3：主样本 + 泄漏量缺失的记录（61 条中有 2 条同时是矛盾记录，仍然排除）。
    unknown_rows = [
        r for r in raw_rows
        if parse_float(r["UNINTENTIONAL_RELEASE"]) is None and not is_skip_conflict(r)
    ]
    spec3_rows = clean_rows + unknown_rows

    # Spec 4: every natural gas record in the window
    # Spec 5: primary sample plus blank-commodity records that pass the same release and conflict rules
    # 口径 4：时间窗口内全部天然气记录，不做其他筛选。
    # 口径 5：主样本 + 气体类型为空、但满足同样泄漏量和一致性规则的记录。
    blank_rows = [
        r for r in in_window
        if clean_str(r["COMMODITY_RELEASED_TYPE"]) == ""
        and (parse_float(r["UNINTENTIONAL_RELEASE"]) or 0.0) > 0.0
        and not is_skip_conflict(r)
    ]
    spec5_rows = clean_rows + blank_rows

    return {
        "spec_1_primary_clean": evaluate_subset(clean_rows, f"Primary sample (N={len(clean_rows):,})"),
        "spec_2_include_conflicts": evaluate_subset(pos_rows, f"Include {len(pos_rows) - len(clean_rows)} conflict records (N={len(pos_rows):,})"),
        "spec_3_include_unknown_volume": evaluate_subset(spec3_rows, f"Include {len(unknown_rows)} missing-volume records (N={len(spec3_rows):,})"),
        "spec_4_all_gas_records": evaluate_subset(raw_rows, f"All natural gas records (N={len(raw_rows):,})"),
        "spec_5_include_blank_commodity": evaluate_subset(spec5_rows, f"Include {len(blank_rows)} blank-commodity records (N={len(spec5_rows):,})"),
    }


def _pct(x: float, digits: int = 2) -> str:
    """0.5838 -> "58.38%"."""
    return f"{x * 100:.{digits}f}%"


def _ci(ci) -> str:
    """Confidence interval as text. 把置信区间写成文字。"""
    return f"[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]" if ci else "n/a"


def generate_markdown_report(res_2010: dict, res_2015: dict, sens: dict) -> str:
    """Write the results as Markdown tables (results/eta_summary_tables.md).
    把结果整理成 Markdown 表格，方便直接阅读和核对。"""
    md = []
    md.append("# Event Tree Analysis (Model M1): Summary Tables\n\n")
    md.append("Generated by `src/eta_pipeline.py` from the PHMSA gas distribution incident file.\n\n")

    md.append("## 1. Scope\n\n")
    md.append("- The tree classifies recorded outcomes among reportable incidents (49 CFR Part 191). "
              "Branch shares are conditional on an incident having been reported; they are not failure rates per mile-year.\n")
    ext = res_2010["external_fire_initiating_cases"]["total_count"]
    md.append(f"- In {ext} Other Outside Force Damage records a nearby fire or explosion damaged the pipe, so ignition came first. "
              "The tree does not claim a fixed time order for every incident.\n")
    md.append("- Minor leaks below the reporting thresholds are not in the data, which raises the apparent ignition share.\n\n")

    md.append("## 2. Sample flow (2010-2024)\n\n")
    md.append("| Step | Records remaining | Records excluded | Notes |\n")
    md.append("|---|---|---|---|\n")
    sf = res_2010["sample_flow"]
    by_c = ", ".join(f"{k.title() if k != '(BLANK)' else 'blank'} {v}" for k, v in sf["non_natural_gas_by_commodity"].items())
    md.append(f"| PHMSA records in file | {sf['raw_total_records']:,} | - | Incidents from January 2010 onward |\n")
    md.append(f"| Incident year 2010-2024 | {sf['period_records']:,} | {sf['records_outside_window']:,} | 2025-2026 records removed |\n")
    md.append(f"| Commodity = natural gas | {sf['natural_gas_records']:,} | {sf['non_natural_gas_excluded']:,} | {by_c} |\n")
    md.append(f"| Release volume > 0 | {sf['positive_release_candidates']:,} | "
              f"{sf['missing_release_records_excluded'] + sf['zero_release_records_excluded']:,} | "
              f"{sf['missing_release_records_excluded']} missing, {sf['zero_release_records_excluded']} zero |\n")
    md.append(f"| Remove ignite/explode conflicts | {sf['final_clean_sample_n']:,} | {sf['reporting_anomalies_excluded']:,} | IGNITE_IND = NO but EXPLODE_IND = YES |\n")
    md.append(f"| **Primary sample** | **{sf['final_clean_sample_n']:,}** | - | Used by the event tree and the neural network |\n\n")
    bc = res_2010["blank_commodity_audit"]
    md.append(f"Blank commodity records: {bc['total_count']} ({bc['severe_incident_count']} severe incidents, "
              f"{bc['fatalities']} fatalities, {bc['injuries']} hospitalizations).\n\n")

    md.append("## 3. Observation window comparison\n\n")
    md.append("| Metric | 2010-2024 | 2015-2024 | Difference |\n")
    md.append("|---|---|---|---|\n")
    n10, n15 = res_2010["sample_flow"]["final_clean_sample_n"], res_2015["sample_flow"]["final_clean_sample_n"]
    p_ign_10 = res_2010["node_probabilities"]["p_ignition"]["point_estimate"]
    p_ign_15 = res_2015["node_probabilities"]["p_ignition"]["point_estimate"]
    p_exp_10 = res_2010["node_probabilities"]["p_explosion_given_ignition"]["point_estimate"]
    p_exp_15 = res_2015["node_probabilities"]["p_explosion_given_ignition"]["point_estimate"]
    sev_rate_10 = res_2010["casualty_totals"]["severe_incident_rate"]
    sev_rate_15 = res_2015["casualty_totals"]["severe_incident_rate"]
    fat_10 = res_2010["casualty_totals"]["fatalities"]
    fat_15 = res_2015["casualty_totals"]["fatalities"]
    inj_10 = res_2010["casualty_totals"]["injuries"]
    inj_15 = res_2015["casualty_totals"]["injuries"]

    md.append(f"| Sample size (N) | {n10:,} | {n15:,} | {n10 - n15:,} incidents in 2010-2014 |\n")
    md.append(f"| P(Ignition) | {_pct(p_ign_10)} | {_pct(p_ign_15)} | {(p_ign_15 - p_ign_10) * 100:+.2f} pp |\n")
    md.append(f"| P(Explosion given ignition) | {_pct(p_exp_10)} | {_pct(p_exp_15)} | {(p_exp_15 - p_exp_10) * 100:+.2f} pp |\n")
    md.append(f"| Severe incident rate | {_pct(sev_rate_10)} | {_pct(sev_rate_15)} | {(sev_rate_15 - sev_rate_10) * 100:+.2f} pp |\n")
    md.append(f"| Fatalities / hospitalizations | {fat_10} / {inj_10} | {fat_15} / {inj_15} | - |\n\n")

    md.append("## 4. Sensitivity to the sample definition\n\n")
    md.append("| Specification | N | P(Ignition) | P(Explosion given ignition) | Severe incidents | Severe rate | Fatalities | Hospitalizations | Total casualties |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for s_data in sens.values():
        md.append(
            f"| {s_data['label']} | {s_data['sample_size']:,} | {_pct(s_data['p_ignition'])} | "
            f"{_pct(s_data['p_explosion_given_ignition'])} | {s_data['severe_incidents']} | "
            f"{_pct(s_data['severe_rate'])} | {s_data['fatalities']} | {s_data['injuries']} | {s_data['total_casualties']} |\n"
        )
    md.append("\n")

    md.append("## 5. Branch probabilities\n\n")
    md.append("| Branch | Count / denominator | Estimate | Wilson 95% CI |\n")
    md.append("|---|---|---|---|\n")
    node_labels = [
        ("p_ignition", "P(Ignition)"),
        ("p_explosion_given_ignition", "P(Explosion given ignition)"),
        ("p_severe_given_ie", "P(Severe harm given ignition and explosion)"),
        ("p_severe_given_ine", "P(Severe harm given ignition, no explosion)"),
        ("p_severe_given_ni", "P(Severe harm given no ignition)"),
    ]
    for key, label in node_labels:
        node = res_2010["node_probabilities"][key]
        md.append(f"| {label} | {node['count']} / {node['denominator']} | {_pct(node['point_estimate'])} | {_ci(node['wilson_ci_95'])} |\n")
    md.append("\n")

    md.append("Path probabilities as products of the branch probabilities:\n\n")
    cr = res_2010["chain_rule_verification"]
    formulas = [
        "P(I) x P(E|I) x P(S|I,E)",
        "P(I) x P(E|I) x (1 - P(S|I,E))",
        "P(I) x (1 - P(E|I)) x P(S|I,not E)",
        "P(I) x (1 - P(E|I)) x (1 - P(S|I,not E))",
        "(1 - P(I)) x P(S|not I)",
        "(1 - P(I)) x (1 - P(S|not I))",
    ]
    for i, f in enumerate(formulas, 1):
        c = cr[f"Path_{i}"]
        md.append(f"- Path {i}: {f} = {c['telescoping_product']} (observed share {c['empirical_fraction']})\n")
    md.append(f"- Sum of the six paths: {cr['sum_of_endpoints']:.6f}\n\n")

    md.append("## 6. Casualties by path\n\n")
    md.append("| Path | Outcome | Incidents | Share | Wilson 95% CI | Fatalities | Hospitalizations | Total |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for p_id, p_info in res_2010["endpoints"].items():
        md.append(
            f"| {p_id} | {p_info['title']} | {p_info['incident_count']} | "
            f"{_pct(p_info['unconditional_probability'])} | {_ci(p_info['wilson_ci_95'])} | "
            f"{p_info['fatalities']} | {p_info['injuries']} | {p_info['total_casualties']} |\n"
        )
    md.append("\n")

    md.append("## 7. Casualties by cause\n\n")
    md.append("| Cause | Incidents | Incident share | P(Ignition) | Severe incidents | Severe rate | Fatalities | Hospitalizations | Total casualties | Share of severe incidents |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for c_name, c_data in sorted(res_2010["cause_stratification"].items(), key=lambda x: x[1]["incident_count"], reverse=True):
        c_sev_pct = c_data["severe_incident_count"] / c_data["incident_count"] * 100
        md.append(
            f"| {c_name} | {c_data['incident_count']} | {_pct(c_data['incident_share'])} | "
            f"{_pct(c_data['p_ignition'])} | {c_data['severe_incident_count']} | "
            f"{c_sev_pct:.1f}% | {c_data['fatalities']} | {c_data['injuries']} | "
            f"{c_data['total_casualties']} | {_pct(c_data['severe_incident_share_of_all_severe'])} |\n"
        )
    md.append("\n")

    md.append("## 8. Other Outside Force Damage by sub-cause\n\n")
    md.append("| Sub-cause (CAUSE_DETAILS) | Incidents | Severe incidents | Fatalities | Hospitalizations |\n")
    md.append("|---|---|---|---|---|\n")
    for s in res_2010["outside_force_subcauses"]:
        md.append(f"| {s['cause_detail']} | {s['incident_count']} | {s['severe_incident_count']} | {s['fatalities']} | {s['injuries']} |\n")
    md.append("\n")

    return "".join(md)


def main():
    """Run both windows and the sensitivity analysis, check the totals, save the outputs.
    计算两个时间窗口和敏感性分析，核对总数，保存结果文件。"""
    print("Running event tree analysis")
    res_2010 = run_event_tree(2010, 2024)
    res_2015 = run_event_tree(2015, 2024)
    sens = compute_sensitivity_analysis()

    # Stop if the headline numbers change (different data file or changed rules).
    # 回归检查：如果数据或筛选规则被意外改动，脚本会直接报错，而不是悄悄写出错误数字。
    assert res_2010["sample_flow"]["final_clean_sample_n"] == 1271, "Primary sample should be 1,271 incidents"
    assert res_2010["casualty_totals"]["fatalities"] == 113, "Fatalities should be 113"
    assert res_2010["casualty_totals"]["injuries"] == 537, "Hospitalizations should be 537"
    assert res_2010["casualty_totals"]["total_casualties"] == 650, "Total casualties should be 650"
    assert res_2010["casualty_totals"]["severe_incidents"] == 287, "Severe incidents should be 287"
    assert len(res_2010["cause_stratification"]) == 8, "Expected 8 cause categories"

    out_json = RESULTS_DIR / "eta_audit_results.json"
    payload = {
        "primary_window_2010_2024": res_2010,
        "baseline_window_2015_2024": res_2015,
        "sensitivity_analysis": sens,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_json}")

    out_md = RESULTS_DIR / "eta_summary_tables.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(res_2010, res_2015, sens))
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
