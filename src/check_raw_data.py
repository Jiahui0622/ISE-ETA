"""Independent recount of the event tree numbers from the raw PHMSA file.
独立核对：用另一套写法，从原始 PHMSA 文件重新数一遍事件树的所有数字。

ISE548 Risk and Crisis Management, Group 1.

Written separately from eta_pipeline.py (standard library only, different code
path) so that the two can be compared. It recounts the sample flow, the six
event tree paths, the cause table and the Other Outside Force sub-causes, then
checks every figure against results/eta_audit_results.json.
这个脚本和 eta_pipeline.py 分开写（只用 Python 标准库，计数方式也不同），
两边结果一致才说明统计没有写错。它重算样本筛选、六条路径、按原因统计和外力子原因，
再与 results/eta_audit_results.json 逐项比较，一致输出 PASS，否则列出差异。

Run from the 03_数据与代码 folder:
    python -B src/check_raw_data.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "incident_gas_distribution_jan2010_present.txt"
ETA_RESULTS = BASE_DIR / "results" / "eta_audit_results.json"


def number(text: str):
    """Text -> float, or None if empty. 文本转数字，空值返回 None。"""
    text = (text or "").replace(",", "").strip()
    try:
        return float(text) if text else None
    except ValueError:
        return None


def upper(text: str) -> str:
    """Trim and upper-case. 去空格并转大写。"""
    return (text or "").strip().upper()


def casualties(rec: dict) -> tuple[int, int]:
    """(deaths, hospitalizations) of one incident. 一起事故的（死亡人数, 住院人数）。"""
    return int(number(rec["FATAL"]) or 0), int(number(rec["INJURE"]) or 0)


def recount() -> dict:
    """Recount everything from the raw file. 从原始文件重新计数。"""
    with RAW_FILE.open(encoding="latin-1", newline="") as f:
        records = list(csv.DictReader(f, delimiter="\t"))

    # Sample selection, step by step (same rules as the report, Table 5).
    # 按报告表 5 的规则逐步筛选样本。
    flow = {"raw_total_records": len(records)}
    window = [r for r in records if 2010 <= int(r["IYEAR"]) <= 2024]
    flow["period_records"] = len(window)

    commodity = Counter(upper(r["COMMODITY_RELEASED_TYPE"]) or "(BLANK)" for r in window)
    gas = [r for r in window if upper(r["COMMODITY_RELEASED_TYPE"]) == "NATURAL GAS"]
    flow["natural_gas_records"] = len(gas)
    flow["non_natural_gas_by_commodity"] = {k: v for k, v in commodity.items() if k != "NATURAL GAS"}

    volume = [number(r["UNINTENTIONAL_RELEASE"]) for r in gas]
    flow["missing_release_records_excluded"] = sum(v is None for v in volume)
    flow["zero_release_records_excluded"] = sum(v == 0 for v in volume)
    positive = [r for r, v in zip(gas, volume) if v is not None and v > 0]
    flow["positive_release_candidates"] = len(positive)

    conflict = [r for r in positive if upper(r["IGNITE_IND"]) == "NO" and upper(r["EXPLODE_IND"]) == "YES"]
    flow["reporting_anomalies_excluded"] = len(conflict)
    sample = [r for r in positive if r not in conflict]
    flow["final_clean_sample_n"] = len(sample)

    # Path number from (ignited, exploded, severe).
    # 用（是否点火, 是否爆炸, 是否严重）三个答案直接查出路径编号。
    path_of = {
        (True, True, True): 1, (True, True, False): 2,
        (True, False, True): 3, (True, False, False): 4,
        (False, False, True): 5, (False, False, False): 6,
    }
    paths = defaultdict(lambda: [0, 0, 0])  # incidents, fatalities, hospitalizations
    causes = defaultdict(lambda: [0, 0, 0, 0])  # incidents, severe, fatalities, hospitalizations
    osf = defaultdict(lambda: [0, 0, 0, 0])
    for r in sample:
        fat, inj = casualties(r)
        severe = fat > 0 or inj > 0
        ignited = upper(r["IGNITE_IND"]) == "YES"
        exploded = ignited and upper(r["EXPLODE_IND"]) == "YES"
        p = paths[path_of[(ignited, exploded, severe)]]
        p[0] += 1
        p[1] += fat
        p[2] += inj
        for table, key in ((causes, upper(r["CAUSE"])), (osf, upper(r["CAUSE_DETAILS"]) if upper(r["CAUSE"]) == "OTHER OUTSIDE FORCE DAMAGE" else None)):
            if key is None:
                continue
            row = table[key]
            row[0] += 1
            row[1] += severe
            row[2] += fat
            row[3] += inj

    return {"flow": flow, "paths": dict(paths), "causes": dict(causes), "osf": dict(osf)}


def compare(counts: dict) -> list[str]:
    """List every figure that differs from the saved results. 列出与保存结果不一致的所有数字。"""
    saved = json.loads(ETA_RESULTS.read_text(encoding="utf-8"))["primary_window_2010_2024"]
    problems = []

    def check(label, mine, theirs):
        if mine != theirs:
            problems.append(f"{label}: recount {mine} vs saved {theirs}")

    for key, value in counts["flow"].items():
        theirs = saved["sample_flow"].get(key)
        if key == "non_natural_gas_by_commodity":
            theirs = dict(theirs or {})
            value = dict(sorted(value.items()))
            theirs = dict(sorted(theirs.items()))
        check(f"sample_flow.{key}", value, theirs)

    for n, (inc, fat, inj) in sorted(counts["paths"].items()):
        key = next(k for k in saved["endpoints"] if k.startswith(f"Path_{n}_"))
        e = saved["endpoints"][key]
        check(f"Path {n}", (inc, fat, inj), (e["incident_count"], e["fatalities"], e["injuries"]))

    for cause, (inc, sev, fat, inj) in counts["causes"].items():
        c = saved["cause_stratification"][cause]
        check(cause, (inc, sev, fat, inj), (c["incident_count"], c["severe_incident_count"], c["fatalities"], c["injuries"]))

    saved_osf = {s["cause_detail"]: s for s in saved["outside_force_subcauses"]}
    for detail, (inc, sev, fat, inj) in counts["osf"].items():
        s = saved_osf.get(detail or "NOT RECORDED")
        if s is None:
            problems.append(f"Outside force sub-cause missing in saved results: {detail}")
            continue
        check(f"OSF {detail[:40]}", (inc, sev, fat, inj), (s["incident_count"], s["severe_incident_count"], s["fatalities"], s["injuries"]))

    totals = [sum(v[i] for v in counts["paths"].values()) for i in range(3)]
    check("total incidents / fatalities / hospitalizations", tuple(totals),
          (saved["casualty_totals"]["total_incidents"], saved["casualty_totals"]["fatalities"], saved["casualty_totals"]["injuries"]))
    return problems


def main():
    """Print the recount and the comparison result. 打印重算结果和比较结论。"""
    counts = recount()
    f = counts["flow"]
    print("Sample flow")
    print(f"  records in file            {f['raw_total_records']:>6,}")
    print(f"  incident year 2010-2024    {f['period_records']:>6,}")
    print(f"  natural gas                {f['natural_gas_records']:>6,}   excluded: {f['non_natural_gas_by_commodity']}")
    print(f"  release volume > 0         {f['positive_release_candidates']:>6,}   missing {f['missing_release_records_excluded']}, zero {f['zero_release_records_excluded']}")
    print(f"  primary sample             {f['final_clean_sample_n']:>6,}   ignite/explode conflicts removed: {f['reporting_anomalies_excluded']}")
    print("Paths (incidents, fatalities, hospitalizations)")
    for n, v in sorted(counts["paths"].items()):
        print(f"  Path {n}: {v[0]:>4} {v[1]:>4} {v[2]:>4}")

    problems = compare(counts)
    if problems:
        print("\nFAIL")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("\nPASS: recount matches results/eta_audit_results.json")


if __name__ == "__main__":
    main()
