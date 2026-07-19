from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "Q3"))
import solve_q3 as q3  # noqa: E402


TABLE_DIR = ROOT / "results" / "Q3" / "tables"
FIGURE_DIR = ROOT / "results" / "Q3" / "figures"
FROZEN = ROOT / "frozen" / "Q3" / "frozen_numbers.json"
REPORT_PATH = ROOT / "results" / "Q3" / "reports" / "q3_verification.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_utf8_bom(path: Path) -> bool:
    return path.read_bytes().startswith(b"\xef\xbb\xbf")


def close(a: float, b: float, tol: float = 1e-3) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    demand, capacity, _ = q3.build_q3_capacity(cold_start=True)
    data = q3.make_model_data(demand, capacity)
    schedule = pd.read_csv(TABLE_DIR / "q3_schedule_compromise.csv", encoding="utf-8-sig").fillna("")
    fulfillment = pd.read_csv(TABLE_DIR / "q3_demand_fulfillment.csv", encoding="utf-8-sig")
    milestones = pd.read_csv(TABLE_DIR / "q3_milestone_fulfillment.csv", encoding="utf-8-sig")
    candidates = pd.read_csv(TABLE_DIR / "q3_candidate_solutions.csv", encoding="utf-8-sig")
    pareto = pd.read_csv(TABLE_DIR / "q3_pareto_front.csv", encoding="utf-8-sig")
    strict = pd.read_csv(TABLE_DIR / "q3_schedule_strict_history.csv", encoding="utf-8-sig").fillna("")
    blocks = pd.read_csv(TABLE_DIR / "q3_schedule_blocks.csv", encoding="utf-8-sig")
    strict_demand, strict_capacity, _ = q3.build_q3_capacity(cold_start=False)
    strict_data = q3.make_model_data(strict_demand, strict_capacity)

    metrics, _, _ = q3.schedule_metrics(schedule, data)
    strict_metrics, _, _ = q3.schedule_metrics(strict, strict_data)
    schedule_validation = q3.validate_schedule(schedule, data)
    strict_validation = q3.validate_schedule(strict, strict_data)

    checks: dict[str, bool] = {}
    checks["schedule_has_644_slots"] = len(schedule) == 644
    checks["slot_keys_unique"] = not schedule.duplicated(["生产日期", "班次", "产线"]).any()
    checks["fourteen_lines"] = schedule["产线"].nunique() == 14
    checks["forty_six_slots_per_line"] = bool((schedule.groupby("产线").size() == 46).all())
    checks["twenty_three_dates"] = schedule["生产日期"].nunique() == 23
    checks["schedule_feasible"] = bool(schedule_validation["passed"])
    checks["strict_schedule_feasible"] = bool(strict_validation["passed"])
    checks["strict_has_no_cold_start_quantity"] = close(strict_metrics["cold_start_quantity"], 0.0)
    checks["strict_hits_exact_anchor"] = close(strict_metrics["total_production"], frozen["strict_history_production_max_anchor"], 1e-2)
    checks["total_demand_matches"] = close(metrics["total_demand"], frozen["main_metrics"]["total_demand"])
    checks["total_production_matches"] = close(metrics["total_production"], frozen["main_metrics"]["total_production"])
    checks["shortage_matches"] = close(metrics["total_shortage"], frozen["main_metrics"]["total_shortage"])
    checks["completion_rate_matches"] = close(metrics["completion_rate"], frozen["main_metrics"]["completion_rate"], 1e-9)
    checks["switch_cost_matches"] = close(metrics["switch_cost"], frozen["main_metrics"]["switch_cost"])
    checks["load_imbalance_matches"] = close(metrics["load_imbalance"], frozen["main_metrics"]["load_imbalance"], 1e-9)
    checks["customer_satisfaction_matches"] = close(metrics["customer_satisfaction"], frozen["main_metrics"]["customer_satisfaction"], 1e-9)
    checks["fulfillment_has_28_products"] = len(fulfillment) == 28
    checks["milestones_have_140_rows"] = len(milestones) == 28 * 5
    checks["candidate_count_matches"] = len(candidates) == frozen["candidate_solution_count"]
    checks["pareto_count_matches"] = len(pareto) == frozen["pareto_solution_count"]
    checks["pareto_flags_recompute"] = bool(q3.nondominated(candidates).equals(candidates["非支配"].astype(bool)))
    checks["selected_solution_is_pareto"] = frozen["selected_solution"] in set(pareto["方案"])
    checks["cold_start_quantity_is_15000"] = close(metrics["cold_start_quantity"], 15000.0)
    checks["schedule_blocks_have_70_rows"] = len(blocks) == 70
    checks["small_batch_repair_resolved_all"] = frozen["small_batch_repair"]["small_slots_after"] == 0
    checks["small_batch_repair_lost_no_quantity"] = close(frozen["small_batch_repair"]["dropped_quantity"], 0.0)
    checks["recommended_plan_hits_anchor"] = close(metrics["total_production"], frozen["production_max_anchor"], 1e-2)
    severe_fragments = 0
    product_index = {key: i for i, key in enumerate(data.products)}
    for row in schedule.itertuples(index=False):
        if not row.产品:
            continue
        i = product_index[(row.产品, row.酒质)]
        threshold = max(
            q3.MIN_ACTIVE_QUANTITY,
            0.10
            * min(
                float(row.班次产能估计),
                float(data.demand.loc[i, "计划生产量"]),
                data.pair_min_batch[(i, row.产线)],
            ),
        )
        if float(row.计划生产量) + 1e-6 < threshold:
            severe_fragments += 1
    checks["no_severe_small_batch_fragments"] = severe_fragments == 0
    checks["history_hash_matches"] = sha256(ROOT / "附件1.csv") == frozen["input_hashes"]["附件1.csv"]
    checks["demand_hash_matches"] = sha256(ROOT / "附件3.csv") == frozen["input_hashes"]["附件3.csv"]

    for name in [
        "q3_capacity_matrix_q95.csv",
        "q3_candidate_solutions.csv",
        "q3_pareto_front.csv",
        "q3_schedule_compromise.csv",
        "q3_schedule_blocks.csv",
        "q3_demand_fulfillment.csv",
        "q3_milestone_fulfillment.csv",
        "q3_schedule_greedy.csv",
        "q3_schedule_strict_history.csv",
        "q3_method_comparison.csv",
        "q3_sensitivity.csv",
    ]:
        checks[f"utf8_bom:{name}"] = is_utf8_bom(TABLE_DIR / name)
    for name in ["q3_schedule_heatmap.png", "q3_pareto_tradeoffs.png", "q3_line_loads.png", "q3_method_comparison.png"]:
        checks[f"figure_nonempty:{name}"] = (FIGURE_DIR / name).stat().st_size > 10_000

    failed = [name for name, ok in checks.items() if not ok]
    report = {
        "passed": not failed,
        "check_count": len(checks),
        "failed": failed,
        "checks": checks,
        "recomputed": {
            "total_production": metrics["total_production"],
            "total_shortage": metrics["total_shortage"],
            "product_switches": metrics["product_switches"],
            "quality_switches": metrics["quality_switches"],
            "switch_cost": metrics["switch_cost"],
            "load_imbalance": metrics["load_imbalance"],
            "customer_satisfaction": metrics["customer_satisfaction"],
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
