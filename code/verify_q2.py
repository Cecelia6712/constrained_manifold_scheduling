from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results" / "Q2" / "tables"
FIGURE = ROOT / "results" / "Q2" / "figures"
RUN = ROOT / "results" / "Q2" / "experiments" / "round1" / "run_summary.json"
FROZEN = ROOT / "frozen" / "Q2" / "frozen_numbers.json"
OUT = ROOT / "results" / "Q2" / "reports" / "q2_verification.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    schedule = pd.read_csv(TABLE / "q2_schedule_milp.csv", encoding="utf-8-sig").fillna({"产品": "", "酒质": ""})
    demand = pd.read_csv(ROOT / "附件2.csv", encoding="utf-8-sig")
    capacity = pd.read_csv(TABLE / "q2_capacity_matrix_q95.csv", encoding="utf-8-sig")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    run = json.loads(RUN.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["schedule_has_140_slots"] = len(schedule) == 140
    checks["slot_keys_unique"] = not schedule.duplicated(["生产日期", "班次", "产线"]).any()
    checks["fourteen_lines"] = schedule["产线"].nunique() == 14
    checks["ten_slots_per_line"] = bool((schedule.groupby("产线").size() == 10).all())

    cap_map = {
        (r.产品, r.酒质, r.产线): float(r.估计班次产能)
        for r in capacity.itertuples(index=False)
    }
    active = schedule[schedule["产品"] != ""]
    checks["all_assignments_compatible"] = all((r.产品, r.酒质, r.产线) in cap_map for r in active.itertuples(index=False))
    checks["no_capacity_violation"] = all(
        float(r.计划生产量) <= cap_map[(r.产品, r.酒质, r.产线)] + 1e-6
        for r in active.itertuples(index=False)
    )
    checks["positive_active_quantities"] = bool((active["计划生产量"] >= 1).all())

    produced = active.groupby(["产品", "酒质"])["计划生产量"].sum().to_dict()
    demand_map = {(r.产品, r.酒质): float(r.计划生产量) for r in demand.itertuples(index=False)}
    checks["no_demand_overshoot"] = all(produced.get(key, 0.0) <= value + 1e-6 for key, value in demand_map.items())
    total_demand = float(demand["计划生产量"].sum())
    total_production = float(active["计划生产量"].sum())
    total_shortage = total_demand - total_production
    numbers = frozen["numbers"]
    checks["frozen_total_demand_matches"] = abs(numbers["total_demand"] - total_demand) < 1e-6
    checks["frozen_total_production_matches"] = abs(numbers["total_production"] - total_production) < 1e-6
    checks["frozen_shortage_matches"] = abs(numbers["total_shortage"] - total_shortage) < 1e-6
    checks["frozen_switches_match"] = int(schedule["产品换产标记"].sum()) == int(numbers["product_switches"])
    checks["run_summary_matches_frozen"] = abs(run["milp_metrics"]["total_production"] - numbers["total_production"]) < 1e-6

    checks["history_hash_matches"] = sha256(ROOT / "附件1.csv") == frozen["input_hashes"]["附件1.csv"]["sha256"]
    checks["demand_hash_matches"] = sha256(ROOT / "附件2.csv") == frozen["input_hashes"]["附件2.csv"]["sha256"]
    for path in TABLE.glob("*.csv"):
        checks[f"utf8_bom:{path.name}"] = path.read_bytes().startswith(b"\xef\xbb\xbf")
    for name in ("q2_schedule_heatmap.png", "q2_fulfillment_rates.png", "q2_sensitivity.png"):
        path = FIGURE / name
        checks[f"figure_nonempty:{name}"] = path.exists() and path.stat().st_size > 10_000

    report = {
        "passed": all(checks.values()),
        "check_count": len(checks),
        "failed": [name for name, value in checks.items() if not value],
        "checks": checks,
        "recomputed": {
            "total_demand": total_demand,
            "total_production": total_production,
            "total_shortage": total_shortage,
            "product_switches": int(schedule["产品换产标记"].sum()),
            "quality_switches": int(schedule["酒质切换标记"].sum()),
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
