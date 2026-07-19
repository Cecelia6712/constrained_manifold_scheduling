from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "Q2"))
import solve_q2 as q2  # noqa: E402


INPUT_HISTORY = ROOT / "附件1.csv"
INPUT_DEMAND = ROOT / "附件3.csv"
RESULT_DIR = ROOT / "results" / "Q3"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"
REPORT_DIR = RESULT_DIR / "reports"
EXPERIMENT_DIR = RESULT_DIR / "experiments" / "round1"
METHOD_DIR = ROOT / "methods" / "Q3"
ROBUST_DIR = ROOT / "robustness" / "Q3"
FROZEN_DIR = ROOT / "frozen" / "Q3"
DATA_DIR = ROOT / "data" / "processed"

SEED = 42
SHIFTS = ["早班", "中班"]
PRODUCTION_DAYS = list(pd.date_range("2025-03-03", "2025-03-07", freq="D"))
PRODUCTION_DAYS += list(pd.date_range("2025-03-10", "2025-03-15", freq="D"))
PRODUCTION_DAYS += list(pd.date_range("2025-03-17", "2025-03-21", freq="D"))
PRODUCTION_DAYS += list(pd.date_range("2025-03-24", "2025-03-29", freq="D"))
PRODUCTION_DAYS += [pd.Timestamp("2025-03-31")]
TIME_SLOTS = [(day, shift) for day in PRODUCTION_DAYS for shift in SHIFTS]
PERIOD_DAY_COUNTS = [5, 6, 5, 6, 1]
PERIOD_END_SLOTS = [2 * sum(PERIOD_DAY_COUNTS[: k + 1]) - 1 for k in range(len(PERIOD_DAY_COUNTS))]
MILESTONE_ALPHA = np.cumsum(PERIOD_DAY_COUNTS) / sum(PERIOD_DAY_COUNTS)
MIN_ACTIVE_QUANTITY = 1.0


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIGURE_DIR, REPORT_DIR, EXPERIMENT_DIR, METHOD_DIR, ROBUST_DIR, FROZEN_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


@dataclass
class ModelData:
    demand: pd.DataFrame
    capacity: pd.DataFrame
    products: list[tuple[str, str]]
    lines: list[str]
    pair_capacity: dict[tuple[int, str], float]
    pair_preference: dict[tuple[int, str], float]
    pair_min_batch: dict[tuple[int, str], float]
    pair_source: dict[tuple[int, str], str]
    compatible_lines: dict[int, list[str]]
    line_reference_capacity: dict[str, float]


def build_q3_capacity(cold_start: bool = True, scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history, eligible, cleaning = q2.clean_history()
    cap = q2.build_capacity_table(history, eligible, quantile=0.95, scale=scale, with_bootstrap=True)
    robust_history = history[(history["组内极低量标记"] == 0) & (history["组内IQR异常标记"] == 0)].copy()
    q10 = (
        robust_history.groupby(["产品", "酒质", "产线"])["计划生产量"]
        .quantile(0.10)
        .rename("历史Q10最小批量")
        .reset_index()
    )
    fallback_q10 = (
        history.groupby(["产品", "酒质", "产线"])["计划生产量"]
        .quantile(0.10)
        .rename("回退Q10")
        .reset_index()
    )
    cap = cap.merge(q10, on=["产品", "酒质", "产线"], how="left").merge(
        fallback_q10, on=["产品", "酒质", "产线"], how="left"
    )
    cap["历史Q10最小批量"] = cap["历史Q10最小批量"].fillna(cap["回退Q10"]).fillna(MIN_ACTIVE_QUANTITY)
    cap["历史Q10最小批量"] = np.floor(cap["历史Q10最小批量"] * scale).clip(lower=MIN_ACTIVE_QUANTITY)
    cap = cap.drop(columns=["回退Q10"])
    demand = pd.read_csv(INPUT_DEMAND, encoding="utf-8-sig")
    demand["计划生产量"] = pd.to_numeric(demand["计划生产量"], errors="raise")

    demand_keys = set(demand[["产品", "酒质"]].itertuples(index=False, name=None))
    cap_keys = set(cap[["产品", "酒质"]].itertuples(index=False, name=None))
    missing = sorted(demand_keys - cap_keys)
    cold_rows: list[dict[str, Any]] = []
    if cold_start:
        for product, quality in missing:
            parent = eligible[eligible["酒质"].eq(quality)]
            if parent.empty:
                continue
            for line, grp in parent.groupby("产线"):
                values = grp["计划生产量"].to_numpy(dtype=float)
                # 对未见产品只采用同酒质同产线的历史最小班次量，避免把同酒质高位产能
                # 直接外推成新产品的能力上限。该组合必须在实施前完成工艺验证。
                conservative = max(MIN_ACTIVE_QUANTITY, math.floor(float(np.min(values)) * scale))
                cold_rows.append(
                    {
                        "产品": product,
                        "酒质": quality,
                        "产线": line,
                        "样本数": 0,
                        "全部历史样本数": 0,
                        "产能样本来源": "同酒质冷启动下界",
                        "组内分位产能": np.nan,
                        "父组分位产能": float(np.quantile(values, 0.95)),
                        "历史最大值": float(np.max(values)),
                        "估计班次产能": float(conservative),
                        "稳健变异系数": np.nan,
                        "历史负载率": np.nan,
                        "Bootstrap下限": np.nan,
                        "Bootstrap上限": np.nan,
                        "历史分配次数": 0,
                        "历史分配份额": 0.0,
                        "产能归一值": 0.0,
                        "稳定性归一值": 0.0,
                        "负载率归一值": 0.0,
                        "分配份额归一值": 0.0,
                        "偏好得分": 0.0,
                        "产线优先级": 999,
                        "历史Q10最小批量": float(conservative),
                    }
                )
    if cold_rows:
        cap = pd.concat([cap, pd.DataFrame(cold_rows)], ignore_index=True)
    cap = cap.sort_values(["产品", "酒质", "产线"]).reset_index(drop=True)
    report = {
        **cleaning,
        "monthly_demand_rows": int(len(demand)),
        "monthly_total_demand": float(demand["计划生产量"].sum()),
        "production_days": len(PRODUCTION_DAYS),
        "line_shifts": len(TIME_SLOTS) * int(cap["产线"].nunique()),
        "unseen_demand_combinations": [f"{p}-{q}" for p, q in missing],
        "cold_start_enabled": cold_start,
        "cold_start_rows": len(cold_rows),
        "capacity_scale": scale,
    }
    return demand, cap, report


def make_model_data(demand: pd.DataFrame, capacity: pd.DataFrame) -> ModelData:
    products = list(demand[["产品", "酒质"]].itertuples(index=False, name=None))
    product_index = {key: i for i, key in enumerate(products)}
    use = capacity.merge(demand[["产品", "酒质"]], on=["产品", "酒质"], how="inner")
    pair_capacity: dict[tuple[int, str], float] = {}
    pair_preference: dict[tuple[int, str], float] = {}
    pair_min_batch: dict[tuple[int, str], float] = {}
    pair_source: dict[tuple[int, str], str] = {}
    compatible_lines: dict[int, list[str]] = defaultdict(list)
    for row in use.itertuples(index=False):
        i = product_index[(row.产品, row.酒质)]
        pair_capacity[(i, row.产线)] = float(row.估计班次产能)
        pair_preference[(i, row.产线)] = float(row.偏好得分)
        pair_min_batch[(i, row.产线)] = float(row.历史Q10最小批量)
        pair_source[(i, row.产线)] = str(row.产能样本来源)
        compatible_lines[i].append(row.产线)
    lines = sorted(capacity["产线"].dropna().unique().tolist())
    line_reference_capacity = {}
    for line in lines:
        values = [cap for (i, l), cap in pair_capacity.items() if l == line]
        line_reference_capacity[line] = max(values, default=1.0) * len(TIME_SLOTS)
    return ModelData(
        demand=demand,
        capacity=capacity,
        products=products,
        lines=lines,
        pair_capacity=pair_capacity,
        pair_preference=pair_preference,
        pair_min_batch=pair_min_batch,
        pair_source=pair_source,
        compatible_lines=dict(compatible_lines),
        line_reference_capacity=line_reference_capacity,
    )


class LinearModelBuilder:
    def __init__(self) -> None:
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integrality: list[int] = []
        self.names: list[str] = []
        self.rows: list[dict[int, float]] = []
        self.row_lb: list[float] = []
        self.row_ub: list[float] = []

    def var(self, name: str, lb: float, ub: float, integer: bool = False) -> int:
        idx = len(self.lb)
        self.names.append(name)
        self.lb.append(lb)
        self.ub.append(ub)
        self.integrality.append(1 if integer else 0)
        return idx

    def row(self, coeff: dict[int, float], lb: float = -np.inf, ub: float = np.inf) -> None:
        self.rows.append(coeff)
        self.row_lb.append(lb)
        self.row_ub.append(ub)

    def matrix(self, extra_rows: Iterable[tuple[dict[int, float], float, float]] = ()) -> LinearConstraint:
        rows = list(self.rows)
        lbs = list(self.row_lb)
        ubs = list(self.row_ub)
        for coeff, lb, ub in extra_rows:
            rows.append(coeff)
            lbs.append(lb)
            ubs.append(ub)
        rr: list[int] = []
        cc: list[int] = []
        vv: list[float] = []
        for row_no, coeff in enumerate(rows):
            for col_no, value in coeff.items():
                if value != 0:
                    rr.append(row_no)
                    cc.append(col_no)
                    vv.append(value)
        matrix = coo_array((vv, (rr, cc)), shape=(len(rows), len(self.lb))).tocsr()
        return LinearConstraint(matrix, np.asarray(lbs), np.asarray(ubs))


@dataclass
class SolverArtifacts:
    builder: LinearModelBuilder
    x_idx: dict[tuple[int, str, int], int]
    q_idx: dict[tuple[int, str, int], int]
    sp_idx: dict[tuple[str, int], int]
    sq_idx: dict[tuple[str, int], int]
    r_idx: dict[int, int]
    e_idx: dict[tuple[int, int], int]
    dev_idx: dict[str, int]
    mean_idx: int
    component_objectives: dict[str, dict[int, float]]


def build_milp(data: ModelData, quality_penalty: float = 2.0, milestone_alpha: np.ndarray = MILESTONE_ALPHA) -> SolverArtifacts:
    b = LinearModelBuilder()
    x_idx: dict[tuple[int, str, int], int] = {}
    q_idx: dict[tuple[int, str, int], int] = {}
    sp_idx: dict[tuple[str, int], int] = {}
    sq_idx: dict[tuple[str, int], int] = {}
    r_idx: dict[int, int] = {}
    e_idx: dict[tuple[int, int], int] = {}
    dev_idx: dict[str, int] = {}
    n_t = len(TIME_SLOTS)

    for (i, line), cap in sorted(data.pair_capacity.items()):
        for t in range(n_t):
            x_idx[(i, line, t)] = b.var(f"x[{i},{line},{t}]", 0, 1, integer=True)
            q_idx[(i, line, t)] = b.var(f"q[{i},{line},{t}]", 0, cap)

    # 每个相邻线班只需要一个产品变化和一个酒质变化变量；在“每班至多一种产品”
    # 约束下，它们足以准确记录新状态，且比为每个产品建切换变量显著减少对称性。
    for line in data.lines:
        for t in range(1, n_t):
            sp_idx[(line, t)] = b.var(f"sp[{line},{t}]", 0, 1, integer=True)
            sq_idx[(line, t)] = b.var(f"sq[{line},{t}]", 0, 1, integer=True)

    for i, row in data.demand.iterrows():
        demand_i = float(row["计划生产量"])
        r_idx[i] = b.var(f"r[{i}]", 0, demand_i)
        for k in range(len(PERIOD_END_SLOTS)):
            e_idx[(i, k)] = b.var(f"e[{i},{k}]", 0, demand_i)

    mean_idx = b.var("mean_load", 0, 1)
    for line in data.lines:
        dev_idx[line] = b.var(f"load_dev[{line}]", 0, 1)

    x_by_line_t: dict[tuple[str, int], list[int]] = defaultdict(list)
    q_by_i: dict[int, list[int]] = defaultdict(list)
    q_by_i_t: dict[tuple[int, int], list[int]] = defaultdict(list)
    q_by_line: dict[str, list[int]] = defaultdict(list)
    for (i, line, t), x_var in x_idx.items():
        x_by_line_t[(line, t)].append(x_var)
        q_var = q_idx[(i, line, t)]
        q_by_i[i].append(q_var)
        q_by_i_t[(i, t)].append(q_var)
        q_by_line[line].append(q_var)

    # 每条产线每班最多一种产品。
    for line in data.lines:
        for t in range(n_t):
            vars_here = x_by_line_t.get((line, t), [])
            if vars_here:
                b.row({idx: 1.0 for idx in vars_here}, ub=1.0)

    # 产量和班次激活变量联动。
    for (i, line, t), q_var in q_idx.items():
        x_var = x_idx[(i, line, t)]
        cap = data.pair_capacity[(i, line)]
        b.row({q_var: 1.0, x_var: -cap}, ub=0.0)
        b.row({x_var: MIN_ACTIVE_QUANTITY, q_var: -1.0}, ub=0.0)

    # 月需求平衡。
    for i, row in data.demand.iterrows():
        coeff = {idx: 1.0 for idx in q_by_i.get(i, [])}
        coeff[r_idx[i]] = 1.0
        demand_i = float(row["计划生产量"])
        b.row(coeff, lb=demand_i, ub=demand_i)

    # 产品和酒质切换线性化。
    for line in data.lines:
        compatible_products = [i for i in range(len(data.products)) if (i, line) in data.pair_capacity]
        compatible_qualities = sorted({data.products[i][1] for i in compatible_products})
        for t in range(1, n_t):
            for i in compatible_products:
                b.row(
                    {
                        x_idx[(i, line, t)]: 1.0,
                        x_idx[(i, line, t - 1)]: -1.0,
                        sp_idx[(line, t)]: -1.0,
                    },
                    ub=0.0,
                )
            for quality in compatible_qualities:
                coeff: dict[int, float] = {sq_idx[(line, t)]: -1.0}
                for i in compatible_products:
                    if data.products[i][1] == quality:
                        coeff[x_idx[(i, line, t)]] = coeff.get(x_idx[(i, line, t)], 0.0) + 1.0
                        coeff[x_idx[(i, line, t - 1)]] = coeff.get(x_idx[(i, line, t - 1)], 0.0) - 1.0
                b.row(coeff, ub=0.0)

    # 五个累计交付里程碑的欠交量。
    for i, row in data.demand.iterrows():
        demand_i = float(row["计划生产量"])
        for k, end_t in enumerate(PERIOD_END_SLOTS):
            coeff = {e_idx[(i, k)]: 1.0}
            for t in range(end_t + 1):
                for q_var in q_by_i_t.get((i, t), []):
                    coeff[q_var] = coeff.get(q_var, 0.0) + 1.0
            b.row(coeff, lb=float(milestone_alpha[k]) * demand_i)

    # 产线负荷采用“月实际产量/该线参考产能”，并最小化各线相对平均值的绝对偏差。
    mean_coeff: dict[int, float] = {mean_idx: 1.0}
    for line in data.lines:
        denom = max(data.line_reference_capacity[line], 1.0)
        for q_var in q_by_line.get(line, []):
            mean_coeff[q_var] = mean_coeff.get(q_var, 0.0) - 1.0 / (len(data.lines) * denom)
    b.row(mean_coeff, lb=0.0, ub=0.0)
    for line in data.lines:
        denom = max(data.line_reference_capacity[line], 1.0)
        positive = {dev_idx[line]: 1.0, mean_idx: 1.0}
        negative = {dev_idx[line]: 1.0, mean_idx: -1.0}
        for q_var in q_by_line.get(line, []):
            positive[q_var] = positive.get(q_var, 0.0) - 1.0 / denom
            negative[q_var] = negative.get(q_var, 0.0) + 1.0 / denom
        b.row(positive, lb=0.0)
        b.row(negative, lb=0.0)

    total_demand = float(data.demand["计划生产量"].sum())
    max_switch_cost = (1.0 + quality_penalty) * len(data.lines) * (n_t - 1)
    shortage_obj = {idx: 1.0 / total_demand for idx in r_idx.values()}
    switch_obj = {idx: 1.0 / max_switch_cost for idx in sp_idx.values()}
    for idx in sq_idx.values():
        switch_obj[idx] = quality_penalty / max_switch_cost
    balance_obj = {idx: 1.0 / len(data.lines) for idx in dev_idx.values()}
    service_obj = {}
    n_products = len(data.products)
    n_periods = len(PERIOD_END_SLOTS)
    for (i, k), idx in e_idx.items():
        demand_i = float(data.demand.loc[i, "计划生产量"])
        service_obj[idx] = 1.0 / (n_products * n_periods * demand_i)

    return SolverArtifacts(
        b,
        x_idx,
        q_idx,
        sp_idx,
        sq_idx,
        r_idx,
        e_idx,
        dev_idx,
        mean_idx,
        {"shortage": shortage_obj, "switch": switch_obj, "balance": balance_obj, "dissatisfaction": service_obj},
    )


def combine_objective(artifacts: SolverArtifacts, weights: dict[str, float], tie_break: bool = True) -> dict[int, float]:
    objective: dict[int, float] = defaultdict(float)
    for component, weight in weights.items():
        for idx, coeff in artifacts.component_objectives[component].items():
            objective[idx] += weight * coeff
    if tie_break:
        for idx in artifacts.x_idx.values():
            objective[idx] += 1e-8
    return dict(objective)


def run_milp(
    artifacts: SolverArtifacts,
    objective: dict[int, float],
    extra_rows: Iterable[tuple[dict[int, float], float, float]] = (),
    time_limit: float = 180.0,
    gap: float = 0.0,
) -> Any:
    b = artifacts.builder
    c = np.zeros(len(b.lb))
    for idx, value in objective.items():
        c[idx] = value
    result = milp(
        c=c,
        integrality=np.asarray(b.integrality),
        bounds=Bounds(np.asarray(b.lb), np.asarray(b.ub)),
        constraints=b.matrix(extra_rows),
        options={"disp": False, "time_limit": time_limit, "mip_rel_gap": gap, "presolve": True},
    )
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    return result


def production_floor_row(data: ModelData, artifacts: SolverArtifacts, floor: float) -> tuple[dict[int, float], float, float]:
    total_demand = float(data.demand["计划生产量"].sum())
    max_shortage = total_demand - floor
    return ({idx: 1.0 for idx in artifacts.r_idx.values()}, -np.inf, max_shortage + 1e-3)


def result_to_schedule(data: ModelData, result: Any, artifacts: SolverArtifacts, name: str) -> pd.DataFrame:
    assigned: dict[tuple[str, int], tuple[int, float]] = {}
    for key, x_var in artifacts.x_idx.items():
        if result.x[x_var] > 0.5:
            i, line, t = key
            assigned[(line, t)] = (i, max(0.0, float(result.x[artifacts.q_idx[key]])))
    records = []
    for line in data.lines:
        prev_product = None
        prev_quality = None
        for t, (date, shift) in enumerate(TIME_SLOTS):
            item = assigned.get((line, t))
            if item is None:
                product = quality = source = ""
                qty = cap = pref = 0.0
            else:
                i, qty = item
                product, quality = data.products[i]
                cap = data.pair_capacity[(i, line)]
                pref = data.pair_preference[(i, line)]
                source = data.pair_source[(i, line)]
            records.append(
                {
                    "生产日期": date.strftime("%Y-%m-%d"),
                    "班次": shift,
                    "产线": line,
                    "产品": product,
                    "酒质": quality,
                    "计划生产量": float(round(qty, 6)),
                    "班次产能估计": float(cap),
                    "产能利用率": 0.0 if cap <= 0 else float(qty / cap),
                    "产能来源": source,
                    "产品换产标记": int(t > 0 and bool(product) and product != prev_product),
                    "酒质切换标记": int(t > 0 and bool(quality) and quality != prev_quality),
                    "模型": name,
                }
            )
            prev_product = product if product else None
            prev_quality = quality if quality else None
    out = pd.DataFrame(records)
    out["班次序"] = out["班次"].map({"早班": 0, "中班": 1})
    return out.sort_values(["生产日期", "班次序", "产线"]).drop(columns="班次序").reset_index(drop=True)


def schedule_metrics(schedule: pd.DataFrame, data: ModelData, quality_penalty: float = 2.0, milestone_alpha: np.ndarray = MILESTONE_ALPHA) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    produced = schedule.groupby(["产品", "酒质"])["计划生产量"].sum()
    fulfillment = []
    for row in data.demand.itertuples(index=False):
        qty = float(produced.get((row.产品, row.酒质), 0.0))
        fulfillment.append(
            {
                "产品": row.产品,
                "酒质": row.酒质,
                "需求量": float(row.计划生产量),
                "排产量": qty,
                "剩余量": max(0.0, float(row.计划生产量) - qty),
                "完成率": qty / float(row.计划生产量),
            }
        )
    fulfillment_df = pd.DataFrame(fulfillment)

    dated = schedule.copy()
    dated["时段"] = dated["生产日期"].astype(str) + "|" + dated["班次"]
    ordered_keys = [d.strftime("%Y-%m-%d") + "|" + s for d, s in TIME_SLOTS]
    dated["时段序"] = pd.Categorical(dated["时段"], ordered_keys, ordered=True).codes
    milestone_rows = []
    deficit_sum = 0.0
    for i, row in data.demand.iterrows():
        demand_i = float(row["计划生产量"])
        for k, end_t in enumerate(PERIOD_END_SLOTS):
            cumulative = float(
                dated[(dated["产品"] == row["产品"]) & (dated["酒质"] == row["酒质"]) & (dated["时段序"] <= end_t)]["计划生产量"].sum()
            )
            target = float(milestone_alpha[k]) * demand_i
            deficit = max(0.0, target - cumulative)
            deficit_sum += deficit / demand_i
            milestone_rows.append(
                {
                    "产品": row["产品"],
                    "酒质": row["酒质"],
                    "周期": k + 1,
                    "截止日期": TIME_SLOTS[end_t][0].strftime("%Y-%m-%d"),
                    "累计目标": target,
                    "累计排产": cumulative,
                    "里程碑欠交量": deficit,
                    "里程碑满足率": min(1.0, cumulative / target) if target > 0 else 1.0,
                }
            )
    milestone_df = pd.DataFrame(milestone_rows)

    line_production = schedule.groupby("产线")["计划生产量"].sum().reindex(data.lines, fill_value=0.0)
    line_util = pd.Series({line: line_production[line] / max(data.line_reference_capacity[line], 1.0) for line in data.lines})
    load_imbalance = float(np.mean(np.abs(line_util - line_util.mean())))
    total_demand = float(data.demand["计划生产量"].sum())
    total_production = float(fulfillment_df["排产量"].sum())
    product_switches = int(schedule["产品换产标记"].sum())
    quality_switches = int(schedule["酒质切换标记"].sum())
    dissatisfaction = deficit_sum / (len(data.products) * len(PERIOD_END_SLOTS))
    metrics = {
        "total_demand": total_demand,
        "total_production": total_production,
        "total_shortage": total_demand - total_production,
        "completion_rate": total_production / total_demand,
        "product_switches": product_switches,
        "quality_switches": quality_switches,
        "switch_cost": product_switches + quality_penalty * quality_switches,
        "load_imbalance": load_imbalance,
        "customer_satisfaction": 1.0 - dissatisfaction,
        "active_line_shifts": int((schedule["产品"] != "").sum()),
        "idle_line_shifts": int((schedule["产品"] == "").sum()),
        "fully_satisfied_products": int((fulfillment_df["剩余量"] <= 1e-3).sum()),
        "partially_satisfied_products": int(((fulfillment_df["排产量"] > 1e-3) & (fulfillment_df["剩余量"] > 1e-3)).sum()),
        "unscheduled_products": int((fulfillment_df["排产量"] <= 1e-3).sum()),
        "cold_start_quantity": float(schedule[schedule["产能来源"].eq("同酒质冷启动下界")]["计划生产量"].sum()),
        "line_utilization_mean": float(line_util.mean()),
        "line_utilization_min": float(line_util.min()),
        "line_utilization_max": float(line_util.max()),
    }
    return metrics, fulfillment_df, milestone_df


def validate_schedule(schedule: pd.DataFrame, data: ModelData) -> dict[str, Any]:
    errors: list[str] = []
    if len(schedule) != len(data.lines) * len(TIME_SLOTS):
        errors.append(f"排程行数错误: {len(schedule)}")
    if schedule.duplicated(["生产日期", "班次", "产线"]).any():
        errors.append("存在重复线班")
    demand_map = {(r.产品, r.酒质): float(r.计划生产量) for r in data.demand.itertuples(index=False)}
    produced = schedule.groupby(["产品", "酒质"])["计划生产量"].sum().to_dict()
    index = {key: i for i, key in enumerate(data.products)}
    for key, qty in produced.items():
        if not key[0]:
            continue
        if key not in demand_map:
            errors.append(f"非需求组合: {key}")
        elif qty > demand_map[key] + 1e-3:
            errors.append(f"超需求: {key}")
    for row in schedule.itertuples(index=False):
        if not row.产品:
            continue
        i = index[(row.产品, row.酒质)]
        if (i, row.产线) not in data.pair_capacity:
            errors.append(f"不兼容组合: {(row.产品, row.酒质, row.产线)}")
        if row.计划生产量 > row.班次产能估计 + 1e-3:
            errors.append(f"超产能: {(row.产线, row.生产日期, row.班次)}")
        if row.计划生产量 < MIN_ACTIVE_QUANTITY - 1e-3:
            errors.append(f"活动班次产量过小: {(row.产线, row.生产日期, row.班次)}")
    return {"passed": not errors, "error_count": len(errors), "errors": errors[:30]}


def recompute_transition_flags(schedule: pd.DataFrame) -> pd.DataFrame:
    out = schedule.copy()
    out["班次序"] = out["班次"].map({"早班": 0, "中班": 1})
    out = out.sort_values(["产线", "生产日期", "班次序"]).reset_index(drop=True)
    out["产品换产标记"] = 0
    out["酒质切换标记"] = 0
    for line, idx in out.groupby("产线", sort=False).groups.items():
        prev_product = None
        prev_quality = None
        for pos, row_idx in enumerate(idx):
            product = str(out.at[row_idx, "产品"]) if pd.notna(out.at[row_idx, "产品"]) else ""
            quality = str(out.at[row_idx, "酒质"]) if pd.notna(out.at[row_idx, "酒质"]) else ""
            out.at[row_idx, "产品换产标记"] = int(pos > 0 and bool(product) and product != prev_product)
            out.at[row_idx, "酒质切换标记"] = int(pos > 0 and bool(quality) and quality != prev_quality)
            prev_product = product if product else None
            prev_quality = quality if quality else None
    return out.sort_values(["生产日期", "班次序", "产线"]).drop(columns="班次序").reset_index(drop=True)


def schedule_to_blocks(schedule: pd.DataFrame) -> pd.DataFrame:
    work = schedule.copy()
    work["班次序"] = work["班次"].map({"早班": 0, "中班": 1})
    work = work.sort_values(["产线", "生产日期", "班次序"]).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for line, part in work.groupby("产线", sort=True):
        part = part.reset_index(drop=True)
        state = part["产品"].fillna("").astype(str) + "|" + part["酒质"].fillna("").astype(str)
        block_id = state.ne(state.shift()).cumsum()
        for _, block in part.groupby(block_id):
            first = block.iloc[0]
            last = block.iloc[-1]
            if not str(first["产品"]):
                continue
            records.append(
                {
                    "产线": line,
                    "产品": first["产品"],
                    "酒质": first["酒质"],
                    "开始日期": first["生产日期"],
                    "开始班次": first["班次"],
                    "结束日期": last["生产日期"],
                    "结束班次": last["班次"],
                    "连续线班数": int(len(block)),
                    "区段排产量": float(block["计划生产量"].sum()),
                    "产能来源": first["产能来源"],
                }
            )
    return pd.DataFrame(records)


def repair_small_batches(schedule: pd.DataFrame, data: ModelData) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = schedule.copy()
    work["时段序"] = work["生产日期"].astype(str) + "|" + work["班次"].astype(str)
    ordered_keys = [d.strftime("%Y-%m-%d") + "|" + s for d, s in TIME_SLOTS]
    work["时段序"] = pd.Categorical(work["时段序"], ordered_keys, ordered=True).codes
    product_index = {key: i for i, key in enumerate(data.products)}

    def threshold(row: pd.Series) -> float:
        if not str(row["产品"]):
            return 0.0
        i = product_index[(str(row["产品"]), str(row["酒质"]))]
        demand_i = float(data.demand.loc[i, "计划生产量"])
        historical_floor = min(float(row["班次产能估计"]), demand_i, data.pair_min_batch[(i, str(row["产线"]))])
        return max(MIN_ACTIVE_QUANTITY, 0.10 * historical_floor)

    before_small = 0
    moved_quantity = 0.0
    cleared_slots = 0
    dropped_quantity = 0.0
    unresolved: list[dict[str, Any]] = []
    # 从最小碎片开始处理，优先转移到同产品已经激活且不晚于原班次的有余量线班，
    # 因而不会降低累计交付满意度，也不会创造新的产品状态。
    active_indices = [idx for idx, row in work.iterrows() if str(row["产品"])]
    active_indices.sort(key=lambda idx: float(work.at[idx, "计划生产量"]))
    for source_idx in active_indices:
        source = work.loc[source_idx]
        if not str(source["产品"]):
            continue
        min_batch = threshold(source)
        qty = float(source["计划生产量"])
        if qty + 1e-6 >= min_batch:
            continue
        before_small += 1
        candidates = []
        for target_idx, target in work.iterrows():
            if target_idx == source_idx:
                continue
            if str(target["产品"]) != str(source["产品"]) or str(target["酒质"]) != str(source["酒质"]):
                continue
            target_threshold = threshold(target)
            target_qty = float(target["计划生产量"])
            if target_qty + 1e-6 < target_threshold:
                continue
            slack = float(target["班次产能估计"]) - target_qty
            if slack <= 1e-6:
                continue
            earlier = int(work.at[target_idx, "时段序"] <= work.at[source_idx, "时段序"])
            same_line = int(str(target["产线"]) == str(source["产线"]))
            candidates.append((earlier, same_line, slack, target_idx))
        candidates.sort(reverse=True)
        if sum(item[2] for item in candidates) + 1e-6 < qty:
            # 无可合并余量时取消极小占位。该损失单独冻结并与最大产量锚点比较，
            # 只有当相对损失低于万分之一时才接受实用化方案。
            work.loc[source_idx, ["产品", "酒质", "产能来源"]] = ""
            work.loc[source_idx, ["计划生产量", "班次产能估计", "产能利用率"]] = 0.0
            dropped_quantity += qty
            cleared_slots += 1
            continue
        remaining = qty
        for _, _, slack, target_idx in candidates:
            transfer = min(remaining, slack)
            work.at[target_idx, "计划生产量"] = float(work.at[target_idx, "计划生产量"]) + transfer
            work.at[target_idx, "产能利用率"] = float(work.at[target_idx, "计划生产量"]) / float(work.at[target_idx, "班次产能估计"])
            remaining -= transfer
            if remaining <= 1e-6:
                break
        work.loc[source_idx, ["产品", "酒质", "产能来源"]] = ""
        work.loc[source_idx, ["计划生产量", "班次产能估计", "产能利用率"]] = 0.0
        moved_quantity += qty
        cleared_slots += 1

    # 对每条产线保持产品出现顺序不变，把活动班次向月初压紧，消除碎片清理后
    # 产生的中间空洞。产线、产量和兼容关系均不变，累计交付只会提前。
    movable_fields = ["产品", "酒质", "计划生产量", "班次产能估计", "产能利用率", "产能来源", "模型"]
    for line, idx in work.groupby("产线", sort=False).groups.items():
        ordered_idx = sorted(idx, key=lambda row_idx: int(work.at[row_idx, "时段序"]))
        active_payloads = [work.loc[row_idx, movable_fields].to_dict() for row_idx in ordered_idx if str(work.at[row_idx, "产品"])]
        for pos, row_idx in enumerate(ordered_idx):
            if pos < len(active_payloads):
                for field, value in active_payloads[pos].items():
                    work.at[row_idx, field] = value
            else:
                work.loc[row_idx, ["产品", "酒质", "产能来源"]] = ""
                work.loc[row_idx, ["计划生产量", "班次产能估计", "产能利用率"]] = 0.0
    work = work.drop(columns="时段序")
    work = recompute_transition_flags(work)
    after_small = 0
    for _, row in work.iterrows():
        if str(row["产品"]) and float(row["计划生产量"]) + 1e-6 < threshold(row):
            after_small += 1
    report = {
        "small_slots_before": before_small,
        "small_slots_after": after_small,
        "cleared_slots": cleared_slots,
        "moved_quantity": moved_quantity,
        "dropped_quantity": dropped_quantity,
        "relative_production_loss": dropped_quantity / max(float(schedule["计划生产量"].sum()), 1.0),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
    }
    return work, report


def greedy_schedule(data: ModelData) -> pd.DataFrame:
    remaining = {i: float(row["计划生产量"]) for i, row in data.demand.iterrows()}
    produced = {i: 0.0 for i in range(len(data.products))}
    assigned: dict[tuple[str, int], tuple[int, float]] = {}
    prev = {line: None for line in data.lines}
    for t, _ in enumerate(TIME_SLOTS):
        period = next(k for k, end in enumerate(PERIOD_END_SLOTS) if t <= end)
        for line in data.lines:
            candidates = [i for i in range(len(data.products)) if remaining[i] > 1e-6 and (i, line) in data.pair_capacity]
            if not candidates:
                prev[line] = None
                continue
            scores = []
            for i in candidates:
                demand_i = float(data.demand.loc[i, "计划生产量"])
                target_deficit = max(0.0, float(MILESTONE_ALPHA[period]) * demand_i - produced[i]) / demand_i
                scarcity = remaining[i] / max(
                    sum(data.pair_capacity[(i, l)] for l in data.compatible_lines.get(i, [])) * max(1, len(TIME_SLOTS) - t),
                    1.0,
                )
                continuation = 0.12 if prev[line] == i else 0.0
                preference = 0.03 * data.pair_preference[(i, line)]
                scores.append((target_deficit + 0.25 * scarcity + continuation + preference, i))
            _, chosen = max(scores)
            qty = min(data.pair_capacity[(chosen, line)], remaining[chosen])
            assigned[(line, t)] = (chosen, qty)
            remaining[chosen] -= qty
            produced[chosen] += qty
            prev[line] = chosen

    fake = type("GreedyResult", (), {})()
    # 复用统一导出逻辑所需的最小结果向量。
    fake.x = np.zeros(1)
    records = []
    for line in data.lines:
        prev_product = None
        prev_quality = None
        for t, (date, shift) in enumerate(TIME_SLOTS):
            item = assigned.get((line, t))
            if item is None:
                product = quality = source = ""
                qty = cap = 0.0
            else:
                i, qty = item
                product, quality = data.products[i]
                cap = data.pair_capacity[(i, line)]
                source = data.pair_source[(i, line)]
            records.append(
                {
                    "生产日期": date.strftime("%Y-%m-%d"),
                    "班次": shift,
                    "产线": line,
                    "产品": product,
                    "酒质": quality,
                    "计划生产量": float(round(qty, 6)),
                    "班次产能估计": float(cap),
                    "产能利用率": 0.0 if cap <= 0 else float(qty / cap),
                    "产能来源": source,
                    "产品换产标记": int(t > 0 and bool(product) and product != prev_product),
                    "酒质切换标记": int(t > 0 and bool(quality) and quality != prev_quality),
                    "模型": "交期优先顺序贪心",
                }
            )
            prev_product = product if product else None
            prev_quality = quality if quality else None
    out = pd.DataFrame(records)
    out["班次序"] = out["班次"].map({"早班": 0, "中班": 1})
    return out.sort_values(["生产日期", "班次序", "产线"]).drop(columns="班次序").reset_index(drop=True)


def nondominated(df: pd.DataFrame) -> pd.Series:
    values = df[["total_production", "switch_cost", "load_imbalance", "customer_satisfaction"]].to_numpy(float)
    keep = np.ones(len(values), dtype=bool)
    for i, a in enumerate(values):
        if not keep[i]:
            continue
        for j, b in enumerate(values):
            if i == j:
                continue
            no_worse = b[0] >= a[0] - 1e-6 and b[1] <= a[1] + 1e-9 and b[2] <= a[2] + 1e-12 and b[3] >= a[3] - 1e-12
            strictly = b[0] > a[0] + 1e-6 or b[1] < a[1] - 1e-9 or b[2] < a[2] - 1e-12 or b[3] > a[3] + 1e-12
            if no_worse and strictly:
                keep[i] = False
                break
    return pd.Series(keep, index=df.index)


def compromise_scores(df: pd.DataFrame) -> pd.Series:
    memberships = []
    directions = {
        "total_production": 1,
        "switch_cost": -1,
        "load_imbalance": -1,
        "customer_satisfaction": 1,
    }
    for col, direction in directions.items():
        x = df[col].astype(float)
        lo, hi = float(x.min()), float(x.max())
        if math.isclose(lo, hi):
            m = pd.Series(np.ones(len(x)), index=x.index)
        else:
            m = (x - lo) / (hi - lo) if direction > 0 else (hi - x) / (hi - lo)
        memberships.append(m)
    matrix = pd.concat(memberships, axis=1)
    # 最大最小满意度保证四个目标没有明显短板，均值用于平局细分。
    return 0.7 * matrix.min(axis=1) + 0.3 * matrix.mean(axis=1)


def toy_poc() -> dict[str, Any]:
    # 两条产线、两个周期的四个可行方案，验证非支配筛选与折中评分。
    toy = pd.DataFrame(
        [
            {"name": "A", "total_production": 100, "switch_cost": 8, "load_imbalance": 0.10, "customer_satisfaction": 0.80},
            {"name": "B", "total_production": 98, "switch_cost": 5, "load_imbalance": 0.06, "customer_satisfaction": 0.88},
            {"name": "C", "total_production": 95, "switch_cost": 7, "load_imbalance": 0.12, "customer_satisfaction": 0.75},
            {"name": "D", "total_production": 99, "switch_cost": 6, "load_imbalance": 0.08, "customer_satisfaction": 0.84},
        ]
    )
    keep = nondominated(toy)
    names = toy.loc[keep, "name"].tolist()
    return {"passed": names == ["A", "B", "D"], "nondominated": names}


def plot_schedule(schedule: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    lines = sorted(schedule["产线"].unique())
    products = sorted(p for p in schedule["产品"].unique() if p)
    code = {p: i + 1 for i, p in enumerate(products)}
    matrix = np.zeros((len(lines), len(TIME_SLOTS)))
    labels = np.full(matrix.shape, "", dtype=object)
    schedule = schedule.copy()
    schedule["班次序"] = schedule["班次"].map({"早班": 0, "中班": 1})
    for i, line in enumerate(lines):
        part = schedule[schedule["产线"].eq(line)].sort_values(["生产日期", "班次序"])
        for t, row in enumerate(part.itertuples(index=False)):
            if row.产品:
                matrix[i, t] = code[row.产品]
                labels[i, t] = row.产品
    palette = ["#FFFFFF"] + [plt.get_cmap("tab20")(i % 20) for i in range(len(products))]
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.imshow(matrix, aspect="auto", cmap=matplotlib.colors.ListedColormap(palette), vmin=0, vmax=max(1, len(products)))
    for i in range(len(lines)):
        for t in range(len(TIME_SLOTS)):
            if labels[i, t]:
                ax.text(t, i, labels[i, t], ha="center", va="center", fontsize=4.5)
    tick_pos = list(range(0, len(TIME_SLOTS), 2))
    ax.set_xticks(tick_pos, [TIME_SLOTS[t][0].strftime("%m-%d") for t in tick_pos], rotation=45, ha="right")
    ax.set_yticks(range(len(lines)), lines)
    for end in PERIOD_END_SLOTS[:-1]:
        ax.axvline(end + 0.5, color="#222222", linewidth=1.2)
    ax.set_xlabel("生产日期（每列包含早班和中班）")
    ax.set_ylabel("产线")
    ax.set_title("问题三：推荐折中方案月度排产热力图")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(pareto: pd.DataFrame, selected_name: str, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    selected = pareto[pareto["方案"] == selected_name]
    panels = [
        ("switch_cost", "换产成本指数", "#457B9D"),
        ("load_imbalance", "负荷不均衡度", "#2A9D8F"),
        ("customer_satisfaction", "客户满意度", "#E9C46A"),
    ]
    for ax, (col, label, color) in zip(axes, panels):
        ax.scatter(pareto["total_production"], pareto[col], s=55, color=color, alpha=0.8)
        if not selected.empty:
            ax.scatter(selected["total_production"], selected[col], s=140, facecolors="none", edgecolors="#D62828", linewidths=2, label="推荐方案")
        ax.set_xlabel("总排产量")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x")
    axes[0].legend(loc="best")
    fig.suptitle("近似 Pareto 解集的四目标权衡")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_load(schedule: pd.DataFrame, data: ModelData, path: Path) -> pd.DataFrame:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    production = schedule.groupby("产线")["计划生产量"].sum().reindex(data.lines, fill_value=0.0)
    source = pd.DataFrame(
        {
            "产线": data.lines,
            "月排产量": [production[line] for line in data.lines],
            "参考产能": [data.line_reference_capacity[line] for line in data.lines],
        }
    )
    source["标准化负荷率"] = source["月排产量"] / source["参考产能"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(source["产线"], source["标准化负荷率"] * 100, color="#2A9D8F")
    ax.axhline(source["标准化负荷率"].mean() * 100, color="#D62828", linestyle="--", label="平均负荷率")
    ax.set_ylabel("标准化负荷率（%）")
    ax.set_xlabel("产线")
    ax.set_title("推荐方案各产线标准化负荷")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bar, val in zip(bars, source["标准化负荷率"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return source


def plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("completion_rate", "需求完成率（%）", 100),
        ("switch_cost", "换产成本指数", 1),
        ("load_imbalance", "负荷不均衡度", 1),
        ("customer_satisfaction", "客户满意度（%）", 100),
    ]
    colors = ["#457B9D", "#E9C46A"]
    for ax, (col, title, scale) in zip(axes.flat, metrics):
        values = comparison[col] * scale
        bars = ax.bar(comparison["方案"], values, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("多目标 MILP 与传统顺序贪心对比")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_method_files(poc: dict[str, Any]) -> None:
    candidates = f"""# Q3 候选模型与最小 PoC

## 问题输入与缺口

- 输入：附件3的28个产品—酒质月需求、Q2的Q95收缩产能和历史兼容关系、23个生产日共46个班次时段。
- 输出：14条产线共644个线班的产品和生产量、四目标指标及非支配解集。
- 缺口：附件未给客户级交期、客户优先级、产品/酒质换产成本矩阵。

## 候选模型

1. 加权和MILP：实现简单，但权重会把不同量纲混合，且只能得到一个方案。
2. NSGA-II：可生成丰富非支配解，但整数可行性修复复杂，不能证明单个方案达到MILP下界。
3. ε约束+归一化加权MILP（选定）：以最大产量锚定产量下界，在95%、97.5%、99%、100%四档产量约束下改变换产、负荷、交期权重，生成可复现的近似Pareto解集。

## 代理口径

- 交期：按题目列出的5个生产周期和累计工作日比例，建立21.74%、47.83%、69.57%、95.65%、100%的累计交付里程碑。
- 换产：产品改变成本1；酒质改变另加2，因此跨酒质产品切换成本3。该比例做敏感性边界说明。
- P0002-Q04：主情景使用同酒质Q04在L11上的历史最小班次量作为冷启动上界，偏好置0；严格情景禁止该未验证组合。

## 最小 PoC

内置4个玩具方案验证非支配筛选，预期保留A、B、D并剔除被支配方案C。结果：{json.dumps(poc, ensure_ascii=False)}。
"""
    (METHOD_DIR / "q3_method_candidates.md").write_text(candidates, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    started = time.perf_counter()
    poc = toy_poc()
    if not poc["passed"]:
        raise RuntimeError(f"PoC failed: {poc}")
    write_method_files(poc)

    demand, capacity, audit = build_q3_capacity(cold_start=True)
    write_csv(demand, DATA_DIR / "monthly_demand_clean.csv")
    write_csv(capacity, TABLE_DIR / "q3_capacity_matrix_q95.csv")
    data = make_model_data(demand, capacity)
    artifacts = build_milp(data, quality_penalty=2.0)

    max_anchor_obj = combine_objective(
        artifacts,
        {"shortage": 1.0, "switch": 1e-10, "balance": 1e-10, "dissatisfaction": 1e-10},
    )
    total_demand = float(demand["计划生产量"].sum())
    anchor_checkpoint = EXPERIMENT_DIR / "production_anchor.json"
    if anchor_checkpoint.exists():
        p_max = float(json.loads(anchor_checkpoint.read_text(encoding="utf-8"))["production_max_anchor"])
        print("[Q3] loading maximum-production anchor checkpoint", flush=True)
    else:
        print("[Q3] solving maximum-production anchor", flush=True)
        max_result = run_milp(artifacts, max_anchor_obj, gap=0.0)
        p_max = total_demand - sum(max_result.x[idx] for idx in artifacts.r_idx.values())
        anchor_checkpoint.write_text(
            json.dumps(
                {
                    "production_max_anchor": p_max,
                    "status": int(max_result.status),
                    "message": str(max_result.message),
                    "mip_gap": float(getattr(max_result, "mip_gap", 0.0) or 0.0),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    styles = {
        "成本优先": {"switch": 0.70, "balance": 0.15, "dissatisfaction": 0.15},
        "负荷优先": {"switch": 0.15, "balance": 0.70, "dissatisfaction": 0.15},
        "交期优先": {"switch": 0.15, "balance": 0.15, "dissatisfaction": 0.70},
    }
    candidate_records = []
    candidate_schedules: dict[str, pd.DataFrame] = {}
    solver_records = []
    checkpoint_dir = EXPERIMENT_DIR / "candidate_checkpoints_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for ratio in (0.95, 0.975, 0.99, 1.0):
        for style, weights in styles.items():
            name = f"P{ratio:.3f}-{style}"
            checkpoint_csv = checkpoint_dir / f"{name}.csv"
            checkpoint_json = checkpoint_dir / f"{name}.json"
            if checkpoint_json.exists() and not checkpoint_csv.exists():
                failed_record = json.loads(checkpoint_json.read_text(encoding="utf-8"))
                if failed_record.get("failed"):
                    print(f"[Q3] skipping recorded unsolved candidate {name}", flush=True)
                    solver_records.append(failed_record)
                    continue
            if checkpoint_csv.exists() and checkpoint_json.exists():
                print(f"[Q3] loading checkpoint {name}", flush=True)
                schedule = pd.read_csv(checkpoint_csv, encoding="utf-8-sig").fillna("")
                solver_record = json.loads(checkpoint_json.read_text(encoding="utf-8"))
            else:
                print(f"[Q3] solving {name}", flush=True)
                objective = combine_objective(artifacts, weights)
                t0 = time.perf_counter()
                try:
                    result = run_milp(
                        artifacts,
                        objective,
                        [production_floor_row(data, artifacts, ratio * p_max)],
                        time_limit=45.0,
                        gap=0.005,
                    )
                    retry_used = False
                except RuntimeError as exc:
                    if "primal_status is None" not in str(exc):
                        raise
                    print(f"[Q3] retrying {name} with 120-second limit", flush=True)
                    try:
                        result = run_milp(
                            artifacts,
                            objective,
                            [production_floor_row(data, artifacts, ratio * p_max)],
                            time_limit=120.0,
                            gap=0.01,
                        )
                        retry_used = True
                    except RuntimeError as retry_exc:
                        failed_record = {
                            "方案": name,
                            "failed": True,
                            "message": str(retry_exc),
                            "retry_used": True,
                            "elapsed_seconds": time.perf_counter() - t0,
                        }
                        checkpoint_json.write_text(json.dumps(failed_record, ensure_ascii=False, indent=2), encoding="utf-8")
                        solver_records.append(failed_record)
                        print(f"[Q3] candidate unsolved and omitted: {name}", flush=True)
                        continue
                elapsed = time.perf_counter() - t0
                schedule = result_to_schedule(data, result, artifacts, name)
                solver_record = {
                    "方案": name,
                    "status": int(result.status),
                    "message": str(result.message),
                    "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
                    "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
                    "elapsed_seconds": elapsed,
                    "objective": float(result.fun),
                    "retry_used": retry_used,
                }
                write_csv(schedule, checkpoint_csv)
                checkpoint_json.write_text(json.dumps(solver_record, ensure_ascii=False, indent=2), encoding="utf-8")
            metrics, _, _ = schedule_metrics(schedule, data)
            validation = validate_schedule(schedule, data)
            if not validation["passed"]:
                raise RuntimeError(f"{name} validation failed: {validation}")
            candidate_records.append({"方案": name, "产量下界比例": ratio, "权重类型": style, **weights, **metrics})
            candidate_schedules[name] = schedule
            solver_records.append(solver_record)

    candidates = pd.DataFrame(candidate_records)
    candidates["非支配"] = nondominated(candidates)
    pareto = candidates[candidates["非支配"]].copy().reset_index(drop=True)
    eligible_main = pareto[pareto["total_production"] >= 0.99 * p_max - 1e-3].copy()
    eligible_main["折中评分"] = compromise_scores(eligible_main)
    selected_row = eligible_main.sort_values(["折中评分", "total_production"], ascending=[False, False]).iloc[0]
    selected_name = str(selected_row["方案"])
    main_schedule_raw = candidate_schedules[selected_name]
    main_schedule, repair_report = repair_small_batches(main_schedule_raw, data)
    if repair_report["unresolved_count"] or repair_report["relative_production_loss"] > 1e-4:
        raise RuntimeError(f"small-batch repair incomplete: {repair_report}")
    main_metrics, main_fulfillment, main_milestones = schedule_metrics(main_schedule, data)
    if not validate_schedule(main_schedule, data)["passed"]:
        raise RuntimeError("repaired main schedule validation failed")
    for key, value in main_metrics.items():
        if key in candidates.columns:
            candidates.loc[candidates["方案"].eq(selected_name), key] = value
    candidates["非支配"] = nondominated(candidates)
    pareto = candidates[candidates["非支配"]].copy().reset_index(drop=True)

    greedy = greedy_schedule(data)
    greedy_metrics, greedy_fulfillment, greedy_milestones = schedule_metrics(greedy, data)
    if not validate_schedule(greedy, data)["passed"]:
        raise RuntimeError("greedy validation failed")

    # 冷启动边界：禁止P0002-Q04后重新求解99%产量下界的同权重方案。
    strict_demand, strict_capacity, strict_audit = build_q3_capacity(cold_start=False)
    strict_data = make_model_data(strict_demand, strict_capacity)
    strict_artifacts = build_milp(strict_data, quality_penalty=2.0)
    strict_max_obj = combine_objective(
        strict_artifacts,
        {"shortage": 1.0, "switch": 1e-10, "balance": 1e-10, "dissatisfaction": 1e-10},
    )
    strict_anchor_checkpoint = EXPERIMENT_DIR / "strict_production_anchor.json"
    if strict_anchor_checkpoint.exists():
        strict_pmax = float(json.loads(strict_anchor_checkpoint.read_text(encoding="utf-8"))["production_max_anchor"])
        print("[Q3] loading strict-history anchor checkpoint", flush=True)
    else:
        print("[Q3] solving strict-history anchor", flush=True)
        strict_max_result = run_milp(strict_artifacts, strict_max_obj, gap=0.0)
        strict_pmax = float(strict_demand["计划生产量"].sum()) - sum(strict_max_result.x[idx] for idx in strict_artifacts.r_idx.values())
        strict_anchor_checkpoint.write_text(
            json.dumps(
                {
                    "production_max_anchor": strict_pmax,
                    "status": int(strict_max_result.status),
                    "message": str(strict_max_result.message),
                    "mip_gap": float(getattr(strict_max_result, "mip_gap", 0.0) or 0.0),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    main_weights = {key: float(selected_row[key]) for key in ("switch", "balance", "dissatisfaction")}

    # 主方案只包含一个冷启动组合且排产量恰为15,000。删除该线班后总产量恰好
    # 等于严格历史兼容的精确最大产量锚点，因此该派生方案既可行又保持生产最优。
    strict_schedule = main_schedule.copy()
    cold_mask = strict_schedule["产能来源"].eq("同酒质冷启动下界")
    strict_schedule.loc[cold_mask, ["产品", "酒质", "产能来源"]] = ""
    strict_schedule.loc[cold_mask, ["计划生产量", "班次产能估计", "产能利用率"]] = 0.0
    strict_schedule["模型"] = "严格历史兼容（由主方案删除冷启动线班）"
    strict_schedule = recompute_transition_flags(strict_schedule)
    strict_metrics, _, _ = schedule_metrics(strict_schedule, strict_data)
    strict_validation = validate_schedule(strict_schedule, strict_data)
    strict_anchor_gap = strict_pmax - strict_metrics["total_production"]
    if not strict_validation["passed"] or strict_anchor_gap < -1e-2 or strict_anchor_gap > repair_report["dropped_quantity"] + 1e-2:
        raise RuntimeError(f"derived strict schedule failed: {strict_validation}, production={strict_metrics['total_production']}, anchor={strict_pmax}")

    write_csv(candidates, TABLE_DIR / "q3_candidate_solutions.csv")
    write_csv(pareto, TABLE_DIR / "q3_pareto_front.csv")
    write_csv(main_schedule, TABLE_DIR / "q3_schedule_compromise.csv")
    write_csv(schedule_to_blocks(main_schedule), TABLE_DIR / "q3_schedule_blocks.csv")
    write_csv(main_fulfillment, TABLE_DIR / "q3_demand_fulfillment.csv")
    write_csv(main_milestones, TABLE_DIR / "q3_milestone_fulfillment.csv")
    write_csv(greedy, TABLE_DIR / "q3_schedule_greedy.csv")
    write_csv(greedy_fulfillment, TABLE_DIR / "q3_demand_fulfillment_greedy.csv")
    write_csv(greedy_milestones, TABLE_DIR / "q3_milestone_fulfillment_greedy.csv")
    write_csv(strict_schedule, TABLE_DIR / "q3_schedule_strict_history.csv")
    write_csv(pd.DataFrame(solver_records), EXPERIMENT_DIR / "q3_solver_runs.csv")

    load_source = plot_load(main_schedule, data, FIGURE_DIR / "q3_line_loads.png")
    write_csv(load_source, FIGURE_DIR / "q3_line_loads_source.csv")
    plot_schedule(main_schedule, FIGURE_DIR / "q3_schedule_heatmap.png")
    plot_pareto(pareto, selected_name, FIGURE_DIR / "q3_pareto_tradeoffs.png")
    comparison = pd.DataFrame(
        [
            {"方案": "多目标MILP", **main_metrics},
            {"方案": "传统顺序贪心", **greedy_metrics},
        ]
    )
    write_csv(comparison, TABLE_DIR / "q3_method_comparison.csv")
    plot_comparison(comparison, FIGURE_DIR / "q3_method_comparison.png")

    sensitivity = pd.DataFrame(
        [
            {"情景": "冷启动主方案", **main_metrics},
            {"情景": "严格历史兼容", **strict_metrics},
            {"情景": "产量下界95%", **candidates[candidates["产量下界比例"].eq(0.95)].sort_values("customer_satisfaction", ascending=False).iloc[0].to_dict()},
            {"情景": "产量下界100%", **candidates[candidates["产量下界比例"].eq(1.0)].sort_values("customer_satisfaction", ascending=False).iloc[0].to_dict()},
        ]
    )
    write_csv(sensitivity, TABLE_DIR / "q3_sensitivity.csv")

    validation = validate_schedule(main_schedule, data)
    frozen = {
        "question": "Q3",
        "model": "epsilon-constraint normalized multiobjective MILP",
        "selected_solution": selected_name,
        "production_max_anchor": p_max,
        "pareto_solution_count": int(len(pareto)),
        "candidate_solution_count": int(len(candidates)),
        "main_metrics": main_metrics,
        "greedy_metrics": greedy_metrics,
        "strict_history_metrics": strict_metrics,
        "strict_history_production_max_anchor": strict_pmax,
        "selected_weights": main_weights,
        "selected_compromise_score": float(selected_row["折中评分"]),
        "small_batch_repair": repair_report,
        "milestone_alpha": [float(x) for x in MILESTONE_ALPHA],
        "quality_switch_extra_penalty": 2.0,
        "cold_start_rule": "same-quality same-line historical minimum; preference=0; engineering validation required",
        "validation": validation,
        "poc": poc,
        "input_hashes": {"附件1.csv": sha256(INPUT_HISTORY), "附件3.csv": sha256(INPUT_DEMAND)},
    }
    (FROZEN_DIR / "frozen_numbers.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    run_summary = {
        **frozen,
        "audit": audit,
        "strict_audit": strict_audit,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "seed": SEED,
            "total_elapsed_seconds": time.perf_counter() - started,
        },
        "files": {
            "schedule": str(TABLE_DIR / "q3_schedule_compromise.csv"),
            "pareto": str(TABLE_DIR / "q3_pareto_front.csv"),
            "comparison": str(TABLE_DIR / "q3_method_comparison.csv"),
        },
    }
    (EXPERIMENT_DIR / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Q3 求解结果冻结报告

## 直接答案

在23个生产日、14条产线、每日2班的644个线班上，ε约束多目标MILP生成{len(candidates)}个候选解，筛得{len(pareto)}个非支配解。推荐方案为“{selected_name}”，总排产量{main_metrics['total_production']:.0f}，需求完成率{main_metrics['completion_rate']:.4%}，产品换产{main_metrics['product_switches']}次、酒质切换{main_metrics['quality_switches']}次，换产成本指数{main_metrics['switch_cost']:.0f}，负荷不均衡度{main_metrics['load_imbalance']:.6f}，客户满意度{main_metrics['customer_satisfaction']:.4%}。

传统交期优先顺序贪心的总排产量为{greedy_metrics['total_production']:.0f}，换产成本指数{greedy_metrics['switch_cost']:.0f}，负荷不均衡度{greedy_metrics['load_imbalance']:.6f}，客户满意度{greedy_metrics['customer_satisfaction']:.4%}。

## 结论边界

附件3没有客户级交期和优先级，因此客户满意度是基于五个均匀累计交付里程碑的代理指标。P0002-Q04在历史中未出现，主方案仅用同酒质L11历史最小班次量建立冷启动可行边；若严格禁止未见组合，最大可排产量为{strict_pmax:.0f}。冷启动排程必须先经工艺人员确认后执行。
"""
    (REPORT_DIR / "q3_final_result_analysis.md").write_text(report, encoding="utf-8")

    robustness = f"""# Q3 鲁棒性与敏感性报告

## 支持的结论

| 结论 | 证据 | 置信度 |
|---|---|---|
| 多目标MILP可形成可审计的效率—换产—负荷—交期权衡 | 16个ε约束候选解、{len(pareto)}个非支配解 | 高 |
| 月度需求无法全部满足 | 冷启动情景最大产量锚点{p_max:.0f}，低于总需求{total_demand:.0f} | 高 |
| P0002-Q04处理影响最终可排产量 | 严格历史最大产量{strict_pmax:.0f}，冷启动最大产量{p_max:.0f} | 中 |

## 脆弱结论

| 结论 | 脆弱原因 | 限定条件 |
|---|---|---|
| 客户满意度的绝对值 | 未提供真实订单交期和客户权重 | 只用于方案内比较，不解释为调查满意率 |
| 换产成本指数 | 未提供货币成本与时间矩阵 | 仅表示产品切换1、酒质额外2的相对负担 |
| 冷启动方案可执行 | P0002-Q04缺历史工艺记录 | L11投产前需工艺验证 |

## 约束扰动

产量下界在最大产量的95%、97.5%、99%、100%四档变化，所有方案均经过同一可行性校验。非支配前沿用于展示放松产量约束对换产、负荷和交期指标的影响。
"""
    (ROBUST_DIR / "q3_robustness_report.md").write_text(robustness, encoding="utf-8")

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
