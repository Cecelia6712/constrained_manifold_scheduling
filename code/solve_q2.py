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
INPUT_HISTORY = ROOT / "附件1.csv"
INPUT_DEMAND = ROOT / "附件2.csv"
DATA_DIR = ROOT / "data" / "processed"
METHOD_DIR = ROOT / "methods" / "Q2"
RESULT_DIR = ROOT / "results" / "Q2"
ROUND_DIR = RESULT_DIR / "experiments" / "round1"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"
REPORT_DIR = RESULT_DIR / "reports"
ROBUST_DIR = ROOT / "robustness" / "Q2"
FROZEN_DIR = ROOT / "frozen" / "Q2"

SEED = 42
DATES = pd.date_range("2025-03-03", "2025-03-07", freq="D")
SHIFTS = ["早班", "中班"]
TIME_SLOTS = [(d, s) for d in DATES for s in SHIFTS]
MIN_ACTIVE_QUANTITY = 1.0


def ensure_dirs() -> None:
    for path in (DATA_DIR, METHOD_DIR, ROUND_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR, ROBUST_DIR, FROZEN_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_history() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(INPUT_HISTORY, encoding="utf-8-sig")
    raw_rows = len(raw)
    duplicate_beyond_first = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy().reset_index(drop=True)
    df["生产日期"] = pd.to_datetime(df["生产日期"], errors="coerce")
    df["班次原始"] = df["班次"]
    df["班次"] = df["班次"].where(df["班次"].isna(), df["班次"].astype(str).str.strip())
    df.loc[~df["班次"].isin(SHIFTS), "班次"] = np.nan
    df["班次是否推断"] = 0

    inferred = 0
    for (_, _), idx in df.groupby(["生产日期", "产线"], dropna=False).groups.items():
        idx = list(idx)
        missing = [j for j in idx if pd.isna(df.at[j, "班次"])]
        known = {df.at[j, "班次"] for j in idx if df.at[j, "班次"] in SHIFTS}
        if len(missing) == 1 and len(known) == 1:
            df.at[missing[0], "班次"] = "中班" if "早班" in known else "早班"
            df.at[missing[0], "班次是否推断"] = 1
            inferred += 1

    known_mask = df["班次"].isin(SHIFTS)
    slot_cols = ["生产日期", "产线", "班次"]
    sizes = df.loc[known_mask].groupby(slot_cols)["产品"].transform("size")
    df["班次冲突标记"] = 0
    df.loc[known_mask, "班次冲突标记"] = (sizes > 1).astype(int).to_numpy()

    group_cols = ["产品", "酒质", "产线"]
    med = df.groupby(group_cols)["计划生产量"].transform("median")
    q1 = df.groupby(group_cols)["计划生产量"].transform(lambda x: x.quantile(0.25))
    q3 = df.groupby(group_cols)["计划生产量"].transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    df["组内IQR异常标记"] = ((df["计划生产量"] < q1 - 1.5 * iqr) | (df["计划生产量"] > q3 + 1.5 * iqr)).astype(int)
    df["组内极低量标记"] = (df["计划生产量"] < 0.1 * med).astype(int)
    df["完整单产品班次标记"] = (df["班次"].isin(SHIFTS) & (df["班次冲突标记"] == 0)).astype(int)

    eligible = df[df["完整单产品班次标记"] == 1].copy()
    report = {
        "raw_rows": raw_rows,
        "exact_duplicates_removed": duplicate_beyond_first,
        "deduplicated_rows": int(len(df)),
        "shift_values_normalized": int((df["班次原始"].fillna("").astype(str) != df["班次"].fillna("").astype(str)).sum()),
        "missing_shifts_after_dedup_before_inference": int(df["班次原始"].isna().sum()),
        "uniquely_inferred_shifts": inferred,
        "missing_shifts_after_inference": int(df["班次"].isna().sum()),
        "known_shift_conflict_rows": int(df["班次冲突标记"].sum()),
        "known_shift_conflict_slots": int(
            df.loc[df["班次冲突标记"] == 1, slot_cols].drop_duplicates().shape[0]
        ),
        "eligible_capacity_rows": int(len(eligible)),
        "group_iqr_flags": int(df["组内IQR异常标记"].sum()),
        "group_extreme_low_flags": int(df["组内极低量标记"].sum()),
    }
    return df, eligible, report


def minmax(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def bootstrap_quantile(values: np.ndarray, quantile: float, rng: np.random.Generator, n_boot: int = 500) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    estimates = np.quantile(samples, quantile, axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_capacity_table(
    history: pd.DataFrame,
    eligible: pd.DataFrame,
    quantile: float = 0.95,
    scale: float = 1.0,
    shrink_strength: float = 5.0,
    raw_max_mode: bool = False,
    with_bootstrap: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    keys = history[["产品", "酒质", "产线"]].drop_duplicates().sort_values(["产品", "酒质", "产线"])
    parent_groups = eligible.groupby(["酒质", "产线"])["计划生产量"]
    eligible_groups = eligible.groupby(["产品", "酒质", "产线"])["计划生产量"]
    all_groups = history.groupby(["产品", "酒质", "产线"])["计划生产量"]
    records: list[dict[str, Any]] = []

    for row in keys.itertuples(index=False):
        key = (row.产品, row.酒质, row.产线)
        all_values = all_groups.get_group(key).to_numpy(dtype=float)
        if key in eligible_groups.groups:
            values = eligible_groups.get_group(key).to_numpy(dtype=float)
            source = "完整单产品班次"
        else:
            values = all_values
            source = "冲突或缺班次回退"
        n = len(values)
        group_q = float(np.max(values) if raw_max_mode else np.quantile(values, quantile))
        group_max = float(np.max(values))
        parent_key = (row.酒质, row.产线)
        if parent_key in parent_groups.groups:
            parent_values = parent_groups.get_group(parent_key).to_numpy(dtype=float)
            parent_q = float(np.max(parent_values) if raw_max_mode else np.quantile(parent_values, quantile))
        else:
            parent_q = group_q
        if raw_max_mode:
            capacity = group_max
        else:
            weight = n / (n + shrink_strength)
            capacity = min(group_max, weight * group_q + (1.0 - weight) * parent_q)
        capacity = max(MIN_ACTIVE_QUANTITY, math.floor(capacity * scale))

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_cv = 0.0 if median <= 0 else 1.4826 * mad / median
        load_factor = float(np.mean(values) / max(capacity / scale, 1.0))
        ci_low = ci_high = None
        if with_bootstrap:
            ci_low, ci_high = bootstrap_quantile(values, quantile, rng)
        records.append(
            {
                "产品": row.产品,
                "酒质": row.酒质,
                "产线": row.产线,
                "样本数": n,
                "全部历史样本数": len(all_values),
                "产能样本来源": source,
                "组内分位产能": group_q,
                "父组分位产能": parent_q,
                "历史最大值": group_max,
                "估计班次产能": float(capacity),
                "稳健变异系数": robust_cv,
                "历史负载率": load_factor,
                "Bootstrap下限": ci_low,
                "Bootstrap上限": ci_high,
            }
        )

    cap = pd.DataFrame(records)
    all_counts = history.groupby(["产品", "酒质", "产线"]).size().rename("历史分配次数").reset_index()
    cap = cap.merge(all_counts, on=["产品", "酒质", "产线"], how="left")
    cap["历史分配份额"] = cap["历史分配次数"] / cap.groupby(["产品", "酒质"])["历史分配次数"].transform("sum")
    cap["产能归一值"] = cap.groupby(["产品", "酒质"])["估计班次产能"].transform(minmax)
    cap["稳定性归一值"] = cap.groupby(["产品", "酒质"])["稳健变异系数"].transform(lambda x: minmax(1.0 / (1.0 + x)))
    cap["负载率归一值"] = cap.groupby(["产品", "酒质"])["历史负载率"].transform(minmax)
    cap["分配份额归一值"] = cap.groupby(["产品", "酒质"])["历史分配份额"].transform(minmax)
    cap["偏好得分"] = (
        0.4 * cap["产能归一值"]
        + 0.2 * cap["稳定性归一值"]
        + 0.2 * cap["负载率归一值"]
        + 0.2 * cap["分配份额归一值"]
    )
    cap["产线优先级"] = cap.groupby(["产品", "酒质"])["偏好得分"].rank(method="dense", ascending=False).astype(int)
    return cap.sort_values(["产品", "酒质", "产线"]).reset_index(drop=True)


@dataclass
class ModelData:
    demand: pd.DataFrame
    capacity: pd.DataFrame
    products: list[tuple[str, str]]
    lines: list[str]
    pair_capacity: dict[tuple[int, str], float]
    pair_preference: dict[tuple[int, str], float]
    compatible_lines: dict[int, list[str]]


def make_model_data(demand: pd.DataFrame, capacity: pd.DataFrame) -> ModelData:
    products = list(demand[["产品", "酒质"]].itertuples(index=False, name=None))
    index = {key: i for i, key in enumerate(products)}
    use = capacity.merge(demand[["产品", "酒质"]], on=["产品", "酒质"], how="inner")
    pair_capacity: dict[tuple[int, str], float] = {}
    pair_preference: dict[tuple[int, str], float] = {}
    compatible_lines: dict[int, list[str]] = defaultdict(list)
    for row in use.itertuples(index=False):
        i = index[(row.产品, row.酒质)]
        pair_capacity[(i, row.产线)] = float(row.估计班次产能)
        pair_preference[(i, row.产线)] = float(row.偏好得分)
        compatible_lines[i].append(row.产线)
    lines = sorted(capacity["产线"].unique().tolist())
    return ModelData(demand, capacity, products, lines, pair_capacity, pair_preference, dict(compatible_lines))


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
        for r, coeff in enumerate(rows):
            for c, value in coeff.items():
                if value != 0:
                    rr.append(r)
                    cc.append(c)
                    vv.append(value)
        A = coo_array((vv, (rr, cc)), shape=(len(rows), len(self.lb))).tocsr()
        return LinearConstraint(A, np.asarray(lbs), np.asarray(ubs))


@dataclass
class SolverArtifacts:
    builder: LinearModelBuilder
    x_idx: dict[tuple[int, str, int], int]
    q_idx: dict[tuple[int, str, int], int]
    sp_idx: dict[tuple[int, str, int], int]
    sq_idx: dict[tuple[str, str, int], int]
    r_idx: dict[int, int]


def build_milp(data: ModelData) -> SolverArtifacts:
    b = LinearModelBuilder()
    x_idx: dict[tuple[int, str, int], int] = {}
    q_idx: dict[tuple[int, str, int], int] = {}
    sp_idx: dict[tuple[int, str, int], int] = {}
    sq_idx: dict[tuple[str, str, int], int] = {}
    r_idx: dict[int, int] = {}
    n_t = len(TIME_SLOTS)

    for (i, line), cap in sorted(data.pair_capacity.items()):
        for t in range(n_t):
            x_idx[(i, line, t)] = b.var(f"x[{i},{line},{t}]", 0, 1, integer=True)
            q_idx[(i, line, t)] = b.var(f"q[{i},{line},{t}]", 0, cap, integer=False)
            if t > 0:
                sp_idx[(i, line, t)] = b.var(f"sp[{i},{line},{t}]", 0, 1, integer=True)

    qualities_by_line: dict[str, set[str]] = defaultdict(set)
    for (i, line) in data.pair_capacity:
        qualities_by_line[line].add(data.products[i][1])
    for line, qualities in qualities_by_line.items():
        for quality in sorted(qualities):
            for t in range(1, n_t):
                sq_idx[(quality, line, t)] = b.var(f"sq[{quality},{line},{t}]", 0, 1, integer=True)

    for i, demand_row in data.demand.iterrows():
        r_idx[i] = b.var(f"r[{i}]", 0, float(demand_row["计划生产量"]), integer=False)

    for line in data.lines:
        for t in range(n_t):
            coeff = {idx: 1.0 for (i2, line2, t2), idx in x_idx.items() if line2 == line and t2 == t}
            if coeff:
                b.row(coeff, ub=1.0)

    for key, q_var in q_idx.items():
        i, line, t = key
        x_var = x_idx[key]
        cap = data.pair_capacity[(i, line)]
        b.row({q_var: 1.0, x_var: -cap}, ub=0.0)
        b.row({x_var: MIN_ACTIVE_QUANTITY, q_var: -1.0}, ub=0.0)

    for i, demand_row in data.demand.iterrows():
        coeff = {q_var: 1.0 for (i2, _, _), q_var in q_idx.items() if i2 == i}
        coeff[r_idx[i]] = 1.0
        b.row(coeff, lb=float(demand_row["计划生产量"]), ub=float(demand_row["计划生产量"]))

    for (i, line, t), sp_var in sp_idx.items():
        b.row(
            {
                x_idx[(i, line, t)]: 1.0,
                x_idx[(i, line, t - 1)]: -1.0,
                sp_var: -1.0,
            },
            ub=0.0,
        )

    for (quality, line, t), sq_var in sq_idx.items():
        coeff: dict[int, float] = {sq_var: -1.0}
        for i, (_, q) in enumerate(data.products):
            if q == quality and (i, line) in data.pair_capacity:
                coeff[x_idx[(i, line, t)]] = coeff.get(x_idx[(i, line, t)], 0.0) + 1.0
                coeff[x_idx[(i, line, t - 1)]] = coeff.get(x_idx[(i, line, t - 1)], 0.0) - 1.0
        b.row(coeff, ub=0.0)

    return SolverArtifacts(b, x_idx, q_idx, sp_idx, sq_idx, r_idx)


def run_milp(
    artifacts: SolverArtifacts,
    objective: dict[int, float],
    extra_rows: Iterable[tuple[dict[int, float], float, float]] = (),
    time_limit: float = 120.0,
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
        options={"disp": False, "time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    return result


def solve_lexicographic(
    data: ModelData,
    time_limit: float = 120.0,
    shortage_mode: str = "total",
) -> tuple[Any, dict[str, Any], SolverArtifacts]:
    artifacts = build_milp(data)
    b = artifacts.builder
    start = time.perf_counter()

    if shortage_mode == "total":
        shortage_obj = {idx: 1.0 for idx in artifacts.r_idx.values()}
    elif shortage_mode == "normalized":
        shortage_obj = {
            idx: 1.0 / float(data.demand.loc[i, "计划生产量"])
            for i, idx in artifacts.r_idx.items()
        }
    else:
        raise ValueError(f"unknown shortage_mode={shortage_mode}")
    stage1 = run_milp(artifacts, shortage_obj, time_limit=time_limit)
    stage1_opt = float(sum(weight * stage1.x[idx] for idx, weight in shortage_obj.items()))
    rows: list[tuple[dict[int, float], float, float]] = [
        (shortage_obj, -np.inf, stage1_opt + (1e-3 if shortage_mode == "total" else 1e-8))
    ]

    switch_obj = {idx: 1.0 for idx in artifacts.sp_idx.values()}
    stage2 = run_milp(artifacts, switch_obj, rows, time_limit=time_limit)
    product_switch_opt = int(round(sum(stage2.x[idx] for idx in artifacts.sp_idx.values())))
    rows.append((switch_obj, -np.inf, product_switch_opt + 1e-6))

    quality_obj = {idx: 1.0 for idx in artifacts.sq_idx.values()}
    stage3 = run_milp(artifacts, quality_obj, rows, time_limit=time_limit)
    quality_switch_opt = int(round(sum(stage3.x[idx] for idx in artifacts.sq_idx.values())))
    rows.append((quality_obj, -np.inf, quality_switch_opt + 1e-6))

    preference_obj: dict[int, float] = {}
    for (i, line, t), idx in artifacts.q_idx.items():
        preference_obj[idx] = -data.pair_preference[(i, line)]
    stage4 = run_milp(artifacts, preference_obj, rows, time_limit=time_limit)
    preference_opt = float(-sum(preference_obj[idx] * stage4.x[idx] for idx in preference_obj))
    preference_tolerance = max(1e-2, abs(float(stage4.fun)) * 1e-6)
    rows.append((preference_obj, -np.inf, float(stage4.fun) + preference_tolerance))

    active_obj = {idx: 1.0 for idx in artifacts.x_idx.values()}
    stage5 = run_milp(artifacts, active_obj, rows, time_limit=time_limit)
    elapsed = time.perf_counter() - start
    shortage_total = float(sum(stage5.x[idx] for idx in artifacts.r_idx.values()))
    stage_summary = {
        "shortage_mode": shortage_mode,
        "stage1_objective_optimum": stage1_opt,
        "shortage_optimum": shortage_total,
        "production_optimum": float(data.demand["计划生产量"].sum() - shortage_total),
        "product_switch_optimum": product_switch_opt,
        "quality_switch_optimum": quality_switch_opt,
        "preference_quantity_score": preference_opt,
        "active_assignments": int(round(sum(stage5.x[idx] for idx in artifacts.x_idx.values()))),
        "elapsed_seconds": elapsed,
        "mip_gap": float(getattr(stage5, "mip_gap", 0.0) or 0.0),
        "mip_node_count": int(getattr(stage5, "mip_node_count", 0) or 0),
        "solver_message": stage5.message,
    }
    return stage5, stage_summary, artifacts


def result_to_schedule(data: ModelData, result: Any, artifacts: SolverArtifacts, model_name: str) -> pd.DataFrame:
    assigned: dict[tuple[str, int], tuple[int, float]] = {}
    for key, x_var in artifacts.x_idx.items():
        i, line, t = key
        if result.x[x_var] > 0.5:
            qty = max(0.0, float(result.x[artifacts.q_idx[key]]))
            assigned[(line, t)] = (i, qty)

    records: list[dict[str, Any]] = []
    for line in data.lines:
        prev_product = None
        prev_quality = None
        for t, (date, shift) in enumerate(TIME_SLOTS):
            item = assigned.get((line, t))
            if item is None:
                product = quality = ""
                qty = cap = pref = 0.0
            else:
                i, qty = item
                product, quality = data.products[i]
                cap = data.pair_capacity[(i, line)]
                pref = data.pair_preference[(i, line)]
            product_switch = int(t > 0 and bool(product) and product != prev_product)
            quality_switch = int(t > 0 and bool(quality) and quality != prev_quality)
            records.append(
                {
                    "生产日期": date.strftime("%Y-%m-%d"),
                    "班次": shift,
                    "产线": line,
                    "产品": product,
                    "酒质": quality,
                    "计划生产量": float(round(qty)),
                    "班次产能估计": round(cap, 3),
                    "产能利用率": 0.0 if cap <= 0 else round(qty / cap, 6),
                    "产线偏好得分": round(pref, 6),
                    "产品换产标记": product_switch,
                    "酒质切换标记": quality_switch,
                    "模型": model_name,
                }
            )
            prev_product = product if product else None
            prev_quality = quality if quality else None
    return pd.DataFrame(records).sort_values(["生产日期", "班次", "产线"], key=lambda s: s.map({"早班": 0, "中班": 1}) if s.name == "班次" else s).reset_index(drop=True)


def greedy_schedule(data: ModelData) -> pd.DataFrame:
    remaining = {i: float(row["计划生产量"]) for i, row in data.demand.iterrows()}
    assigned: dict[tuple[str, int], tuple[int, float]] = {}
    prev_by_line: dict[str, int | None] = {line: None for line in data.lines}
    n_t = len(TIME_SLOTS)
    for t in range(n_t):
        for line in data.lines:
            candidates = [i for i in range(len(data.products)) if (i, line) in data.pair_capacity and remaining[i] > 1e-9]
            if not candidates:
                prev_by_line[line] = None
                continue
            scored = []
            for i in candidates:
                cap = data.pair_capacity[(i, line)]
                future_capacity = 0.0
                remaining_slots = n_t - t
                for compatible_line in data.compatible_lines.get(i, []):
                    future_capacity += data.pair_capacity[(i, compatible_line)] * remaining_slots
                scarcity = remaining[i] / max(future_capacity, 1.0)
                continuation = 0.15 if prev_by_line[line] == i else 0.0
                scored.append((scarcity + continuation, min(cap, remaining[i]), data.pair_preference[(i, line)], i))
            _, _, _, chosen = max(scored)
            qty = min(data.pair_capacity[(chosen, line)], remaining[chosen])
            assigned[(line, t)] = (chosen, qty)
            remaining[chosen] -= qty
            prev_by_line[line] = chosen

    records: list[dict[str, Any]] = []
    for line in data.lines:
        prev_product = None
        prev_quality = None
        for t, (date, shift) in enumerate(TIME_SLOTS):
            item = assigned.get((line, t))
            if item is None:
                product = quality = ""
                qty = cap = pref = 0.0
            else:
                i, qty = item
                product, quality = data.products[i]
                cap = data.pair_capacity[(i, line)]
                pref = data.pair_preference[(i, line)]
            records.append(
                {
                    "生产日期": date.strftime("%Y-%m-%d"),
                    "班次": shift,
                    "产线": line,
                    "产品": product,
                    "酒质": quality,
                    "计划生产量": round(qty, 3),
                    "班次产能估计": round(cap, 3),
                    "产能利用率": 0.0 if cap <= 0 else round(qty / cap, 6),
                    "产线偏好得分": round(pref, 6),
                    "产品换产标记": int(t > 0 and bool(product) and product != prev_product),
                    "酒质切换标记": int(t > 0 and bool(quality) and quality != prev_quality),
                    "模型": "瓶颈优先贪心",
                }
            )
            prev_product = product if product else None
            prev_quality = quality if quality else None
    return pd.DataFrame(records).sort_values(["生产日期", "班次", "产线"], key=lambda s: s.map({"早班": 0, "中班": 1}) if s.name == "班次" else s).reset_index(drop=True)


def schedule_metrics(schedule: pd.DataFrame, demand: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    produced = schedule.groupby(["产品", "酒质"], dropna=False)["计划生产量"].sum()
    rows = []
    for row in demand.itertuples(index=False):
        qty = float(produced.get((row.产品, row.酒质), 0.0))
        rows.append(
            {
                "产品": row.产品,
                "酒质": row.酒质,
                "需求量": float(row.计划生产量),
                "排产量": qty,
                "剩余量": max(0.0, float(row.计划生产量) - qty),
                "完成率": qty / float(row.计划生产量),
            }
        )
    fulfillment = pd.DataFrame(rows)
    total_demand = float(fulfillment["需求量"].sum())
    total_production = float(fulfillment["排产量"].sum())
    metrics = {
        "total_demand": total_demand,
        "total_production": total_production,
        "total_shortage": float(fulfillment["剩余量"].sum()),
        "completion_rate": total_production / total_demand,
        "product_switches": int(schedule["产品换产标记"].sum()),
        "quality_switches": int(schedule["酒质切换标记"].sum()),
        "active_line_shifts": int((schedule["产品"] != "").sum()),
        "idle_line_shifts": int((schedule["产品"] == "").sum()),
        "fully_satisfied_products": int((fulfillment["剩余量"] <= 1e-3).sum()),
        "partially_satisfied_products": int(((fulfillment["排产量"] > 1e-3) & (fulfillment["剩余量"] > 1e-3)).sum()),
        "unscheduled_products": int((fulfillment["排产量"] <= 1e-3).sum()),
        "mean_assigned_preference": float(
            schedule.loc[schedule["产品"] != "", "产线偏好得分"].mean()
        ),
    }
    return metrics, fulfillment


def validate_schedule(schedule: pd.DataFrame, demand: pd.DataFrame, data: ModelData) -> dict[str, Any]:
    errors: list[str] = []
    if schedule.duplicated(["生产日期", "班次", "产线"]).any():
        errors.append("存在重复班次槽")
    demand_map = {(r.产品, r.酒质): float(r.计划生产量) for r in demand.itertuples(index=False)}
    produced = schedule.groupby(["产品", "酒质"])["计划生产量"].sum().to_dict()
    for key, qty in produced.items():
        if not key[0]:
            continue
        if key not in demand_map:
            errors.append(f"排入非需求组合 {key}")
        elif qty > demand_map[key] + 1e-3:
            errors.append(f"超需求 {key}: {qty}>{demand_map[key]}")
    product_index = {key: i for i, key in enumerate(data.products)}
    for row in schedule.itertuples(index=False):
        if not row.产品:
            continue
        i = product_index[(row.产品, row.酒质)]
        if (i, row.产线) not in data.pair_capacity:
            errors.append(f"不兼容分配 {(row.产品, row.酒质, row.产线)}")
        if row.计划生产量 > row.班次产能估计 + 1e-3:
            errors.append(f"超产能 {(row.产品, row.产线, row.生产日期, row.班次)}")
        if row.计划生产量 < MIN_ACTIVE_QUANTITY - 1e-3:
            errors.append(f"活动槽数量过小 {(row.产品, row.产线)}")
    return {"passed": not errors, "error_count": len(errors), "errors": errors[:20]}


def solve_scenario(demand: pd.DataFrame, capacity: pd.DataFrame, name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    data = make_model_data(demand, capacity)
    result, stages, artifacts = solve_lexicographic(data)
    schedule = result_to_schedule(data, result, artifacts, name)
    metrics, _ = schedule_metrics(schedule, demand)
    validation = validate_schedule(schedule, demand, data)
    if not validation["passed"]:
        raise RuntimeError(f"scenario validation failed {name}: {validation['errors']}")
    return {"scenario": name, **metrics, **{f"stage_{k}": v for k, v in stages.items()}}, schedule


def toy_poc() -> dict[str, Any]:
    # 两产品、单产线、三班次：穷举验证最大产量与最小换产的字典序结果。
    capacities = {"A": 10, "B": 7}
    demand = {"A": 12, "B": 12}
    best = None
    for states in __import__("itertools").product(["", "A", "B"], repeat=3):
        remaining = dict(demand)
        produced = 0
        switches = 0
        prev = None
        for t, state in enumerate(states):
            if state:
                q = min(capacities[state], remaining[state])
                if q <= 0:
                    break
                remaining[state] -= q
                produced += q
                if t > 0 and state != prev:
                    switches += 1
            prev = state or None
        else:
            candidate = (produced, -switches, states)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return {"passed": best is not None and best[0] == 22 and -best[1] == 1, "best_production": best[0], "best_switches": -best[1], "states": best[2]}


def plot_schedule_heatmap(schedule: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    schedule = schedule.copy()
    schedule[["产品", "酒质"]] = schedule[["产品", "酒质"]].fillna("")
    lines = sorted(schedule["产线"].unique())
    slot_labels = [f"{d.strftime('%m-%d')}\n{s}" for d, s in TIME_SLOTS]
    products = sorted(p for p in schedule["产品"].unique() if isinstance(p, str) and p)
    code = {p: i + 1 for i, p in enumerate(products)}
    matrix = np.zeros((len(lines), len(TIME_SLOTS)))
    labels = np.full(matrix.shape, "", dtype=object)
    for i, line in enumerate(lines):
        line_df = schedule[schedule["产线"] == line].sort_values(["生产日期", "班次"], key=lambda s: s.map({"早班": 0, "中班": 1}) if s.name == "班次" else s)
        for t, row in enumerate(line_df.itertuples(index=False)):
            if row.产品:
                matrix[i, t] = code[row.产品]
                labels[i, t] = row.产品
    palette = ["#FFFFFF"] + [plt.get_cmap("tab20")(i % 20) for i in range(len(products))]
    cmap = matplotlib.colors.ListedColormap(palette)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, len(products)))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if labels[i, j]:
                ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=7, color="black")
    ax.set_xticks(range(len(slot_labels)), slot_labels)
    ax.set_yticks(range(len(lines)), lines)
    ax.set_xlabel("日期与班次")
    ax.set_ylabel("产线")
    ax.set_title("问题二：字典序 MILP 周排产方案（单元格为产品编号）")
    ax.set_xticks(np.arange(-0.5, len(slot_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(lines), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fulfillment(fulfillment: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    df = fulfillment.sort_values("完成率", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = np.where(df["完成率"] >= 0.999, "#2A9D8F", np.where(df["完成率"] > 0, "#E9C46A", "#E76F51"))
    bars = ax.barh(df["产品"], df["完成率"] * 100, color=colors)
    ax.axvline(100, color="#264653", linestyle="--", linewidth=1)
    ax.set_xlim(0, max(105, float(df["完成率"].max() * 105)))
    ax.set_xlabel("需求完成率（%）")
    ax.set_ylabel("产品")
    ax.set_title("问题二：各产品需求完成率")
    for bar, value in zip(bars, df["完成率"] * 100):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    scale_df = sensitivity[sensitivity["scenario"].str.startswith("scale_")].copy()
    scale_df["scale"] = scale_df["scenario"].str.replace("scale_", "", regex=False).astype(float)
    scale_df = scale_df.sort_values("scale")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(scale_df["scale"] * 100, scale_df["total_production"], marker="o", color="#1D3557")
    axes[0].set_xlabel("班次产能缩放（基准=100%）")
    axes[0].set_ylabel("总排产量")
    axes[0].set_title("产能扰动对总排产量的影响")
    axes[0].grid(alpha=0.25)
    quantile_df = sensitivity[sensitivity["scenario"].isin(["q90_shrunk", "q95_shrunk", "raw_max"])].copy()
    labels = {"q90_shrunk": "稳健Q90", "q95_shrunk": "稳健Q95", "raw_max": "历史最大值"}
    axes[1].bar([labels[x] for x in quantile_df["scenario"]], quantile_df["total_production"], color=["#457B9D", "#2A9D8F", "#E9C46A"])
    axes[1].set_ylabel("总排产量")
    axes[1].set_title("产能估计口径敏感性")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_method_files() -> None:
    candidates = """# Q2 候选方法

1. 字典序 MILP（选定）：先最小化剩余需求，再最小化产品换产，再最小化酒质切换，最后最大化产线偏好并压缩活动槽数。全局最优、约束透明。
2. CP-SAT：相邻状态逻辑表达简洁，但连续生产量需要缩放，未采用。
3. 瓶颈优先贪心：速度快，作为传统基线，无全局最优保证。

最小 PoC：代码内置两产品、单产线、三班次穷举，核验字典序目标应得到总产量 22、换产 1 次。
"""
    (METHOD_DIR / "q2_method_candidates.md").write_text(candidates, encoding="utf-8")

    explanation = """# Q2 最终方法说明

## 模型类型

带连续生产量和 0-1 排班变量的字典序混合整数线性规划（MILP）。

## 变量

| 符号 | 含义 | 类型 |
|---|---|---|
| x_ilt | 产品-酒质组合 i 是否在线 l 的班次 t 排产 | 0-1 |
| q_ilt | 对应计划生产量 | 连续非负 |
| r_i | 需求剩余量 | 连续非负 |
| s_ilt | 产品 i 在产线 l、班次 t 新开一个连续生产块 | 0-1 |
| h_glt | 酒质 g 在产线 l、班次 t 新开一个连续生产块 | 0-1 |

## 目标的字典序

1. 最小化 sum_i r_i；
2. 冻结最小剩余量，最小化 sum_i,l,t s_ilt；
3. 冻结产品换产数，最小化 sum_g,l,t h_glt；
4. 最大化产量加权的历史产线偏好；
5. 压缩活动班次数，消除零量或碎片化占位。

## 硬约束

每条产线每班最多一个产品；q_ilt <= C_il x_ilt；活动槽 q_ilt >= 1；仅允许历史兼容产线；sum_l,t q_ilt + r_i = D_i。新生产块满足 s_ilt >= x_ilt-x_il,t-1，酒质切换同理。

## 假设

- Hard/Q2：历史出现过的产品-酒质-产线关系视为可兼容；违反时缺少工程依据。
- Soft/Q2：估计班次产能可在计划期实现；通过 Q90、Q95、历史最大值及 ±5%/10%/20% 扰动检验。
- Soft/Q2：未提供生产量单位，模型保持附件原量纲。
- Soft/Q2：班次开始后至少安排 1 个原单位，防止零产量占位；该值来自数据最小正记录。
"""
    (METHOD_DIR / "q2_final_method_explanation.md").write_text(explanation, encoding="utf-8")


def write_reports(
    main_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    balanced_metrics: dict[str, Any],
    sensitivity: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    improvement = main_metrics["total_production"] - baseline_metrics["total_production"]
    switch_change = main_metrics["product_switches"] - baseline_metrics["product_switches"]
    report = f"""# Q2 最终结果分析（G4 冻结报告）

## 直接答案

字典序 MILP 在 140 个可用线班中安排 {main_metrics['active_line_shifts']} 个活动线班，总排产量为 {main_metrics['total_production']:.3f}，剩余需求 {main_metrics['total_shortage']:.3f}，总完成率 {main_metrics['completion_rate']:.6%}。产品换产 {main_metrics['product_switches']} 次，酒质切换 {main_metrics['quality_switches']} 次。

## 基线对比

瓶颈优先贪心总排产量 {baseline_metrics['total_production']:.3f}、剩余 {baseline_metrics['total_shortage']:.3f}、产品换产 {baseline_metrics['product_switches']} 次。MILP 相对基线增加产量 {improvement:.3f}，产品换产变化 {switch_change:+d} 次。

## 目标口径敏感性

若把第一目标改为“最小化各产品标准化剩余率之和”，均衡覆盖方案总排产量为 {balanced_metrics['total_production']:.3f}，完成率 {balanced_metrics['completion_rate']:.6%}，完全未排产品 {balanced_metrics['unscheduled_products']} 个，产品换产 {balanced_metrics['product_switches']} 次。该方案用于说明公平性与总产量之间的取舍，不替代题面总剩余量主结果。

## 解释边界

由于总需求显著超过周产能，本结果回答的是“在历史兼容和稳健班次产能口径下，优先压低总剩余量的最优计划”，并不承诺所有产品均得到相同比例满足。附件未提供单位，所有数量保持原数据量纲。

## 约束验证

验证状态：{'通过' if validation['passed'] else '失败'}；错误数 {validation['error_count']}。完整排程见 `tables/q2_schedule_milp.csv`。
"""
    (REPORT_DIR / "q2_final_result_analysis.md").write_text(report, encoding="utf-8")

    prod_values = sensitivity["total_production"]
    robust = f"""# Q2 鲁棒性与敏感性报告

## 支持的结论

| 结论 | 支持证据 | 置信度 |
|---|---|---|
| 周需求无法全部满足 | 基准模型和全部产能扰动场景均存在正剩余量 | 高 |
| 排产结果对产能口径敏感 | 场景总产量范围 {prod_values.min():.3f} 至 {prod_values.max():.3f} | 中 |
| 字典序目标避免任意权重 | 先冻结最小剩余量，再优化换产 | 高 |

## 脆弱结论

| 结论 | 脆弱原因 | 限定条件 |
|---|---|---|
| 具体产品分配与完成率 | 多个最优产量解可在产品间重新分配 | 以偏好目标选取当前代表解 |
| 班次产能 | 只有计划量、没有实际产量和单位 | 必须结合现场能力复核 |

## 结论边界

若真实产能、兼容关系或最小批量与附件历史不同，需要重新运行脚本；下游论文数字必须随之解冻并更新。
"""
    (ROBUST_DIR / "q2_robustness_report.md").write_text(robust, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_method_files()
    overall_start = time.perf_counter()
    poc = toy_poc()
    if not poc["passed"]:
        raise RuntimeError(f"toy PoC failed: {poc}")

    history, eligible, cleaning_report = clean_history()
    demand = pd.read_csv(INPUT_DEMAND, encoding="utf-8-sig")
    base_capacity = build_capacity_table(history, eligible, quantile=0.95, with_bootstrap=True)
    model_data = make_model_data(demand, base_capacity)

    history.to_csv(DATA_DIR / "history_clean_q2.csv", index=False, encoding="utf-8-sig")
    demand.to_csv(DATA_DIR / "weekly_demand_clean.csv", index=False, encoding="utf-8-sig")
    base_capacity.to_csv(TABLE_DIR / "q2_capacity_matrix_q95.csv", index=False, encoding="utf-8-sig")

    result, stages, artifacts = solve_lexicographic(model_data)
    schedule = result_to_schedule(model_data, result, artifacts, "字典序MILP-Q95")
    metrics, fulfillment = schedule_metrics(schedule, demand)
    validation = validate_schedule(schedule, demand, model_data)
    if not validation["passed"]:
        raise RuntimeError(validation["errors"])

    greedy = greedy_schedule(model_data)
    greedy_metrics, greedy_fulfillment = schedule_metrics(greedy, demand)
    greedy_validation = validate_schedule(greedy, demand, model_data)
    if not greedy_validation["passed"]:
        raise RuntimeError(greedy_validation["errors"])

    balanced_result, balanced_stages, balanced_artifacts = solve_lexicographic(model_data, shortage_mode="normalized")
    balanced = result_to_schedule(model_data, balanced_result, balanced_artifacts, "标准化剩余率MILP")
    balanced_metrics, balanced_fulfillment = schedule_metrics(balanced, demand)
    balanced_validation = validate_schedule(balanced, demand, model_data)
    if not balanced_validation["passed"]:
        raise RuntimeError(balanced_validation["errors"])

    schedule.to_csv(TABLE_DIR / "q2_schedule_milp.csv", index=False, encoding="utf-8-sig")
    greedy.to_csv(TABLE_DIR / "q2_schedule_greedy.csv", index=False, encoding="utf-8-sig")
    balanced.to_csv(TABLE_DIR / "q2_schedule_balanced.csv", index=False, encoding="utf-8-sig")
    fulfillment.to_csv(TABLE_DIR / "q2_demand_fulfillment.csv", index=False, encoding="utf-8-sig")
    greedy_fulfillment.to_csv(TABLE_DIR / "q2_demand_fulfillment_greedy.csv", index=False, encoding="utf-8-sig")
    balanced_fulfillment.to_csv(TABLE_DIR / "q2_demand_fulfillment_balanced.csv", index=False, encoding="utf-8-sig")

    scenarios: list[tuple[str, pd.DataFrame]] = [
        ("q90_shrunk", build_capacity_table(history, eligible, quantile=0.90)),
        ("q95_shrunk", base_capacity.drop(columns=["Bootstrap下限", "Bootstrap上限"]).copy()),
        ("raw_max", build_capacity_table(history, eligible, quantile=1.0, raw_max_mode=True)),
    ]
    for scale in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20):
        scenarios.append((f"scale_{scale:.2f}", build_capacity_table(history, eligible, quantile=0.95, scale=scale)))
    sensitivity_rows = []
    for name, cap in scenarios:
        summary, _ = solve_scenario(demand, cap, name)
        sensitivity_rows.append(summary)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLE_DIR / "q2_sensitivity.csv", index=False, encoding="utf-8-sig")

    plot_schedule_heatmap(schedule, FIGURE_DIR / "q2_schedule_heatmap.png")
    plot_fulfillment(fulfillment, FIGURE_DIR / "q2_fulfillment_rates.png")
    plot_sensitivity(sensitivity, FIGURE_DIR / "q2_sensitivity.png")
    fulfillment.to_csv(FIGURE_DIR / "q2_fulfillment_rates_source.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(FIGURE_DIR / "q2_sensitivity_source.csv", index=False, encoding="utf-8-sig")
    schedule.to_csv(FIGURE_DIR / "q2_schedule_heatmap_source.csv", index=False, encoding="utf-8-sig")

    write_reports(metrics, greedy_metrics, balanced_metrics, sensitivity, validation)
    run_summary = {
        "question": "Q2",
        "seed": SEED,
        "inputs": {
            str(INPUT_HISTORY.relative_to(ROOT)): {"sha256": sha256(INPUT_HISTORY), "rows": cleaning_report["raw_rows"]},
            str(INPUT_DEMAND.relative_to(ROOT)): {"sha256": sha256(INPUT_DEMAND), "rows": int(len(demand))},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "solver": "scipy.optimize.milp / HiGHS",
        },
        "method": "lexicographic MILP: shortage -> product switches -> quality switches -> preference -> active slots",
        "poc": poc,
        "cleaning": cleaning_report,
        "capacity": {
            "quantile": 0.95,
            "shrink_strength": 5.0,
            "compatible_product_line_pairs": int(len(model_data.pair_capacity)),
            "products": int(len(model_data.products)),
            "lines": int(len(model_data.lines)),
            "time_slots_per_line": len(TIME_SLOTS),
        },
        "solver_stages": stages,
        "milp_metrics": metrics,
        "greedy_metrics": greedy_metrics,
        "balanced_objective_stages": balanced_stages,
        "balanced_metrics": balanced_metrics,
        "validation": validation,
        "greedy_validation": greedy_validation,
        "balanced_validation": balanced_validation,
        "sensitivity": sensitivity.to_dict(orient="records"),
        "elapsed_seconds_total": time.perf_counter() - overall_start,
    }
    (ROUND_DIR / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    frozen = {
        "status": "frozen",
        "question": "Q2",
        "source_run_summary": str((ROUND_DIR / "run_summary.json").relative_to(ROOT)),
        "input_hashes": run_summary["inputs"],
        "numbers": {
            "total_demand": metrics["total_demand"],
            "total_production": metrics["total_production"],
            "total_shortage": metrics["total_shortage"],
            "completion_rate": metrics["completion_rate"],
            "product_switches": metrics["product_switches"],
            "quality_switches": metrics["quality_switches"],
            "active_line_shifts": metrics["active_line_shifts"],
            "fully_satisfied_products": metrics["fully_satisfied_products"],
            "partially_satisfied_products": metrics["partially_satisfied_products"],
            "unscheduled_products": metrics["unscheduled_products"],
            "greedy_total_production": greedy_metrics["total_production"],
            "greedy_product_switches": greedy_metrics["product_switches"],
            "balanced_total_production": balanced_metrics["total_production"],
            "balanced_completion_rate": balanced_metrics["completion_rate"],
            "balanced_product_switches": balanced_metrics["product_switches"],
            "balanced_unscheduled_products": balanced_metrics["unscheduled_products"],
        },
        "artifacts": [
            str((TABLE_DIR / "q2_schedule_milp.csv").relative_to(ROOT)),
            str((TABLE_DIR / "q2_demand_fulfillment.csv").relative_to(ROOT)),
            str((TABLE_DIR / "q2_schedule_balanced.csv").relative_to(ROOT)),
            str((TABLE_DIR / "q2_sensitivity.csv").relative_to(ROOT)),
            str((FIGURE_DIR / "q2_schedule_heatmap.png").relative_to(ROOT)),
            str((FIGURE_DIR / "q2_fulfillment_rates.png").relative_to(ROOT)),
            str((FIGURE_DIR / "q2_sensitivity.png").relative_to(ROOT)),
        ],
    }
    (FROZEN_DIR / "frozen_numbers.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    trace = """# Q2 数字溯源

| 数值/声明 | 来源 |
|---|---|
| 总需求、总排产量、剩余量、完成率、换产次数 | `frozen/Q2/frozen_numbers.json` |
| 每班具体安排 | `results/Q2/tables/q2_schedule_milp.csv` |
| 各产品完成率 | `results/Q2/tables/q2_demand_fulfillment.csv` |
| 敏感性范围 | `results/Q2/tables/q2_sensitivity.csv` |
| 数据清洗计数、求解器状态、输入哈希 | `results/Q2/experiments/round1/run_summary.json` |
"""
    (FROZEN_DIR / "number_trace.md").write_text(trace, encoding="utf-8")
    print(json.dumps({"milp_metrics": metrics, "greedy_metrics": greedy_metrics, "balanced_metrics": balanced_metrics, "stages": stages, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
