from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["产品", "酒质", "产线"]
PQ_KEYS = ["产品", "酒质"]
REQUIRED_COLUMNS = KEYS + ["生产日期", "班次", "计划生产量"]
SHIFT_ORDER = {"早班": 0, "中班": 1}
EPS = 1e-12


def trimmed_mean(values: np.ndarray, proportion: float = 0.05) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    cut = int(np.floor(proportion * len(ordered)))
    kept = ordered[cut : len(ordered) - cut] if cut > 0 else ordered
    return float(np.mean(kept))


def stable_seed(base_seed: int, parts: tuple[str, ...]) -> int:
    payload = (str(base_seed) + "|" + "|".join(parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def minmax(values: pd.Series) -> pd.Series:
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= EPS:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - lo) / (hi - lo)


def bootstrap_features(
    frame: pd.DataFrame, rounds: int, base_seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(KEYS, sort=True):
        x = group["计划生产量"].to_numpy(dtype=float)
        n = int(x.size)
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if n >= 2 else float("nan")
        cv = std / abs(mean) if n >= 2 and abs(mean) > EPS else float("inf")

        rng = np.random.default_rng(stable_seed(base_seed, tuple(map(str, key))))
        samples = rng.choice(x, size=(rounds, n), replace=True)
        bootstrap_means = samples.mean(axis=1)
        lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
        lower = float(lower)
        upper = float(upper)
        denom = upper + lower
        rel_width = (
            float(2.0 * (upper - lower) / denom) if denom > EPS else float("inf")
        )
        rigid = bool(n >= 5 and rel_width < 0.05 and cv < 0.1)
        rigid_capacity = trimmed_mean(x, 0.05) if rigid else float("nan")
        lambda_discount = max(0.30, lower / upper) if upper > EPS else 0.30
        risk = max(0.0, min(1.0, (upper - lower) / upper)) if upper > EPS else 1.0

        rows.append(
            {
                "产品": key[0],
                "酒质": key[1],
                "产线": key[2],
                "G_ID": f"{key[0]}-{key[1]}-{key[2]}",
                "样本数": n,
                "均值": mean,
                "样本标准差": std,
                "变异系数CV": cv,
                "Bootstrap95%下限": lower,
                "Bootstrap95%上限": upper,
                "相对区间宽度": rel_width,
                "分层": "刚性" if rigid else "弹性",
                "刚性产能": rigid_capacity,
                "弹性折扣系数": lambda_discount,
                "风险指数": risk,
                "触发阈值": lower,
                "模型产能参数": rigid_capacity if rigid else upper,
                "有效产能": rigid_capacity if rigid else upper * lambda_discount,
                "置信等级": "高" if rigid else ("中" if n >= 5 else "低"),
            }
        )
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def critic_weights(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("CRITIC评价矩阵必须为 N×3")
    if matrix.shape[0] < 2:
        return np.full(3, 1.0 / 3.0)
    std = np.std(matrix, axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    conflict = np.sum(1.0 - corr, axis=0)
    info = std * conflict
    total = float(info.sum())
    return info / total if total > EPS else np.full(3, 1.0 / 3.0)


def build_preferences(
    clean: pd.DataFrame, capacity: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    counts = (
        clean.groupby(KEYS, as_index=False)
        .size()
        .rename(columns={"size": "历史班次数"})
    )
    pref = capacity.merge(counts, on=KEYS, how="left", validate="one_to_one")
    pref["候选产线数"] = pref.groupby(PQ_KEYS)["产线"].transform("nunique")
    pref["历史偏好度"] = pref["历史班次数"] / pref.groupby(PQ_KEYS)[
        "历史班次数"
    ].transform("sum")
    pref["稳定性指数"] = np.where(
        np.isfinite(pref["变异系数CV"]), 1.0 / (1.0 + pref["变异系数CV"]), 0.0
    )
    pref["产能归一化"] = pref.groupby(PQ_KEYS, group_keys=False)[
        "模型产能参数"
    ].transform(minmax)
    pref["稳定性归一化"] = pref.groupby(PQ_KEYS, group_keys=False)[
        "稳定性指数"
    ].transform(minmax)
    pref["偏好度归一化"] = pref.groupby(PQ_KEYS, group_keys=False)[
        "历史偏好度"
    ].transform(minmax)

    flexible = pref[pref["候选产线数"] >= 2]
    matrix = flexible[["产能归一化", "稳定性归一化", "偏好度归一化"]].to_numpy(
        dtype=float
    )
    weights = critic_weights(matrix)
    pref["综合适配分"] = (
        pref["产能归一化"] * weights[0]
        + pref["稳定性归一化"] * weights[1]
        + pref["偏好度归一化"] * weights[2]
    )
    pref["优先级"] = (
        pref.groupby(PQ_KEYS)["综合适配分"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    pref["是否首选"] = (pref["优先级"] == 1).astype(int)
    return pref.sort_values(PQ_KEYS + ["优先级"]).reset_index(drop=True), weights


def build_lock_decisions(preference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    z_lock: dict[str, int] = {}
    for key, group in preference.groupby(PQ_KEYS, sort=True):
        ordered = group.sort_values(["优先级", "产线"])
        lines = ordered["产线"].tolist()
        best = str(ordered.iloc[0]["产线"])
        pq_id = f"{key[0]}-{key[1]}"
        if len(lines) == 1:
            decision = "历史唯一产线锁定"
            locked_line = best
        elif pq_id == "P0070-Q03":
            decision = "保留全局优化自由变量"
            locked_line = ""
        else:
            decision = "CRITIC首选产线锁定"
            locked_line = best
        if locked_line:
            z_lock[f"{pq_id}|{locked_line}"] = 1
        rows.append(
            {
                "产品": key[0],
                "酒质": key[1],
                "候选产线数": len(lines),
                "候选产线": ",".join(lines),
                "首选产线": best,
                "锁定产线": locked_line,
                "决策方式": decision,
            }
        )
    return pd.DataFrame(rows), z_lock


def build_switch_cost(
    clean: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    work = clean.copy()
    work["生产日期"] = pd.to_datetime(work["生产日期"], errors="raise")
    work["班次顺序"] = work["班次"].map(SHIFT_ORDER)
    if work["班次顺序"].isna().any():
        bad = sorted(work.loc[work["班次顺序"].isna(), "班次"].astype(str).unique())
        raise ValueError(f"存在无法排序的班次: {bad}")
    work = work.sort_values(["产线", "生产日期", "班次顺序", "产品", "酒质"])
    work["前产品"] = work.groupby("产线")["产品"].shift(1)
    work["前产量"] = work.groupby("产线")["计划生产量"].shift(1)
    transitions = work.dropna(subset=["前产品", "前产量"]).copy()
    ratio = transitions["计划生产量"] / transitions["前产量"].replace(0, np.nan)
    transitions["损失率"] = (1.0 - ratio).clip(lower=0.0, upper=1.0).fillna(0.15)
    observed = (
        transitions.groupby(["前产品", "产品"], as_index=False)["损失率"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "平均换产损失率", "count": "相邻记录数"})
    )

    products = sorted(clean["产品"].astype(str).unique())
    index = {p: i for i, p in enumerate(products)}
    dense = np.full((len(products), len(products)), 0.15, dtype=float)
    np.fill_diagonal(dense, 0.0)
    for row in observed.itertuples(index=False):
        dense[index[row.前产品], index[row.产品]] = (
            0.0 if row.前产品 == row.产品 else float(row.平均换产损失率)
        )
    return dense, observed, products


def build_stamina(clean: pd.DataFrame) -> pd.DataFrame:
    stats = clean.groupby("产线")["计划生产量"].agg(["sum", "count", "mean"]).reset_index()
    baseline = float(stats["mean"].mean())
    stats["产线耐力系数"] = stats["mean"] / baseline if baseline > EPS else 1.0
    return stats.rename(columns={"sum": "总产量", "count": "班次数", "mean": "单班均产量"})


def json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题一稳健产能与偏好矩阵预处理")
    parser.add_argument("--input", default="clean.csv")
    parser.add_argument("--output-dir", default="outputs/q1_model_run_20260713")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "q1_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    source = Path(args.input)
    clean = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in clean.columns]
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    if clean[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("clean.csv 的模型必需字段存在缺失值")
    if (clean["计划生产量"] <= 0).any():
        raise ValueError("计划生产量必须全部为正数")

    logging.info("开始计算: rows=%d bootstrap_rounds=%d", len(clean), args.bootstrap_rounds)
    capacity = bootstrap_features(clean, args.bootstrap_rounds, args.seed)
    preference, weights = build_preferences(clean, capacity)
    locks, z_lock = build_lock_decisions(preference)
    switch_matrix, switch_observed, products = build_switch_cost(clean)
    stamina = build_stamina(clean)

    capacity.to_csv(output_dir / "q1_capacity.csv", index=False, encoding="utf-8-sig")
    preference.to_csv(output_dir / "q1_preference.csv", index=False, encoding="utf-8-sig")
    locks.to_csv(output_dir / "q1_lock_decisions.csv", index=False, encoding="utf-8-sig")
    switch_observed.to_csv(
        output_dir / "q1_switch_observed.csv", index=False, encoding="utf-8-sig"
    )
    stamina.to_csv(output_dir / "q1_stamina.csv", index=False, encoding="utf-8-sig")

    rigid = capacity[capacity["分层"] == "刚性"]
    flex = capacity[capacity["分层"] == "弹性"]
    multi = locks[locks["候选产线数"] >= 2]
    free = locks[locks["决策方式"] == "保留全局优化自由变量"]
    z_hash = hashlib.sha256(
        json.dumps(z_lock, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary = {
        "input": str(source.resolve()),
        "input_rows": len(clean),
        "bootstrap_rounds": args.bootstrap_rounds,
        "seed": args.seed,
        "actual_dimensions": {
            "products": clean["产品"].nunique(),
            "qualities": clean["酒质"].nunique(),
            "lines": clean["产线"].nunique(),
            "production_units_product_quality_line": len(capacity),
            "product_quality_combinations": len(locks),
        },
        "document_claim_discrepancies": {
            "document_products": 86,
            "actual_products": clean["产品"].nunique(),
            "document_single_line_combinations": 121,
            "actual_single_line_combinations": int((locks["候选产线数"] == 1).sum()),
        },
        "layering": {
            "rigid_units": len(rigid),
            "flex_units": len(flex),
            "rigid_share": len(rigid) / len(capacity),
        },
        "assignment_reduction": {
            "single_line_combinations": int((locks["候选产线数"] == 1).sum()),
            "multi_line_combinations": len(multi),
            "critic_locked_multi_line_combinations": int(
                (multi["决策方式"] == "CRITIC首选产线锁定").sum()
            ),
            "free_product_quality_combinations": free[
                ["产品", "酒质", "候选产线"]
            ].to_dict("records"),
            "z_lock_sha256": z_hash,
        },
        "critic_weights": {
            "capacity": weights[0],
            "stability": weights[1],
            "historical_preference": weights[2],
        },
        "switch_matrix": {
            "dimension": list(switch_matrix.shape),
            "observed_transition_pairs": len(switch_observed),
            "unobserved_default": 0.15,
        },
        "integrity_notes": [
            "Urgency_vec 未生成：现有历史排产表没有订单计划日期和实际完成日期。",
            "P0120-Q05 不存在于 clean.csv，故无法复现文档指定的该单元拉闸测试。",
            "本次运行完成问题一静态参数预处理，不包含问题二至四的完整MILP求解。",
        ],
    }
    summary = json.loads(json.dumps(summary, ensure_ascii=False, default=json_safe))
    with (output_dir / "q1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    params = {
        "G_rigid_set": rigid["G_ID"].tolist(),
        "G_flex_set": flex["G_ID"].tolist(),
        "C_rigid_dict": dict(zip(rigid["G_ID"], rigid["刚性产能"])),
        "U_upper_dict": dict(zip(flex["G_ID"], flex["Bootstrap95%上限"])),
        "Lambda_dict": dict(zip(flex["G_ID"], flex["弹性折扣系数"])),
        "Risk_dict": dict(zip(capacity["G_ID"], capacity["风险指数"])),
        "Trigger_dict": dict(zip(capacity["G_ID"], capacity["触发阈值"])),
        "Z_Lock_dict": z_lock,
        "SwitchMatrix": switch_matrix,
        "SwitchProducts": products,
        "SwitchDefault": 0.15,
        "Urgency_vec": None,
        "Stamina_vec": dict(zip(stamina["产线"], stamina["产线耐力系数"])),
        "CRITIC_weights": weights,
        "metadata": summary,
    }
    with (output_dir / "q1_model_params.pkl").open("wb") as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

    assert len(capacity) == clean.groupby(KEYS).ngroups
    assert set(rigid["G_ID"]).isdisjoint(set(flex["G_ID"]))
    assert len(rigid) + len(flex) == len(capacity)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all((capacity["弹性折扣系数"] >= 0.30) & (capacity["弹性折扣系数"] <= 1.0 + EPS))
    assert switch_matrix.shape == (clean["产品"].nunique(), clean["产品"].nunique())
    assert len(free) == 1 and free.iloc[0]["产品"] == "P0070" and free.iloc[0]["酒质"] == "Q03"

    logging.info(
        "运行完成: units=%d rigid=%d flex=%d pq=%d multi=%d free=%d",
        len(capacity),
        len(rigid),
        len(flex),
        len(locks),
        len(multi),
        len(free),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
