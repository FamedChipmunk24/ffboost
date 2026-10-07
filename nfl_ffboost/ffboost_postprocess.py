from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


RELEVANT_TOPK_BY_POS = {"QB": 24, "RB": 60, "WR": 80, "TE": 36}


def _name_key(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.lower().str.strip()
    x = x.str.replace(r"[^\w\s]", "", regex=True)
    x = x.str.replace(r"\b(jr|sr|iii|ii|iv)\b", "", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True)
    return x.str.strip()


def _load_adp(adp_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(adp_csv)
    df = df.rename(
        columns={
            "player_name": "player",
            "overall_rank": "adp_overall_rank",
            "position_rank": "adp_pos_rank",
            "adp": "adp_value",
        }
    )
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["player_key"] = _name_key(df["player"])
    for c in ["adp_overall_rank", "adp_pos_rank", "adp_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["season", "position", "player_key", "adp_overall_rank", "adp_pos_rank", "adp_value"]].dropna(
        subset=["season", "position", "player_key"]
    )


def _merge_adp(df: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["position"] = out["position"].astype(str).str.upper().str.strip()
    out["player_key"] = _name_key(out["player"])
    out = out.merge(adp, on=["season", "position", "player_key"], how="left")
    return out


def _optimize_blend_weights(bt: pd.DataFrame) -> Dict[str, float]:
    """
    Choose w per position for blended_score = w*model_score + (1-w)*adp_score
    where adp_score is higher for earlier ADP (negative ranks).
    Optimize MAE on actual top-K slice averaged across seasons.
    """
    weights: Dict[str, float] = {}
    grid = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]

    for pos, k in RELEVANT_TOPK_BY_POS.items():
        g = bt[bt["position"] == pos].copy()
        if g.empty:
            continue
        g["actual_rank"] = pd.to_numeric(g["actual_rank"], errors="coerce")
        g["pred_rank_score"] = pd.to_numeric(g["pred_rank_score"], errors="coerce")
        # adp score: prefer positional rank if available, else overall rank
        adp_pos = pd.to_numeric(g.get("adp_pos_rank"), errors="coerce")
        adp_ov = pd.to_numeric(g.get("adp_overall_rank"), errors="coerce")
        adp_rank = adp_pos.where(adp_pos.notna(), adp_ov)
        # fill missing adp with very late number so it doesn't dominate
        adp_rank = adp_rank.fillna(adp_rank.max() if adp_rank.notna().any() else 999.0)
        g["_adp_score"] = -adp_rank

        # Evaluate only on actual top-K
        g = g[g["actual_rank"] <= k].copy()
        if g.empty:
            continue

        best_w = 1.0
        best_mae = float("inf")
        for w in grid:
            g["_blend_score"] = w * g["pred_rank_score"] + (1.0 - w) * g["_adp_score"]
            # rank within each season
            g["_blend_rank"] = g.groupby("season")["_blend_score"].rank(method="first", ascending=False)
            mae = float((g["_blend_rank"] - g["actual_rank"]).abs().mean())
            if mae < best_mae:
                best_mae = mae
                best_w = float(w)
        weights[pos] = best_w

    return weights


def _apply_blend(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["pred_rank_score"] = pd.to_numeric(out["pred_rank_score"], errors="coerce")
    adp_pos = pd.to_numeric(out.get("adp_pos_rank"), errors="coerce")
    adp_ov = pd.to_numeric(out.get("adp_overall_rank"), errors="coerce")
    adp_rank = adp_pos.where(adp_pos.notna(), adp_ov)
    adp_rank = adp_rank.fillna(adp_rank.max() if adp_rank.notna().any() else 999.0)
    out["_adp_score"] = -adp_rank

    def _w(pos: str) -> float:
        return float(weights.get(str(pos).upper().strip(), 1.0))

    w_series = out["position"].astype(str).map(_w).astype(float)
    out["pred_rank_score"] = w_series * out["pred_rank_score"] + (1.0 - w_series) * out["_adp_score"]

    # Recompute pred_rank after blending
    out["pred_rank"] = out.groupby(["season", "position"])["pred_rank_score"].rank(method="first", ascending=False)
    out = out.drop(columns=["_adp_score"], errors="ignore")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest_csv", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--adp_csv", required=True)
    ap.add_argument("--out_backtest_csv", required=True)
    ap.add_argument("--out_pred_csv", required=True)
    ap.add_argument("--out_weights_json", required=True)
    args = ap.parse_args()

    adp = _load_adp(Path(args.adp_csv))

    bt = pd.read_csv(args.backtest_csv)
    pr = pd.read_csv(args.pred_csv)

    bt = _merge_adp(bt, adp)
    pr = _merge_adp(pr, adp)

    weights = _optimize_blend_weights(bt)
    Path(args.out_weights_json).write_text(json.dumps(weights, indent=2), encoding="utf-8")

    bt2 = _apply_blend(bt, weights)
    pr2 = _apply_blend(pr, weights)

    Path(args.out_backtest_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pred_csv).parent.mkdir(parents=True, exist_ok=True)
    bt2.to_csv(args.out_backtest_csv, index=False)
    pr2.to_csv(args.out_pred_csv, index=False)
    print(f"Wrote: {args.out_backtest_csv}")
    print(f"Wrote: {args.out_pred_csv}")
    print(f"Wrote: {args.out_weights_json}")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()

