from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

from nfl_ffboost.nflverse_role_data import merge_player_season_snaps
from nfl_ffboost.supplemental_merge import merge_supplemental_features_optional


DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]
RELEVANT_TOPK_BY_POS: Dict[str, int] = {"QB": 24, "RB": 60, "WR": 80, "TE": 36}
TRAIN_POOL_CAP_BY_POS: Dict[str, int] = {"QB": 36, "RB": 100, "WR": 140, "TE": 60}


@dataclass(frozen=True)
class Cols:
    season: str = "season"
    player: str = "player_display_name"
    position: str = "position"
    team: str = "recent_team"
    points: str = "fantasy_points_ppr"


def _add_similar_trend_features(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    """
    Leakage-safe similar-player cohort features.

    For each row (season t, target_season t+1), we compute bins for:
      - age
      - prior-year fantasy points
      - position-specific volume proxy

    Then, using ONLY historical rows with target_season < current target_season,
    we compute typical next-year outcomes for that cohort:
      - sim_next_points_mean
      - sim_next_delta_mean
      - sim_bust_rate (next_points < 0.8 * current_points)

    This approximates "similar player trajectories" efficiently via groupby.
    """
    out = df.copy()
    out[cols.position] = out[cols.position].astype(str).str.upper().str.strip()
    out[cols.season] = pd.to_numeric(out[cols.season], errors="coerce")
    out["target_season"] = pd.to_numeric(out.get("target_season"), errors="coerce")
    out[cols.points] = pd.to_numeric(out[cols.points], errors="coerce")
    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")

    # Volume proxy by position
    attempts = pd.to_numeric(out.get("attempts"), errors="coerce")
    carries = pd.to_numeric(out.get("carries"), errors="coerce")
    targets = pd.to_numeric(out.get("targets"), errors="coerce")

    vol = pd.Series(np.nan, index=out.index, dtype=float)
    pos = out[cols.position]
    vol = np.where(pos == "QB", attempts, vol)
    vol = np.where(pos != "QB", carries.fillna(0) + targets.fillna(0), vol)
    out["_vol_proxy"] = pd.to_numeric(vol, errors="coerce")

    # Bins (fixed edges; boundaries don't use future labels, just stabilize grouping)
    out["_age_bin"] = pd.cut(out["age"], bins=[0, 22, 24, 26, 28, 30, 32, 60], labels=False, include_lowest=True)
    out["_pts_bin"] = pd.cut(out[cols.points], bins=[-1, 80, 120, 160, 200, 240, 280, 340, 600], labels=False, include_lowest=True)
    out["_vol_bin"] = pd.cut(out["_vol_proxy"], bins=[-1, 30, 60, 100, 140, 200, 300, 500, 2000], labels=False, include_lowest=True)

    # Requires next-season points already computed
    out["target_next_points"] = pd.to_numeric(out.get("target_next_points"), errors="coerce")
    out["_next_delta"] = out["target_next_points"] - out[cols.points]
    out["_bust"] = (out["target_next_points"] < (0.8 * out[cols.points])).astype(float)

    keys = [cols.position, "_age_bin", "_pts_bin", "_vol_bin"]
    out["sim_next_points_mean"] = np.nan
    out["sim_next_delta_mean"] = np.nan
    out["sim_bust_rate"] = np.nan

    # Iterate by target season to enforce leakage-safety
    ts_vals = (
        pd.to_numeric(out["target_season"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    hist = out.dropna(subset=["target_season", "target_next_points", cols.points]).copy()
    for ts in ts_vals:
        # history only from earlier target seasons
        hist_pool = hist[hist["target_season"] < ts]
        if hist_pool.empty:
            continue
        agg = (
            hist_pool.groupby(keys, dropna=False)
            .agg(
                sim_next_points_mean=("target_next_points", "mean"),
                sim_next_delta_mean=("_next_delta", "mean"),
                sim_bust_rate=("_bust", "mean"),
            )
            .reset_index()
        )
        mask = (out["target_season"] == ts)
        if not mask.any():
            continue
        merged = out.loc[mask, keys].merge(agg, on=keys, how="left")
        out.loc[mask, "sim_next_points_mean"] = merged["sim_next_points_mean"].to_numpy()
        out.loc[mask, "sim_next_delta_mean"] = merged["sim_next_delta_mean"].to_numpy()
        out.loc[mask, "sim_bust_rate"] = merged["sim_bust_rate"].to_numpy()

    out = out.drop(columns=["_vol_proxy", "_age_bin", "_pts_bin", "_vol_bin", "_next_delta", "_bust"], errors="ignore")
    return out


def _name_key(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.lower().str.strip()
    # common punctuation / suffix cleanup
    x = x.str.replace(r"[^\w\s]", "", regex=True)
    x = x.str.replace(r"\b(jr|sr|iii|ii|iv)\b", "", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True)
    return x.str.strip()


def _load_adp(adp_csv: str) -> pd.DataFrame:
    df = pd.read_csv(adp_csv)
    # Expected columns from fantasydata_adp_2010_2025.csv (or fantasydata_adp_2014_2025.csv):
    # season, overall_rank, player_name, position, position_rank, adp, team ...
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


def _merge_adp_for_target_season(df: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """
    Attach preseason ADP for the target season (t+1) onto rows from season t.
    """
    out = df.copy()
    out["player_key"] = _name_key(out["player_display_name"])
    out["adp_season"] = pd.to_numeric(out["target_season"], errors="coerce")
    merged = out.merge(
        adp,
        left_on=["adp_season", "position", "player_key"],
        right_on=["season", "position", "player_key"],
        how="left",
    )
    merged = merged.drop(columns=["season"])
    return merged


def _rank_within(df: pd.DataFrame, season_col: str, pos_col: str, points_col: str) -> pd.Series:
    return df.groupby([season_col, pos_col])[points_col].rank(method="first", ascending=False).astype(int)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    out = pd.to_numeric(a, errors="coerce") / b2
    return out.replace([np.inf, -np.inf], np.nan)


def _build_neo_features(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    """
    Feature builder for Neo v1. Uses only information available in the row's season.
    Produces a compact set of stable, high-signal features designed to predict next-year rank.
    """
    out = df.copy()
    # core
    out[cols.season] = pd.to_numeric(out[cols.season], errors="coerce").astype("Int64")
    out[cols.position] = out[cols.position].astype(str).str.upper().str.strip()
    out[cols.player] = out[cols.player].astype(str).str.strip()
    out[cols.team] = out[cols.team].astype(str).str.upper().str.strip()
    out[cols.points] = pd.to_numeric(out[cols.points], errors="coerce")

    # numeric columns we may use (if present)
    for c in ["age", "games", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
              "carries", "rushing_yards", "rushing_tds",
              "targets", "receptions", "receiving_yards", "receiving_tds"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Efficiency + rates
    if "attempts" in out.columns and "passing_yards" in out.columns:
        out["pass_yds_per_att"] = _safe_div(out["passing_yards"], out["attempts"])
    if "attempts" in out.columns and "passing_tds" in out.columns:
        out["pass_td_per_att"] = _safe_div(out["passing_tds"], out["attempts"])
    if "attempts" in out.columns and "passing_interceptions" in out.columns:
        out["int_per_att"] = _safe_div(out["passing_interceptions"], out["attempts"])

    if "carries" in out.columns and "rushing_yards" in out.columns:
        out["rush_yds_per_att"] = _safe_div(out["rushing_yards"], out["carries"])
    if "carries" in out.columns and "rushing_tds" in out.columns:
        out["rush_td_per_att"] = _safe_div(out["rushing_tds"], out["carries"])

    if "targets" in out.columns and "receptions" in out.columns:
        out["catch_rate"] = _safe_div(out["receptions"], out["targets"])
    if "receptions" in out.columns and "receiving_yards" in out.columns:
        out["yds_per_rec"] = _safe_div(out["receiving_yards"], out["receptions"])
    if "receptions" in out.columns and "receiving_tds" in out.columns:
        out["tds_per_rec"] = _safe_div(out["receiving_tds"], out["receptions"])

    # Simple rolling history (per-player) on points and key volume proxies
    out = out.sort_values([cols.player, cols.position, cols.season])
    grp = out.groupby([cols.player, cols.position], sort=False)
    out["pts_roll3_mean"] = grp[cols.points].rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    out["pts_roll3_std"] = grp[cols.points].rolling(3, min_periods=2).std().reset_index(level=[0, 1], drop=True)
    out["pts_lag1"] = grp[cols.points].shift(1)
    out["pts_delta"] = out[cols.points] - out["pts_lag1"]

    if "targets" in out.columns:
        out["targets_lag1"] = grp["targets"].shift(1)
        out["targets_delta"] = out["targets"] - out["targets_lag1"]
    if "carries" in out.columns:
        out["carries_lag1"] = grp["carries"].shift(1)
        out["carries_delta"] = out["carries"] - out["carries_lag1"]
    if "attempts" in out.columns:
        out["attempts_lag1"] = grp["attempts"].shift(1)
        out["attempts_delta"] = out["attempts"] - out["attempts_lag1"]

    return out


def _make_model(
    position: str,
    random_state: int = 42,
    model_mode: str = "hybrid",
    override_params: Dict[str, Any] | None = None,
):
    pos = (position or "").upper().strip()
    model_mode = (model_mode or "hybrid").strip().lower()
    if model_mode not in {"hybrid", "rank_all", "reg_all"}:
        model_mode = "hybrid"
    # rank_all: ranker for every position
    # reg_all: regressor for every position
    # hybrid: ranker for QB, regressor for RB/WR/TE
    if model_mode == "rank_all":
        use_ranker = True
    elif model_mode == "reg_all":
        use_ranker = False
    else:
        use_ranker = (pos == "QB")

    # Position-specific defaults (WR especially tends to overfit).
    if pos == "WR":
        n_estimators = 700
        max_depth = 4
        learning_rate = 0.05
        reg_alpha = 0.2
        reg_lambda = 1.4
    elif pos == "RB":
        n_estimators = 650
        max_depth = 5
        learning_rate = 0.05
        reg_alpha = 0.12
        reg_lambda = 1.3
    elif pos == "TE":
        n_estimators = 650
        max_depth = 5
        learning_rate = 0.05
        reg_alpha = 0.12
        reg_lambda = 1.3
    else:  # QB
        n_estimators = 650
        max_depth = 6
        learning_rate = 0.05
        reg_alpha = 0.1
        reg_lambda = 1.2

    common: Dict[str, Any] = dict(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    if override_params:
        common.update({k: v for k, v in override_params.items() if v is not None})

    if use_ranker:
        return XGBRanker(objective="rank:pairwise", eval_metric="ndcg", **common)
    return XGBRegressor(objective="reg:squarederror", **common)


def _make_relevance_model(position: str, random_state: int = 42) -> XGBClassifier:
    """
    Stage-1 model: predict probability of finishing in relevant top-K next season.
    """
    pos = (position or "").upper().strip()
    if pos == "WR":
        max_depth = 4
        n_estimators = 500
        learning_rate = 0.06
    elif pos == "RB":
        max_depth = 5
        n_estimators = 450
        learning_rate = 0.06
    elif pos == "TE":
        max_depth = 5
        n_estimators = 450
        learning_rate = 0.06
    else:  # QB
        max_depth = 5
        n_estimators = 400
        learning_rate = 0.06
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.2,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def _make_preprocessor(num_feats: List[str], cat_feats: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)], remainder="drop")


def _feature_candidates(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Neo v1: allow many numeric candidates (fast selection per split).
    We explicitly exclude anything that looks like a target/next-year leakage.
    """
    excluded = {
        "target_season",
        "target_next_points",
        "target_next_rank",
        "actual_points",
        "actual_rank",
    }
    excluded |= {"player_display_name", "player", "position", "season", "recent_team", "team"}

    num_cols: List[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        cl = str(c).lower()
        if "target_" in cl or cl.startswith("target") or "actual_" in cl:
            continue
        if "next" in cl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    # Categorical: keep team as a soft roster/context signal
    cat_cols = ["recent_team"] if "recent_team" in df.columns else []
    return num_cols, cat_cols


def _select_topk_numeric_features(train_df: pd.DataFrame, num_candidates: List[str], k: int) -> List[str]:
    """
    Fast, robust feature selection based on absolute Spearman correlation
    with next-year points (proxy for ordering).
    """
    if k <= 0:
        return []
    y = pd.to_numeric(train_df["target_next_points"], errors="coerce")
    if y.isna().all():
        return []

    scores: List[tuple[str, float]] = []
    for c in num_candidates:
        s = pd.to_numeric(train_df[c], errors="coerce")
        if s.isna().all():
            continue
        # rank-corr is more stable for skewed stats
        corr = s.rank(pct=True).corr(y.rank(pct=True))
        if corr is None or not np.isfinite(corr):
            continue
        scores.append((c, float(abs(corr))))
    scores.sort(key=lambda t: t[1], reverse=True)
    top = [c for (c, _) in scores[:k]]

    # Always keep last-year points if present (strong baseline)
    if "fantasy_points_ppr" in train_df.columns and "fantasy_points_ppr" not in top:
        top = ["fantasy_points_ppr"] + top
        top = top[:k]
    return top


def _k_features_for_pos(default_k: int, position: str) -> int:
    pos = (position or "").upper().strip()
    if pos == "WR":
        return max(40, int(default_k * 0.75))
    if pos == "RB":
        return max(60, int(default_k * 1.1))
    if pos == "TE":
        return max(50, int(default_k * 0.9))
    return default_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--positions", default=",".join(DEFAULT_POSITIONS))
    ap.add_argument(
        "--skip-adp",
        dest="skip_adp",
        action="store_true",
        help="Disable preseason ADP merge (default: ADP on; uses fantasydata_adp_2010_2025.csv).",
    )
    ap.add_argument("--adp_csv", default=str(Path(__file__).resolve().parent.parent / "nfl_data" / "fantasydata_adp_2010_2025.csv"))
    ap.add_argument(
        "--skip-snaps",
        dest="skip_snaps",
        action="store_true",
        help="Disable nflverse snap share features (2012+). Default: merge snaps.",
    )
    ap.add_argument(
        "--skip-supplemental",
        dest="skip_supplemental",
        action="store_true",
        help="Disable supplemental draft/Elo/schedule features (default: on if nfl_data files exist).",
    )
    ap.add_argument(
        "--snaps_csv",
        default=str(Path(__file__).resolve().parent.parent / "nfl_data" / "nflverse_snap_player_season_2012_2025.csv"),
        help="Cache path for aggregated player-season snap features.",
    )
    ap.add_argument("--train_start", type=int, default=2002)
    ap.add_argument("--qb_train_start", type=int, default=2010)
    ap.add_argument("--rb_wr_te_train_start", type=int, default=2010)
    ap.add_argument("--initial_train_end", type=int, default=2015)
    ap.add_argument("--test_end", type=int, default=2025)
    ap.add_argument("--k_features", type=int, default=200)
    ap.add_argument("--n_ensemble", type=int, default=5)
    ap.add_argument("--relevant_topk_qb", type=int, default=RELEVANT_TOPK_BY_POS["QB"])
    ap.add_argument("--relevant_topk_rb", type=int, default=RELEVANT_TOPK_BY_POS["RB"])
    ap.add_argument("--relevant_topk_wr", type=int, default=RELEVANT_TOPK_BY_POS["WR"])
    ap.add_argument("--relevant_topk_te", type=int, default=RELEVANT_TOPK_BY_POS["TE"])
    ap.add_argument("--relevant_weight", type=float, default=3.0)
    ap.add_argument("--train_pool_cap_qb", type=int, default=TRAIN_POOL_CAP_BY_POS["QB"])
    ap.add_argument("--train_pool_cap_rb", type=int, default=TRAIN_POOL_CAP_BY_POS["RB"])
    ap.add_argument("--train_pool_cap_wr", type=int, default=TRAIN_POOL_CAP_BY_POS["WR"])
    ap.add_argument("--train_pool_cap_te", type=int, default=TRAIN_POOL_CAP_BY_POS["TE"])
    ap.add_argument("--use_relevance_model", action="store_true")
    ap.add_argument("--model_mode", default="hybrid", choices=["hybrid", "rank_all", "reg_all"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cols = Cols()
    raw = pd.read_csv(args.data)
    df = _build_neo_features(raw, cols)

    nfl_data_dir = Path(__file__).resolve().parent.parent / "nfl_data"

    # Optional: merge snap share features (offense/defense/ST) by player-season.
    if not bool(args.skip_snaps):
        try:
            df = merge_player_season_snaps(
                df,
                cache_dir=nfl_data_dir,
                season_col=cols.season,
                player_id_col="player_id",
                snaps_csv=str(args.snaps_csv) if args.snaps_csv else None,
            )
        except Exception:
            pass

    # Draft capital + team Elo end-season + schedule rest (leakage-safe).
    df = merge_supplemental_features_optional(
        df,
        enabled=not bool(args.skip_supplemental),
        season_col=cols.season,
        team_col=cols.team,
        player_id_col="player_id",
        nfl_data_dir=nfl_data_dir,
    )

    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    df = df[df[cols.position].isin(positions)].copy()

    # targets
    df = df.dropna(subset=[cols.season, cols.player, cols.position, cols.points]).copy()
    df[cols.season] = df[cols.season].astype(int)
    df["target_season"] = df[cols.season] + 1

    # Optional ADP merge (preseason ADP for target season)
    if (not bool(args.skip_adp)) and args.adp_csv:
        try:
            adp = _load_adp(args.adp_csv)
            df = _merge_adp_for_target_season(df, adp)
        except Exception:
            pass
    # next-year points and rank
    df = df.sort_values([cols.player, cols.position, cols.season])
    grp = df.groupby([cols.player, cols.position], sort=False)
    df["target_next_points"] = grp[cols.points].shift(-1)
    df = df.dropna(subset=["target_next_points"]).copy()
    df["target_next_points"] = pd.to_numeric(df["target_next_points"], errors="coerce")
    df = df.dropna(subset=["target_next_points"]).copy()

    # Similar-player cohort trend features (leakage-safe)
    df = _add_similar_trend_features(df, cols)

    df["target_next_rank"] = df.groupby([cols.position, "target_season"])["target_next_points"].rank(
        method="first", ascending=False
    )

    # actuals for eval
    actual = df[[cols.player, cols.position, "target_season", "target_next_points"]].copy()
    actual = actual.rename(
        columns={
            cols.player: "player",
            cols.position: "position",
            "target_season": "season",
            "target_next_points": "actual_points",
        }
    )
    actual["actual_rank"] = actual.groupby(["position", "season"])["actual_points"].rank(method="first", ascending=False)

    metrics: List[Dict[str, object]] = []
    preds: List[Dict[str, object]] = []

    rel_topk = {
        "QB": int(args.relevant_topk_qb),
        "RB": int(args.relevant_topk_rb),
        "WR": int(args.relevant_topk_wr),
        "TE": int(args.relevant_topk_te),
    }
    train_pool_cap = {
        "QB": int(args.train_pool_cap_qb),
        "RB": int(args.train_pool_cap_rb),
        "WR": int(args.train_pool_cap_wr),
        "TE": int(args.train_pool_cap_te),
    }

    for pos in positions:
        pos_df = df[df[cols.position] == pos].copy()
        if pos_df.empty:
            continue
        train_start = args.qb_train_start if pos == "QB" else args.rb_wr_te_train_start
        train_start = max(int(args.train_start), int(train_start))

        num_candidates, cat_feats = _feature_candidates(pos_df)

        for test_year in range(int(args.initial_train_end) + 1, int(args.test_end) + 1):
            train_end = test_year - 2
            if train_end < train_start:
                continue
            train = pos_df[(pos_df[cols.season] >= train_start) & (pos_df[cols.season] <= train_end)].copy()
            test = pos_df[pos_df[cols.season] == (test_year - 1)].copy()
            if train.empty or test.empty:
                continue

            # Train only on a relevant pool to avoid fitting the chaotic deep tail.
            cap = int(train_pool_cap.get(pos, 9999))
            train = train[pd.to_numeric(train["target_next_rank"], errors="coerce") <= cap].copy()
            if train.empty:
                continue

            k_pos = _k_features_for_pos(int(args.k_features), pos)
            num_feats = _select_topk_numeric_features(train, num_candidates, k_pos)
            Xtr = train[num_feats + cat_feats]
            ytr = pd.to_numeric(train["target_next_points"], errors="coerce")
            Xte = test[num_feats + cat_feats]

            pre = _make_preprocessor(num_feats, cat_feats)
            Xtr_m = pre.fit_transform(Xtr)
            Xte_m = pre.transform(Xte)

            k_rel = int(rel_topk.get(pos, 60))
            # Increase weight on future top-K finishers in training
            sw = np.where(pd.to_numeric(train["target_next_rank"], errors="coerce").to_numpy() <= k_rel, float(args.relevant_weight), 1.0)

            # Stage-1: relevance probability
            if args.use_relevance_model:
                y_rel = (pd.to_numeric(train["target_next_rank"], errors="coerce").to_numpy() <= k_rel).astype(int)
                rel_models = []
                rel_preds_train = []
                rel_preds_test = []
                for seed in range(max(1, int(args.n_ensemble))):
                    rm = _make_relevance_model(position=pos, random_state=7_000 + seed)
                    rm.fit(Xtr_m, y_rel, sample_weight=sw)
                    rel_models.append(rm)
                    rel_preds_train.append(rm.predict_proba(Xtr_m)[:, 1])
                    rel_preds_test.append(rm.predict_proba(Xte_m)[:, 1])
                p_rel_train = np.mean(np.vstack(rel_preds_train), axis=0)
                p_rel_test = np.mean(np.vstack(rel_preds_test), axis=0)
                Xtr_m = hstack([Xtr_m, csr_matrix(p_rel_train.reshape(-1, 1))]).tocsr()
                Xte_m = hstack([Xte_m, csr_matrix(p_rel_test.reshape(-1, 1))]).tocsr()
            else:
                rel_models = None
                p_rel_test = None

            n_ens = max(1, int(args.n_ensemble))
            pred_scores = []
            for seed in range(n_ens):
                model = _make_model(position=pos, random_state=42 + seed, model_mode=str(args.model_mode))
                if isinstance(model, XGBRanker):
                    # group by target season (season+1), within this position
                    tr_group = (
                        train[["target_season"]]
                        .assign(_idx=np.arange(len(train)))
                        .sort_values("target_season")
                    )
                    order = tr_group["_idx"].to_numpy()
                    ytr_ord = ytr.to_numpy()[order]
                    Xtr_ord = Xtr_m[order]
                    group_sizes = tr_group.groupby("target_season").size().to_list()
                    # This XGBoost build expects group-level weights for rankers.
                    # We keep weighting for regressors (RB/WR/TE) and skip it for rankers (QB).
                    model.fit(Xtr_ord, ytr_ord, group=group_sizes)
                else:
                    model.fit(Xtr_m, ytr.to_numpy(), sample_weight=sw)
                pred_scores.append(model.predict(Xte_m))
            pred_score = np.mean(np.vstack(pred_scores), axis=0)

            out = pd.DataFrame(
                {
                    "season": test_year,
                    "player": test[cols.player].astype(str).values,
                    "position": pos,
                    "team": test[cols.team].astype(str).values,
                    "pred_rank_score": pred_score,
                }
            )
            if p_rel_test is not None:
                out["p_relevant"] = p_rel_test
            out["pred_rank"] = out.groupby(["season", "position"])["pred_rank_score"].rank(method="first", ascending=False)
            out = out.merge(actual, on=["player", "position", "season"], how="left")
            out = out.dropna(subset=["actual_rank", "actual_points"])
            if out.empty:
                continue
            # compute points estimate for display: map pred_rank -> avg actual points by rank (from train window)
            train_actual = actual[(actual["position"] == pos) & (actual["season"] >= train_start) & (actual["season"] <= train_end)]
            by_rank = train_actual.groupby("actual_rank")["actual_points"].mean().sort_index()
            if by_rank.empty:
                out["pred_points"] = np.nan
            else:
                max_rank = int(max(by_rank.index.max(), out["pred_rank"].max()))
                filled = (
                    by_rank.reindex(range(1, max_rank + 1))
                    .interpolate(limit_direction="both")
                    .ffill()
                    .bfill()
                )
                out["pred_points"] = out["pred_rank"].round().astype(int).clip(1, max_rank).map(filled).astype(float)
            out = out.dropna(subset=["pred_points"])
            if out.empty:
                continue

            # Keep only relevant players for reporting:
            # - predicted top-K (what you'd actually draft from)
            # - plus actual top-K (so we can evaluate MAE on the true relevant set)
            out["is_relevant_pred"] = (pd.to_numeric(out["pred_rank"], errors="coerce") <= k_rel).astype(int)
            out["is_relevant_actual"] = (pd.to_numeric(out["actual_rank"], errors="coerce") <= k_rel).astype(int)
            out = out[(out["is_relevant_pred"] == 1) | (out["is_relevant_actual"] == 1)].copy()
            if out.empty:
                continue

            eval_k = out[out["is_relevant_actual"] == 1].copy()
            if eval_k.empty:
                continue

            metrics.append(
                dict(
                    position=pos,
                    test_year=test_year,
                    train_start=train_start,
                    train_end=train_end,
                    k_features=int(k_pos),
                    n_ensemble=int(n_ens),
                    relevant_topk=int(k_rel),
                    relevant_weight=float(args.relevant_weight),
                    train_pool_cap=int(cap),
                    rank_mae_all=float(mean_absolute_error(eval_k["actual_rank"], eval_k["pred_rank"])),
                    points_mae_all=float(mean_absolute_error(eval_k["actual_points"], eval_k["pred_points"])),
                    n_test_players=int(len(eval_k)),
                )
            )
            preds.extend(out.to_dict(orient="records"))

        # final model fit through 2024 (features 2025 -> predict 2026)
        final_train_end = int(args.test_end) - 1
        final = pos_df[(pos_df[cols.season] >= train_start) & (pos_df[cols.season] <= final_train_end)].copy()
        cap = int(train_pool_cap.get(pos, 9999))
        final = final[pd.to_numeric(final["target_next_rank"], errors="coerce") <= cap].copy()
        if final.empty:
            continue
        # select features on full window for the final model
        k_pos = _k_features_for_pos(int(args.k_features), pos)
        num_feats = _select_topk_numeric_features(final, num_candidates, k_pos)
        Xf = final[num_feats + cat_feats]
        yf = pd.to_numeric(final["target_next_points"], errors="coerce")
        pre = _make_preprocessor(num_feats, cat_feats)
        Xf_m = pre.fit_transform(Xf)
        k_rel = int(rel_topk.get(pos, 60))
        sw = np.where(pd.to_numeric(final["target_next_rank"], errors="coerce").to_numpy() <= k_rel, float(args.relevant_weight), 1.0)

        # Fit relevance models on final window and append p(relevant) as an extra feature
        if args.use_relevance_model:
            y_rel = (pd.to_numeric(final["target_next_rank"], errors="coerce").to_numpy() <= k_rel).astype(int)
            rel_models = []
            rel_preds_train = []
            for seed in range(max(1, int(args.n_ensemble))):
                rm = _make_relevance_model(position=pos, random_state=7_000 + seed)
                rm.fit(Xf_m, y_rel, sample_weight=sw)
                rel_models.append(rm)
                rel_preds_train.append(rm.predict_proba(Xf_m)[:, 1])
            p_rel_train = np.mean(np.vstack(rel_preds_train), axis=0)
            Xf_m = hstack([Xf_m, csr_matrix(p_rel_train.reshape(-1, 1))]).tocsr()
        else:
            rel_models = None

        n_ens = max(1, int(args.n_ensemble))
        models = []
        for seed in range(n_ens):
            model = _make_model(position=pos, random_state=42 + seed, model_mode=str(args.model_mode))
            if isinstance(model, XGBRanker):
                fg = final[["target_season"]].assign(_idx=np.arange(len(final))).sort_values("target_season")
                forder = fg["_idx"].to_numpy()
                group_sizes = fg.groupby("target_season").size().to_list()
                model.fit(Xf_m[forder], yf.to_numpy()[forder], group=group_sizes)
            else:
                model.fit(Xf_m, yf.to_numpy(), sample_weight=sw)
            models.append(model)
        joblib.dump(
            dict(
                pre=pre,
                models=models,
                rel_models=rel_models,
                num_feats=num_feats,
                cat_feats=cat_feats,
                position=pos,
                k_features=int(k_pos),
                n_ensemble=int(n_ens),
                relevant_topk=int(k_rel),
                use_relevance_model=bool(args.use_relevance_model),
            ),
            models_dir / f"ffboost_model_{pos}.joblib",
        )

    pd.DataFrame(metrics).to_csv(out_dir / "backtest_metrics_ffboost.csv", index=False)
    pd.DataFrame(preds).to_csv(out_dir / "backtest_predictions_ffboost.csv", index=False)
    print(f"Wrote: {out_dir / 'backtest_metrics_ffboost.csv'}")
    print(f"Wrote: {out_dir / 'backtest_predictions_ffboost.csv'}")


if __name__ == "__main__":
    main()

