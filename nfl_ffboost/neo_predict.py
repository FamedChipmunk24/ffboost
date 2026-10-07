from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack  # type: ignore

from nfl_ffboost.nflverse_role_data import merge_player_season_snaps
from nfl_ffboost.supplemental_merge import merge_supplemental_features_optional


DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]
RELEVANT_TOPK_BY_POS = {"QB": 24, "RB": 60, "WR": 80, "TE": 36}


def _name_key(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.lower().str.strip()
    x = x.str.replace(r"[^\w\s]", "", regex=True)
    x = x.str.replace(r"\b(jr|sr|iii|ii|iv)\b", "", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True)
    return x.str.strip()


def _load_adp(adp_csv: str) -> pd.DataFrame:
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


@dataclass(frozen=True)
class Cols:
    season: str = "season"
    player: str = "player_display_name"
    position: str = "position"
    team: str = "recent_team"
    points: str = "fantasy_points_ppr"


def _add_similar_trend_features(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    """
    Build the same similar-player cohort features as training, but for prediction.
    For predict season 2026, we compute features for rows from season 2025 using
    historical cohorts with target_season < 2026.
    """
    out = df.copy()
    out[cols.position] = out[cols.position].astype(str).str.upper().str.strip()
    out[cols.season] = pd.to_numeric(out[cols.season], errors="coerce")
    out["target_season"] = pd.to_numeric(out.get("target_season"), errors="coerce")
    out[cols.points] = pd.to_numeric(out[cols.points], errors="coerce")
    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")

    attempts = pd.to_numeric(out.get("attempts"), errors="coerce")
    carries = pd.to_numeric(out.get("carries"), errors="coerce")
    targets = pd.to_numeric(out.get("targets"), errors="coerce")

    vol = pd.Series(np.nan, index=out.index, dtype=float)
    pos = out[cols.position]
    vol = np.where(pos == "QB", attempts, vol)
    vol = np.where(pos != "QB", carries.fillna(0) + targets.fillna(0), vol)
    out["_vol_proxy"] = pd.to_numeric(vol, errors="coerce")

    out["_age_bin"] = pd.cut(out["age"], bins=[0, 22, 24, 26, 28, 30, 32, 60], labels=False, include_lowest=True)
    out["_pts_bin"] = pd.cut(out[cols.points], bins=[-1, 80, 120, 160, 200, 240, 280, 340, 600], labels=False, include_lowest=True)
    out["_vol_bin"] = pd.cut(out["_vol_proxy"], bins=[-1, 30, 60, 100, 140, 200, 300, 500, 2000], labels=False, include_lowest=True)

    out["target_next_points"] = pd.to_numeric(out.get("target_next_points"), errors="coerce")
    out["_next_delta"] = out["target_next_points"] - out[cols.points]
    out["_bust"] = (out["target_next_points"] < (0.8 * out[cols.points])).astype(float)

    keys = [cols.position, "_age_bin", "_pts_bin", "_vol_bin"]
    out["sim_next_points_mean"] = np.nan
    out["sim_next_delta_mean"] = np.nan
    out["sim_bust_rate"] = np.nan

    # For each target season, merge in cohort averages from earlier target seasons
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


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    out = pd.to_numeric(a, errors="coerce") / b2
    return out.replace([np.inf, -np.inf], np.nan)


def _build_neo_features(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    out = df.copy()
    out[cols.season] = pd.to_numeric(out[cols.season], errors="coerce").astype("Int64")
    out[cols.position] = out[cols.position].astype(str).str.upper().str.strip()
    out[cols.player] = out[cols.player].astype(str).str.strip()
    out[cols.team] = out[cols.team].astype(str).str.upper().str.strip()
    out[cols.points] = pd.to_numeric(out[cols.points], errors="coerce")

    for c in ["age", "games", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
              "carries", "rushing_yards", "rushing_tds",
              "targets", "receptions", "receiving_yards", "receiving_tds"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--predict_season", type=int, default=2026)
    ap.add_argument("--positions", default=",".join(DEFAULT_POSITIONS))
    ap.add_argument(
        "--skip-adp",
        dest="skip_adp",
        action="store_true",
        help="Disable preseason ADP merge (default: merge ADP for predict_season when available).",
    )
    ap.add_argument("--adp_csv", default=str(Path(__file__).resolve().parent.parent / "nfl_data" / "fantasydata_adp_2010_2025.csv"))
    ap.add_argument(
        "--skip-snaps",
        dest="skip_snaps",
        action="store_true",
        help="Disable nflverse snap features. Default: merge snaps.",
    )
    ap.add_argument(
        "--skip-supplemental",
        dest="skip_supplemental",
        action="store_true",
        help="Disable supplemental draft/Elo/schedule features.",
    )
    ap.add_argument(
        "--snaps_csv",
        default=str(Path(__file__).resolve().parent.parent / "nfl_data" / "nflverse_snap_player_season_2012_2025.csv"),
        help="Cache path for aggregated player-season snap features.",
    )
    ap.add_argument("--relevant_only", action="store_true")
    ap.add_argument("--relevant_topk_qb", type=int, default=RELEVANT_TOPK_BY_POS["QB"])
    ap.add_argument("--relevant_topk_rb", type=int, default=RELEVANT_TOPK_BY_POS["RB"])
    ap.add_argument("--relevant_topk_wr", type=int, default=RELEVANT_TOPK_BY_POS["WR"])
    ap.add_argument("--relevant_topk_te", type=int, default=RELEVANT_TOPK_BY_POS["TE"])
    args = ap.parse_args()

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

    # Build target_next_points for historical cohort feature computation
    df = df.dropna(subset=[cols.season, cols.player, cols.position, cols.points]).copy()
    df[cols.season] = pd.to_numeric(df[cols.season], errors="coerce").astype(int)
    df = df.sort_values([cols.player, cols.position, cols.season])
    grp = df.groupby([cols.player, cols.position], sort=False)
    df["target_season"] = df[cols.season] + 1
    df["target_next_points"] = pd.to_numeric(grp[cols.points].shift(-1), errors="coerce")

    # Similar-player cohort features (leakage-safe)
    df = _add_similar_trend_features(df, cols)

    base = df[df[cols.season] == (int(args.predict_season) - 1)].copy()
    base[cols.season] = int(args.predict_season)
    base["target_season"] = int(args.predict_season)

    # Apply current-season team assignments (FA / trades) when the override CSV exists.
    overrides_path = nfl_data_dir / f"roster_overrides_{int(args.predict_season)}.csv"
    if overrides_path.exists():
        try:
            ov = pd.read_csv(overrides_path)
            if {"player", "team"}.issubset(ov.columns):
                ov["player_key"] = _name_key(ov["player"])
                ov["team"] = ov["team"].astype(str).str.upper().str.strip()
                if "position" in ov.columns:
                    ov["position"] = ov["position"].astype(str).str.upper().str.strip()
                    ov = ov.dropna(subset=["player_key", "position", "team"]).drop_duplicates(
                        ["player_key", "position"]
                    )
                    base["player_key"] = _name_key(base[cols.player])
                    base = base.merge(
                        ov[["player_key", "position", "team"]].rename(columns={"team": "_team_override"}),
                        left_on=["player_key", cols.position],
                        right_on=["player_key", "position"],
                        how="left",
                        suffixes=("", "_ov"),
                    )
                else:
                    ov = ov.dropna(subset=["player_key", "team"]).drop_duplicates("player_key")
                    base["player_key"] = _name_key(base[cols.player])
                    base = base.merge(
                        ov[["player_key", "team"]].rename(columns={"team": "_team_override"}),
                        on="player_key",
                        how="left",
                    )
                mask = base["_team_override"].notna()
                base.loc[mask, cols.team] = base.loc[mask, "_team_override"]
                base = base.drop(columns=["_team_override", "position_ov"], errors="ignore")

                # Refresh team-context features (Elo / schedule rest) so movers get
                # their NEW team's context instead of last year's club.
                from nfl_ffboost.supplemental_merge import (
                    _norm_team_abbr,
                    _team_season_end_elo,
                    _team_season_schedule_rest,
                )

                prior_season = int(args.predict_season) - 1
                base["_team_k"] = base[cols.team].map(_norm_team_abbr)
                elo_path = nfl_data_dir / "team_elo_2010_2025.csv"
                if elo_path.exists() and "team_elo_end_season" in base.columns:
                    te = _team_season_end_elo(pd.read_csv(elo_path))
                    te = te[te["season"] == prior_season]
                    base["team_elo_end_season"] = base["_team_k"].map(
                        dict(zip(te["team"], te["team_elo_end_season"]))
                    )
                sched_path = nfl_data_dir / "nflverse_supplemental" / "nflverse_schedules_all.csv"
                if sched_path.exists() and "team_avg_rest_reg" in base.columns:
                    tr = _team_season_schedule_rest(pd.read_csv(sched_path))
                    tr = tr[tr["season"] == prior_season]
                    base["team_avg_rest_reg"] = base["_team_k"].map(
                        dict(zip(tr["team"], tr["team_avg_rest_reg"]))
                    )
                    if "team_reg_games" in base.columns:
                        base["team_reg_games"] = base["_team_k"].map(
                            dict(zip(tr["team"], tr["team_reg_games"]))
                        )
                base = base.drop(columns=["_team_k"], errors="ignore")
        except Exception:
            pass

    # Attach preseason ADP for predict_season, if available
    if (not bool(args.skip_adp)) and args.adp_csv:
        try:
            adp = _load_adp(args.adp_csv)
            base["player_key"] = _name_key(base[cols.player])
            base = base.merge(
                adp,
                left_on=[cols.season, cols.position, "player_key"],
                right_on=["season", "position", "player_key"],
                how="left",
                suffixes=("", "_adp"),
            )
            # drop duplicate season column from adp side
            if "season_adp" in base.columns:
                base = base.drop(columns=["season_adp"])
        except Exception:
            pass

    models_dir = Path(args.models_dir)
    rows: List[Dict[str, object]] = []
    rel_topk = {
        "QB": int(args.relevant_topk_qb),
        "RB": int(args.relevant_topk_rb),
        "WR": int(args.relevant_topk_wr),
        "TE": int(args.relevant_topk_te),
    }

    for pos in positions:
        model_path = models_dir / f"ffboost_model_{pos}.joblib"
        if not model_path.exists():
            continue
        payload = joblib.load(model_path)
        pre = payload["pre"]
        models = payload.get("models")
        rel_models = payload.get("rel_models")
        model = payload.get("model")  # backward-compat
        num_feats = payload["num_feats"]
        cat_feats = payload["cat_feats"]
        use_rel = bool(payload.get("use_relevance_model", False))

        g = base[base[cols.position] == pos].copy()
        if g.empty:
            continue
        X = pre.transform(g[num_feats + cat_feats])
        p_rel = None
        if use_rel and rel_models:
            rel_ps = [rm.predict_proba(X)[:, 1] for rm in rel_models]
            p_rel = np.mean(np.vstack(rel_ps), axis=0)
            X = hstack([X, csr_matrix(p_rel.reshape(-1, 1))])
        if models:
            preds = [m.predict(X) for m in models]
            score = np.mean(np.vstack(preds), axis=0)
        else:
            score = model.predict(X)
        out = pd.DataFrame(
            {
                "season": int(args.predict_season),
                "player": g[cols.player].astype(str).values,
                "position": pos,
                "team": g[cols.team].astype(str).values,
                "pred_rank_score": score,
            }
        )
        if p_rel is not None:
            out["p_relevant"] = p_rel
        out["pred_rank"] = out.groupby(["season", "position"])["pred_rank_score"].rank(method="first", ascending=False).astype(int)
        # For display: simple points proxy by rank using prior-year position points distribution
        hist = raw[(raw[cols.position].astype(str).str.upper() == pos) & (raw[cols.season] == (int(args.predict_season) - 1))].copy()
        hist[cols.points] = pd.to_numeric(hist[cols.points], errors="coerce")
        hist = hist.dropna(subset=[cols.points])
        hist["rank"] = hist[cols.points].rank(method="first", ascending=False).astype(int)
        by_rank = hist.groupby("rank")[cols.points].mean().sort_index()
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
            out["pred_points"] = out["pred_rank"].clip(1, max_rank).map(filled).astype(float)
        rows.extend(out.to_dict(orient="records"))

    out_df = pd.DataFrame(rows)
    if args.relevant_only and not out_df.empty:
        keep = []
        for pos in positions:
            k = int(rel_topk.get(pos, 60))
            g = out_df[out_df["position"] == pos].copy()
            if g.empty:
                continue
            keep.append(g[g["pred_rank"] <= k])
        out_df = pd.concat(keep, ignore_index=True) if keep else out_df.iloc[0:0]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

