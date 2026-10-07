"""
Leakage-safe supplemental features merged onto player-season rows.

Sources (files under nfl_data/):
  - team_elo_2010_2025.csv — end-of-regular-season Elo per team
  - nflverse_supplemental/nflverse_schedules_all.csv — rest days between games
  - nflverse_supplemental/nflverse_draft_picks.csv — rookie draft round/pick (skill positions)

All joins use only information available in that calendar season (draft for rookies;
team Elo / schedule summarize the season that already occurred for the row).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _norm_team_abbr(t: str) -> str:
    """Align dataset team codes with team_elo / schedules."""
    t = str(t).upper().strip()
    # nflverse schedules + elo use LA for Rams; some datasets use LAR
    if t == "LAR":
        return "LA"
    return t


def _team_season_end_elo(elo_df: pd.DataFrame) -> pd.DataFrame:
    """Last REG game post-Elo per team-season."""
    e = elo_df[elo_df["game_type"].astype(str).str.upper() == "REG"].copy()
    if e.empty:
        return pd.DataFrame(columns=["season", "team", "team_elo_end_season"])
    rows = []
    for _, r in e.iterrows():
        rows.append(
            {
                "season": int(r["season"]),
                "team": _norm_team_abbr(r["home_team"]),
                "week": int(r["week"]),
                "elo": float(r["elo_home_post"]),
            }
        )
        rows.append(
            {
                "season": int(r["season"]),
                "team": _norm_team_abbr(r["away_team"]),
                "week": int(r["week"]),
                "elo": float(r["elo_away_post"]),
            }
        )
    t = pd.DataFrame(rows)
    t = t.sort_values(["season", "team", "week"])
    return t.groupby(["season", "team"], as_index=False).last().rename(columns={"elo": "team_elo_end_season"})[
        ["season", "team", "team_elo_end_season"]
    ]


def _team_season_schedule_rest(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Mean rest (days) between games and REG game count per team-season."""
    s = sched_df[sched_df["game_type"].astype(str).str.upper() == "REG"].copy()
    if s.empty:
        return pd.DataFrame(columns=["season", "team", "team_avg_rest_reg", "team_reg_games"])
    rows = []
    for _, r in s.iterrows():
        ar = pd.to_numeric(r.get("away_rest"), errors="coerce")
        hr = pd.to_numeric(r.get("home_rest"), errors="coerce")
        rows.append(
            {
                "season": int(r["season"]),
                "team": _norm_team_abbr(r["away_team"]),
                "rest": float(ar) if np.isfinite(ar) else np.nan,
            }
        )
        rows.append(
            {
                "season": int(r["season"]),
                "team": _norm_team_abbr(r["home_team"]),
                "rest": float(hr) if np.isfinite(hr) else np.nan,
            }
        )
    t = pd.DataFrame(rows)
    g = t.groupby(["season", "team"], as_index=False).agg(
        team_avg_rest_reg=("rest", "mean"),
        team_reg_games=("rest", "count"),
    )
    return g


def _draft_skill_tables(draft_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rookie-year draft row + first-ever draft row per player."""
    dp = draft_df.copy()
    dp["pos"] = dp["position"].astype(str).str.upper().str.strip()
    dp = dp[dp["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    dp["gsis_id"] = dp["gsis_id"].astype(str).str.strip()
    dp = dp[dp["gsis_id"].str.startswith("00-", na=False)].copy()
    dp["season"] = pd.to_numeric(dp["season"], errors="coerce")
    dp["round"] = pd.to_numeric(dp["round"], errors="coerce")
    dp["pick"] = pd.to_numeric(dp["pick"], errors="coerce")
    dp = dp.dropna(subset=["season", "gsis_id", "round"])

    rookie = dp[["season", "gsis_id", "round", "pick"]].rename(
        columns={
            "season": "join_season",
            "round": "rookie_draft_round",
            "pick": "rookie_draft_pick_in_round",
        }
    )
    first = dp.sort_values(["gsis_id", "season"]).groupby("gsis_id", as_index=False).first()
    career = first[["gsis_id", "round", "pick"]].rename(
        columns={"round": "career_draft_round", "pick": "career_draft_pick_in_round"}
    )
    return rookie, career


def merge_supplemental_features(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    team_col: str = "recent_team",
    player_id_col: str = "player_id",
    nfl_data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Left-merge supplemental numeric features. Safe to call multiple times; skips missing files.
    """
    out = df.copy()
    base = Path(nfl_data_dir) if nfl_data_dir else Path(__file__).resolve().parent.parent / "nfl_data"
    supp = base / "nflverse_supplemental"
    elo_path = base / "team_elo_2010_2025.csv"
    if not elo_path.exists():
        elo_path = base / "team_elo_2010_2025.csv"

    # --- Team Elo + schedule ---
    try:
        out[season_col] = pd.to_numeric(out[season_col], errors="coerce")
        out["_team_k"] = out[team_col].map(_norm_team_abbr)
        if elo_path.exists():
            elo_raw = pd.read_csv(elo_path)
            te = _team_season_end_elo(elo_raw).rename(columns={"season": "_elo_season", "team": "_elo_team"})
            out = out.merge(
                te,
                left_on=[season_col, "_team_k"],
                right_on=["_elo_season", "_elo_team"],
                how="left",
            ).drop(columns=["_elo_season", "_elo_team"], errors="ignore")
        if supp.exists():
            sp = supp / "nflverse_schedules_all.csv"
            if sp.exists():
                sched = pd.read_csv(sp)
                tr = _team_season_schedule_rest(sched).rename(
                    columns={"season": "_sched_season", "team": "_sched_team"}
                )
                out = out.merge(
                    tr,
                    left_on=[season_col, "_team_k"],
                    right_on=["_sched_season", "_sched_team"],
                    how="left",
                ).drop(columns=["_sched_season", "_sched_team"], errors="ignore")
    except Exception:
        pass

    # --- Draft (skill positions) ---
    try:
        dp_path = supp / "nflverse_draft_picks.csv"
        if dp_path.exists():
            dp = pd.read_csv(dp_path)
            rookie, career = _draft_skill_tables(dp)
            out[player_id_col] = out[player_id_col].astype(str).str.strip()
            out = out.merge(
                rookie,
                left_on=[player_id_col, season_col],
                right_on=["gsis_id", "join_season"],
                how="left",
                suffixes=("", "_rook"),
            )
            out = out.drop(columns=["gsis_id", "join_season"], errors="ignore")
            out = out.merge(career, left_on=player_id_col, right_on="gsis_id", how="left")
            out = out.drop(columns=["gsis_id"], errors="ignore")
    except Exception:
        pass

    out = out.drop(columns=["_team_k"], errors="ignore")
    return out


def merge_supplemental_features_optional(
    df: pd.DataFrame,
    *,
    enabled: bool,
    season_col: str = "season",
    team_col: str = "recent_team",
    player_id_col: str = "player_id",
    nfl_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if not enabled:
        return df
    return merge_supplemental_features(
        df,
        season_col=season_col,
        team_col=team_col,
        player_id_col=player_id_col,
        nfl_data_dir=nfl_data_dir,
    )
