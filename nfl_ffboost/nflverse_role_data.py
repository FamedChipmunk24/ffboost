from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnapFeatureConfig:
    seasons: tuple[int, ...] = tuple(range(2012, 2026))  # snap_counts availability (PFR) in nflverse
    cache_players_csv: str = "nflverse_players.csv"
    cache_snaps_csv: str = "nflverse_snap_player_season_2012_2025.csv"


def _download_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def _players_map(cache_dir: Path, cfg: SnapFeatureConfig) -> pd.DataFrame:
    """
    Returns mapping between PFR id and nflverse/GSIS id.
    Columns: pfr_id, gsis_id
    """
    cache_path = cache_dir / cfg.cache_players_csv
    if cache_path.exists():
        players = pd.read_csv(cache_path)
    else:
        url = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
        players = _download_csv(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        players.to_csv(cache_path, index=False)
    players = players.rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"})
    players["pfr_player_id"] = players["pfr_player_id"].astype(str)
    players["player_id"] = players["player_id"].astype(str)
    return players[["pfr_player_id", "player_id"]].dropna().drop_duplicates()


def _load_snap_counts_for_season(season: int) -> pd.DataFrame:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{int(season)}.csv.gz"
    df = pd.read_csv(url, compression="gzip")
    return df


def _aggregate_snap_counts(snaps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates game-level snap counts to player-season.

    Output columns:
      - season
      - pfr_player_id
      - snaps_offense, snaps_defense, snaps_st
      - snap_offense_pct, snap_defense_pct, snap_st_pct  (weighted by snaps for offense/defense, mean for ST)
    """
    out = snaps.copy()
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce")
    out["pfr_player_id"] = out.get("pfr_player_id")
    for c in ["offense_snaps", "defense_snaps", "st_snaps", "offense_pct", "defense_pct", "st_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["season", "pfr_player_id"]).copy()
    out["pfr_player_id"] = out["pfr_player_id"].astype(str)

    def _wavg(val: pd.Series, w: pd.Series) -> float:
        w = pd.to_numeric(w, errors="coerce").fillna(0.0)
        val = pd.to_numeric(val, errors="coerce")
        s = float(w.sum())
        if s <= 0:
            return float(np.nan)
        return float((val * w).sum() / s)

    g = out.groupby(["season", "pfr_player_id"], dropna=False)
    agg = g.agg(
        snaps_offense=("offense_snaps", "sum"),
        snaps_defense=("defense_snaps", "sum"),
        snaps_st=("st_snaps", "sum"),
    ).reset_index()
    # Weighted percentages
    pct = g.apply(
        lambda d: pd.Series(
            {
                "snap_offense_pct": _wavg(d.get("offense_pct"), d.get("offense_snaps")),
                "snap_defense_pct": _wavg(d.get("defense_pct"), d.get("defense_snaps")),
                "snap_st_pct": _wavg(d.get("st_pct"), d.get("st_snaps")),
            }
        )
    ).reset_index()
    merged = agg.merge(pct, on=["season", "pfr_player_id"], how="left")
    return merged


def load_or_build_player_season_snaps(
    cache_dir: str | Path,
    seasons: Iterable[int] | None = None,
    snaps_csv: str | Path | None = None,
    cfg: SnapFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Returns player-season snap features keyed by nflverse `player_id` and `season`.

    - Uses open nflverse `snap_counts` (PFR snap counts) by season.
    - Joins snap counts `pfr_player_id` -> dataset `player_id` via nflverse `players.csv` mapping.

    Notes:
    - Snap counts are available (in nflverse) from 2012 onward.
    - This does NOT provide true "route participation" (paid sources usually required),
      but offensive snap share is a strong proxy for WR/TE usage and RB involvement.
    """
    cfg = cfg or SnapFeatureConfig()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if snaps_csv is None:
        snaps_csv = cache_dir / cfg.cache_snaps_csv
    else:
        snaps_csv = Path(snaps_csv)

    if Path(snaps_csv).exists():
        feats = pd.read_csv(snaps_csv)
        feats["season"] = pd.to_numeric(feats["season"], errors="coerce")
        feats["player_id"] = feats["player_id"].astype(str)
        return feats

    seasons2 = tuple(int(s) for s in (seasons if seasons is not None else cfg.seasons))
    seasons2 = tuple([s for s in seasons2 if 2012 <= s <= 2025])

    pmap = _players_map(cache_dir, cfg)

    parts = []
    for s in seasons2:
        try:
            raw = _load_snap_counts_for_season(s)
        except Exception:
            continue
        agg = _aggregate_snap_counts(raw)
        parts.append(agg)
    if not parts:
        return pd.DataFrame(columns=["season", "player_id"])

    snaps_agg = pd.concat(parts, ignore_index=True)
    snaps_agg = snaps_agg.merge(pmap, on="pfr_player_id", how="left")
    snaps_agg = snaps_agg.dropna(subset=["player_id"]).copy()
    snaps_agg["season"] = pd.to_numeric(snaps_agg["season"], errors="coerce")
    snaps_agg["player_id"] = snaps_agg["player_id"].astype(str)

    keep = [
        "season",
        "player_id",
        "snaps_offense",
        "snaps_defense",
        "snaps_st",
        "snap_offense_pct",
        "snap_defense_pct",
        "snap_st_pct",
    ]
    snaps_agg = snaps_agg[keep].drop_duplicates(subset=["season", "player_id"])

    Path(snaps_csv).parent.mkdir(parents=True, exist_ok=True)
    snaps_agg.to_csv(snaps_csv, index=False)
    return snaps_agg


def merge_player_season_snaps(
    df: pd.DataFrame,
    cache_dir: str | Path,
    season_col: str = "season",
    player_id_col: str = "player_id",
    snaps_csv: str | Path | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if player_id_col not in out.columns or season_col not in out.columns:
        return out
    feats = load_or_build_player_season_snaps(cache_dir=cache_dir, snaps_csv=snaps_csv)
    if feats.empty:
        return out
    out[season_col] = pd.to_numeric(out[season_col], errors="coerce")
    out[player_id_col] = out[player_id_col].astype(str)
    feats["season"] = pd.to_numeric(feats["season"], errors="coerce")
    feats["player_id"] = feats["player_id"].astype(str)
    return out.merge(
        feats,
        left_on=[season_col, player_id_col],
        right_on=["season", "player_id"],
        how="left",
        suffixes=("", "_snap"),
    ).drop(columns=["season_snap", "player_id_snap"], errors="ignore")

