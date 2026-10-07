#!/usr/bin/env python3
"""
Aggregate scored fire events across nearby stations and produce a vote-based summary.

Inputs: outputs/{SITE_ID}/{SITE_ID}_pm_fire_events_scored.csv for each station.
Output: outputs/aggregated/nearby_group_events_voted.csv with columns:
    - group_id
    - group_start (UTC)
    - group_end (UTC)
    - duration_min
    - vote_count
    - stations (comma-separated)
    - first_station
    - first_station_start (UTC)
    - event_count_per_station (JSON-like string)

Usage:
    python aggregate_fire_events.py \
        --stations aa-44-16-29 aa-44-16-33 aa-44-12-15 aa-44-13-97 aa-44-17-41 \
        --tolerance "2min" \
        --outputs_dir outputs

Notes:
 - Two events (from different stations) are considered the same candidate if their
   time windows overlap within the specified tolerance. Specifically, for intervals
   [start_i, end_i] and [start_j, end_j], we expand both sides by tolerance and
   check interval intersection.
 - Timestamps are treated as UTC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fire events across stations with voting")
    parser.add_argument(
        "--stations",
        nargs="*",
        default=[
            "aa-44-16-29",
            "aa-44-16-33",
            "aa-44-12-15",
            "aa-44-13-97",
            "aa-44-17-41",
        ],
        help="List of station IDs to include",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Base outputs directory that contains per-site subfolders",
    )
    parser.add_argument(
        "--tolerance",
        type=str,
        default="2min",
        help="Time tolerance for event overlap (e.g., '2min', '30s')",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="nearby_group_events_voted.csv",
        help="Output CSV filename under outputs/aggregated",
    )
    return parser.parse_args()


def read_site_events(outputs_dir: Path, site_id: str) -> pd.DataFrame:
    csv_path = outputs_dir / site_id / f"{site_id}_pm_fire_events_scored.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["start", "end", "score", "score_blend"])  # empty
    df = pd.read_csv(csv_path)
    # Robust parsing of timestamps
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True)
    # Drop invalid rows
    df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
    df["station"] = site_id
    return df


def intervals_overlap(a: Tuple[pd.Timestamp, pd.Timestamp],
                      b: Tuple[pd.Timestamp, pd.Timestamp],
                      tol: pd.Timedelta) -> bool:
    a0, a1 = a
    b0, b1 = b
    # Expand both intervals by tolerance and check intersection
    a0e, a1e = a0 - tol, a1 + tol
    b0e, b1e = b0 - tol, b1 + tol
    latest_start = max(a0e, b0e)
    earliest_end = min(a1e, b1e)
    return latest_start <= earliest_end


def cluster_events(all_events: pd.DataFrame, tolerance: pd.Timedelta) -> List[List[int]]:
    # Sort by start for deterministic clustering
    all_events = all_events.sort_values(["start", "end"]).reset_index(drop=True)
    clusters: List[List[int]] = []
    for idx, row in all_events.iterrows():
        placed = False
        for cl in clusters:
            # Compare with the cluster representative (first item) for speed;
            # fallback to any in cluster if needed
            rep = all_events.iloc[cl[0]]
            if intervals_overlap((row.start, row.end), (rep.start, rep.end), tolerance):
                cl.append(idx)
                placed = True
                break
            # If representative fails, try full cluster
            if not placed:
                for j in cl:
                    r2 = all_events.iloc[j]
                    if intervals_overlap((row.start, row.end), (r2.start, r2.end), tolerance):
                        cl.append(idx)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            clusters.append([idx])
    return clusters


def build_group_summary(all_events: pd.DataFrame, clusters: List[List[int]]) -> pd.DataFrame:
    rows = []
    for gid, cl in enumerate(clusters, start=1):
        sub = all_events.iloc[cl]
        group_start = sub["start"].min()
        group_end = sub["end"].max()
        duration_min = (group_end - group_start).total_seconds() / 60.0
        stations = sorted(sub["station"].unique().tolist())
        vote_count = len(stations)
        # First reporter: earliest start; if tie, lexicographically smallest station
        first_idx = sub["start"].idxmin()
        first_station = str(all_events.loc[first_idx, "station"])
        first_station_start = all_events.loc[first_idx, "start"]

        # Count events per station in this group
        counts = sub.groupby("station").size().to_dict()
        rows.append({
            "group_id": gid,
            "group_start": group_start,
            "group_end": group_end,
            "duration_min": round(duration_min, 3),
            "vote_count": int(vote_count),
            "stations": ",".join(stations),
            "first_station": first_station,
            "first_station_start": first_station_start,
            "event_count_per_station": json.dumps(counts, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    tolerance = pd.Timedelta(args.tolerance)

    # Read per-station scored events
    dfs = []
    for sid in args.stations:
        df = read_site_events(outputs_dir, sid)
        if not df.empty:
            dfs.append(df[["start", "end", "station"]].copy())
    if not dfs:
        print("No events found for the provided stations.")
        return

    all_events = pd.concat(dfs, ignore_index=True)

    # Cluster events across stations using overlap with tolerance
    clusters = cluster_events(all_events, tolerance)

    # Build summary table
    summary = build_group_summary(all_events, clusters)

    # Ensure output directory
    out_dir = outputs_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.summary_name
    summary.sort_values(["group_start", "group_id"]).to_csv(out_csv, index=False)
    print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()


