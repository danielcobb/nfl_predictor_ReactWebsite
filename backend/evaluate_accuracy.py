"""
Evaluate prediction accuracy by comparing stored predictions against actual NFL results.

Usage:
    python backend/evaluate_accuracy.py 2025
    python backend/evaluate_accuracy.py 2025 --backfill   # also writes actual_winner to DB
"""
from __future__ import annotations
import sys
import sqlite3
from main import get_schedule, load_predictions, ensure_db

DB_PATH = "predictions.db"


def get_actual_winners(season: int) -> dict[tuple[int, str, str], str]:
    """Return {(week, home_team, away_team): winner} from the NFL schedule."""
    schedule = get_schedule(season)
    winners = {}
    for _, g in schedule.iterrows():
        hs, aws = g.get("home_score"), g.get("away_score")
        if hs is None or aws is None:
            continue
        try:
            hs, aws = float(hs), float(aws)
        except (TypeError, ValueError):
            continue
        if hs != hs or aws != aws:  # NaN check
            continue
        winner = str(g["home_team"]) if hs > aws else str(g["away_team"])
        winners[(int(g["week"]), str(g["home_team"]), str(g["away_team"]))] = winner
    return winners


def backfill_actual_winners(db_path: str, season: int, winners: dict) -> int:
    """Write actual_winner into prediction rows. Returns rows updated."""
    ensure_db(db_path)
    updated = 0
    with sqlite3.connect(db_path) as conn:
        for (week, home, away), winner in winners.items():
            cur = conn.execute(
                """UPDATE predictions SET actual_winner = ?
                   WHERE season = ? AND week = ? AND home_team = ? AND away_team = ?
                   AND (actual_winner IS NULL OR actual_winner != ?)""",
                (winner, season, week, home, away, winner),
            )
            updated += cur.rowcount
        conn.commit()
    return updated


def evaluate(season: int, backfill: bool = False) -> None:
    winners = get_actual_winners(season)
    if not winners:
        print(f"No actual results found for {season}.")
        return

    if backfill:
        n = backfill_actual_winners(DB_PATH, season, winners)
        print(f"Backfilled {n} rows with actual_winner.\n")

    total, correct = 0, 0
    week_stats: dict[int, dict] = {}

    print(f"{'Wk':>3}  {'Matchup':<15} {'Predicted':<6} {'Actual':<6} {'Result'}")
    print("-" * 55)

    for week in range(1, 23):
        preds = load_predictions(DB_PATH, season, week)
        if not preds:
            continue
        for p in preds:
            key = (week, p["home_team"], p["away_team"])
            actual = winners.get(key)
            if actual is None:
                continue
            hit = p["predicted_winner"] == actual
            total += 1
            correct += int(hit)
            ws = week_stats.setdefault(week, {"correct": 0, "total": 0})
            ws["total"] += 1
            ws["correct"] += int(hit)

            mark = "OK" if hit else "MISS"
            matchup = f"{p['away_team']}@{p['home_team']}"
            print(f"{week:>3}  {matchup:<15} {p['predicted_winner']:<6} {actual:<6} {mark}")

    if total == 0:
        print("\nNo predictions matched actual results.")
        return

    print(f"\n{'='*55}")
    print(f"Overall: {correct}/{total} = {correct/total:.1%}\n")
    print(f"{'Week':>4}  {'Correct':>7} / {'Total':<5}  {'Accuracy':>8}")
    print("-" * 35)
    for w in sorted(week_stats):
        s = week_stats[w]
        print(f"{w:>4}  {s['correct']:>7} / {s['total']:<5}  {s['correct']/s['total']:>8.1%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_accuracy.py <season> [--backfill]")
        sys.exit(1)
    season = int(sys.argv[1])
    backfill = "--backfill" in sys.argv
    evaluate(season, backfill=backfill)
