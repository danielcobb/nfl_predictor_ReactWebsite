import type { GamePrediction } from "../types";
import TeamBadge from "./TeamBadge";

type GameListProps = {
  games: GamePrediction[];
  loading?: boolean;
};

function confidenceLabel(conf: number): { label: string; cls: string } {
  if (conf > 0.7) return { label: "High", cls: "conf-high" };
  if (conf > 0.5) return { label: "Med", cls: "conf-med" };
  return { label: "Low", cls: "conf-low" };
}

function formatDate(dateStr?: string): string | null {
  if (!dateStr) return null;
  const d = new Date(dateStr + "T12:00:00"); // force noon to avoid timezone flip
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

function SkeletonCard() {
  return (
    <li className="list-group-item game-item skeleton-card">
      <div className="skeleton-line" style={{ width: "60%", height: "14px", marginBottom: "10px" }} />
      <div className="skeleton-line" style={{ width: "40%", height: "10px", marginBottom: "8px" }} />
      <div className="skeleton-line" style={{ width: "100%", height: "8px" }} />
    </li>
  );
}

export default function GameList({ games, loading }: GameListProps) {
  if (loading) {
    return (
      <div className="game-list-container">
        <ul className="list-group game-list">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div className="game-list-container">
      {!games.length ? (
        <p className="text-light m-0">Select a week to see predictions.</p>
      ) : (
        <ul className="list-group game-list">
          {games.map((g) => {
            const awayProb = 1 - g.home_win_prob;
            const conf = confidenceLabel(g.confidence);
            const date = formatDate(g.game_date);

            return (
              <li key={g.game_id} className="list-group-item game-item">
                {/* Top row: matchup + confidence badge */}
                <div className="d-flex justify-content-between align-items-start">
                  <div className="matchup-block">
                    <div className="matchup-row">
                      <div className="team-side away">
                        <TeamBadge abbr={g.away_team} />
                      </div>
                      <div className="at-symbol">@</div>
                      <div className="team-side home">
                        <TeamBadge abbr={g.home_team} />
                      </div>
                    </div>

                    <div className="predicted-winner">
                      Predicted Winner:{" "}
                      <span className="winner">{g.predicted_winner}</span>
                    </div>

                    {date && <div className="game-date">{date}</div>}
                  </div>

                  <span className={`conf-badge ${conf.cls}`}>{conf.label}</span>
                </div>

                {/* Win probability bar */}
                <div className="prob-bar-wrapper">
                  <div className="prob-bar-labels">
                    <span>{g.away_team} {(awayProb * 100).toFixed(0)}%</span>
                    <span>{g.home_team} {(g.home_win_prob * 100).toFixed(0)}%</span>
                  </div>
                  <div className="prob-bar">
                    <div
                      className="prob-bar-away"
                      style={{ width: `${awayProb * 100}%` }}
                    />
                    <div
                      className="prob-bar-home"
                      style={{ width: `${g.home_win_prob * 100}%` }}
                    />
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
