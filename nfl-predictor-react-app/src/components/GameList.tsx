import type { GamePrediction } from "../types";
import TeamBadge from "./TeamBadge";

type GameListProps = {
  games: GamePrediction[];
};

export default function GameList({ games }: GameListProps) {
  return (
    <div className="game-list-container">
      {!games.length ? (
        <p className="text-light m-0">Select a week to see predictions.</p>
      ) : (
        <ul className="list-group game-list">
          {games.map((g) => (
            <li key={g.game_id} className="list-group-item game-item">
              <div className="d-flex justify-content-between align-items-center">
                {/* Left side: matchup */}
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
                </div>

                {/* Right side: probabilities */}
                <div className="text-end">
                  <div className="fw-semibold pct">
                    {Number.isFinite(g.home_win_prob)
                      ? `${(g.home_win_prob * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                  <div className="small opacity-75">
                    Conf:{" "}
                    {Number.isFinite(g.confidence)
                      ? `${(g.confidence * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
