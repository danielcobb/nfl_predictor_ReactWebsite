import type { GamePrediction } from "../types";
import { TEAM_LOGOS } from "../teamLogos";

type GameListProps = {
  games: GamePrediction[];
  loading?: boolean;
};

function confidenceInfo(conf: number): { label: string; cls: string } {
  if (conf > 0.7) return { label: "High", cls: "conf-high" };
  if (conf > 0.5) return { label: "Med",  cls: "conf-med" };
  return { label: "Low", cls: "conf-low" };
}

function formatDate(dateStr?: string): string | null {
  if (!dateStr) return null;
  const d = new Date(dateStr + "T12:00:00");
  return d.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <div className="skeleton-line" style={{ height: 46, width: 46, borderRadius: 8, flexShrink: 0 }} />
        <div className="skeleton-line" style={{ height: 24, flex: 1 }} />
        <div className="skeleton-line" style={{ height: 46, width: 46, borderRadius: 8, flexShrink: 0 }} />
      </div>
      <div className="skeleton-line" style={{ height: 34, width: "100%" }} />
      <div className="skeleton-line" style={{ height: 5, width: "100%" }} />
    </div>
  );
}

export default function GameList({ games, loading }: GameListProps) {
  if (loading) {
    return (
      <div className="games-grid">
        {Array.from({ length: 6 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (!games.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🏈</div>
        <p>Select a week to see predictions</p>
      </div>
    );
  }

  return (
    <div className="games-grid">
      {games.map((g, idx) => {
        const awayProb = 1 - g.home_win_prob;
        const conf = confidenceInfo(g.confidence);
        const date = formatDate(g.game_date);
        const awayLogo = TEAM_LOGOS[g.away_team];
        const homeLogo = TEAM_LOGOS[g.home_team];

        return (
          <div
            key={g.game_id}
            className={`game-card ${conf.cls}`}
            style={{ animationDelay: `${idx * 0.045}s` }}
          >
            <div className="card-inner">
              {/* ── Matchup ─────────────────────────────── */}
              <div className="matchup-section">
                <div className="team-col away">
                  {awayLogo ? (
                    <img
                      className="team-logo-lg"
                      src={awayLogo}
                      alt={g.away_team}
                    />
                  ) : (
                    <div className="team-logo-placeholder" />
                  )}
                  <span className="team-name-lg">{g.away_team}</span>
                </div>

                <div className="matchup-divider">
                  <span className="vs-text">@</span>
                  {date && (
                    <span className="game-date-center">{date}</span>
                  )}
                </div>

                <div className="team-col home">
                  {homeLogo ? (
                    <img
                      className="team-logo-lg"
                      src={homeLogo}
                      alt={g.home_team}
                    />
                  ) : (
                    <div className="team-logo-placeholder" />
                  )}
                  <span className="team-name-lg">{g.home_team}</span>
                </div>
              </div>

              {/* ── Winner Callout ───────────────────────── */}
              <div className="winner-callout">
                <span className="winner-label">Pick</span>
                <span className="winner-team">{g.predicted_winner}</span>
                <span className="winner-spacer" />
                <span className={`conf-badge ${conf.cls}`}>{conf.label}</span>
              </div>

              {/* ── Probability Bar ──────────────────────── */}
              <div className="prob-section">
                <div className="prob-bar-track">
                  <div
                    className="prob-away-fill"
                    style={{ width: `${awayProb * 100}%` }}
                  />
                  <div
                    className="prob-home-fill"
                    style={{ width: `${g.home_win_prob * 100}%` }}
                  />
                </div>
                <div className="prob-labels">
                  <span className="prob-label">
                    {g.away_team}{" "}
                    <span className="prob-pct">
                      {(awayProb * 100).toFixed(0)}%
                    </span>
                  </span>
                  <span className="prob-label">
                    <span className="prob-pct">
                      {(g.home_win_prob * 100).toFixed(0)}%
                    </span>{" "}
                    {g.home_team}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
