import { useEffect, useState } from "react";
import Header from "./components/Header";
import SelectMenu from "./components/SelectMenu";
import GameList from "./components/GameList";
import "./index.css";
import type { GamePrediction, PredictionsResponse } from "./types.tsx";

const API_BASE = "http://3.12.104.234";

export default function App() {
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null);
  const [selectedSeason, setSelectedSeason] = useState<number>(2026);
  const [games, setGames] = useState<GamePrediction[]>([]);
  const [roundName, setRoundName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [sortByConf, setSortByConf] = useState(false);

  useEffect(() => {
    if (selectedWeek === null) return;

    const run = async () => {
      setLoading(true);
      setError("");

      try {
        const res = await fetch(
          `${API_BASE}/predictions?week=${selectedWeek}&season=${selectedSeason}`
        );

        if (!res.ok) {
          const errJson = (await res.json().catch(() => null)) as {
            detail?: string;
          } | null;
          throw new Error(errJson?.detail ?? `Request failed (${res.status})`);
        }

        const data = (await res.json()) as PredictionsResponse;
        setGames(data.predictions);
        setRoundName(data.round_name ?? null);
      } catch (e) {
        setGames([]);
        setRoundName(null);
        setError(
          e instanceof Error ? e.message : "Failed to load predictions."
        );
      } finally {
        setLoading(false);
      }
    };

    run();
  }, [selectedWeek, selectedSeason]);

  const displayGames: GamePrediction[] = sortByConf
    ? [...games].sort((a, b) => b.confidence - a.confidence)
    : games;

  const isPlayoff = selectedWeek !== null && selectedWeek >= 19;

  return (
    <div className="app-wrapper">
      <Header />

      <SelectMenu
        value={selectedWeek}
        onChange={setSelectedWeek}
        season={selectedSeason}
        onSeasonChange={setSelectedSeason}
      />

      {error && <div className="error-banner">{error}</div>}

      {games.length > 0 && (
        <div className="results-header">
          {isPlayoff && roundName ? (
            <div className="playoff-round-label">
              <span className="playoff-badge">PLAYOFFS</span>
              <span className="playoff-round-name">{roundName}</span>
            </div>
          ) : (
            <div className="results-week-label">
              Week {selectedWeek} · {selectedSeason} Season
            </div>
          )}

          <button
            className={`sort-btn${sortByConf ? " active" : ""}`}
            onClick={() => setSortByConf((v) => !v)}
          >
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
              <path d="M1 2h9M3 5.5h5M5 9h1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            By Confidence
          </button>
        </div>
      )}

      <GameList games={displayGames} loading={loading} />
    </div>
  );
}
