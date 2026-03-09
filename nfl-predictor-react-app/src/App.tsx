import { useEffect, useState } from "react";
import Header from "./components/Header";
import SelectMenu from "./components/SelectMenu";
import GameList from "./components/GameList";
import "bootstrap/dist/css/bootstrap.min.css";
import type { GamePrediction, PredictionsResponse } from "./types.tsx";

const API_BASE = "https://nfl-predictor-reactwebsite.onrender.com";

export default function App() {
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null);
  const [selectedSeason, setSelectedSeason] = useState<number>(2025);
  const [games, setGames] = useState<GamePrediction[]>([]);
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
      } catch (e) {
        setGames([]);
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

  return (
    <div>
      <Header />

      <SelectMenu
        value={selectedWeek}
        onChange={setSelectedWeek}
        season={selectedSeason}
        onSeasonChange={setSelectedSeason}
      />

      {error && <p className="text-danger">{error}</p>}

      {games.length > 0 && (
        <div className="sort-toggle-container">
          <button
            className={`sort-toggle-btn${sortByConf ? " active" : ""}`}
            onClick={() => setSortByConf((v) => !v)}
          >
            {sortByConf ? "Sorted by Confidence" : "Sort by Confidence"}
          </button>
        </div>
      )}

      <GameList games={displayGames} loading={loading} />
    </div>
  );
}
