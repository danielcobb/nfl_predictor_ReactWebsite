import { useEffect, useState } from "react";
import Header from "./components/Header";
import SelectMenu from "./components/SelectMenu";
import GameList from "./components/GameList";
import "bootstrap/dist/css/bootstrap.min.css";
import type { GamePrediction, PredictionsResponse } from "./types.tsx";

const API_BASE = "https://nfl-predictor-reactwebsite.onrender.com";

export default function App() {
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null);
  const [games, setGames] = useState<GamePrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (selectedWeek === null) return;

    const run = async () => {
      setLoading(true);
      setError("");

      try {
        const res = await fetch(
          `${API_BASE}/predictions?week=${selectedWeek}&season=2025`
        );

        if (!res.ok) {
          // FastAPI usually returns { detail: "..." }
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
  }, [selectedWeek]);

  return (
    <div>
      <Header />

      <SelectMenu value={selectedWeek} onChange={setSelectedWeek} />

      {loading && <p className="text-light">Running predictions…</p>}
      {error && <p className="text-danger">{error}</p>}

      <GameList games={games} />
    </div>
  );
}
