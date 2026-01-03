export type GamePrediction = {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  home_win_prob: number; // 0..1
  confidence: number; // 0..1
  game_date: string | null;
};

export type PredictionsResponse = {
  season: number;
  week: number;
  num_games: number;
  predictions: GamePrediction[];
};