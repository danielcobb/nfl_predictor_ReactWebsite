const SEASONS = [2022, 2023, 2024, 2025];

type SelectMenuProps = {
  value: number | null;
  onChange: (week: number | null) => void;
  season: number;
  onSeasonChange: (season: number) => void;
};

export default function SelectMenu({
  value,
  onChange,
  season,
  onSeasonChange,
}: SelectMenuProps) {
  return (
    <div className="controls-panel">
      <div className="selectors-row">
        <div className="select-wrapper">
          <select
            className="game-select"
            value={season}
            onChange={(e) => onSeasonChange(Number(e.target.value))}
          >
            {SEASONS.map((y) => (
              <option key={y} value={y}>
                {y} Season
              </option>
            ))}
          </select>
        </div>

        <div className="select-wrapper">
          <select
            className="game-select"
            value={value ?? ""}
            onChange={(e) => {
              const v = e.target.value;
              onChange(v === "" ? null : Number(v));
            }}
          >
            <option value="" disabled>
              Select Week
            </option>
            {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
              <option key={w} value={w}>
                Week {w}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
