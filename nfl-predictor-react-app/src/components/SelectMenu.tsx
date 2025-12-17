type SelectMenuProps = {
  value: number | null;
  onChange: (week: number | null) => void;
};

export default function SelectMenu({ value, onChange }: SelectMenuProps) {
  return (
    <div className="game-control-container">
      <select
        className="form-select game-select"
        value={value ?? ""}
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? null : Number(v));
        }}
      >
        <option value="" disabled>
          Select NFL Week
        </option>

        {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
          <option key={w} value={w}>
            Week {w}
          </option>
        ))}
      </select>
    </div>
  );
}
