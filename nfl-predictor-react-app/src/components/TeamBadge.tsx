import { TEAM_LOGOS } from "../teamLogos";

type Props = {
  abbr: string;
};

export default function TeamBadge({ abbr }: Props) {
  const src = TEAM_LOGOS[abbr];

  return (
    <span className="team-badge">
      {src && <img className="team-logo" src={src} alt={`${abbr} logo`} />}
      <span className="team-abbr">{abbr}</span>
    </span>
  );
}
