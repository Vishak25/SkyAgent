interface Props {
  log: string[]
}

// Prefix → icon mapping. Order matters: first match wins.
const ICONS: [string, string][] = [
  ['FlightMonitor', '✈'],
  ['Track',         '✈'],
  ['Weather',       '🌤'],
  ['DelayRisk',     '📊'],
  ['Rerouting',     '🔀'],
  ['Summary',       '🤖'],
]

function getIcon(entry: string): string {
  for (const [prefix, icon] of ICONS) {
    if (entry.startsWith(prefix)) return icon
  }
  return '•'
}

export default function AgentActivityFeed({ log }: Props) {
  if (!log || log.length === 0) return null

  return (
    <div className="card agent-feed">
      <h3 className="card-title">Agent Pipeline</h3>
      <ol className="agent-log">
        {log.map((entry, i) => (
          <li key={i} className="log-entry">
            <span className="log-icon">{getIcon(entry)}</span>
            <span className="log-text">{entry}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
