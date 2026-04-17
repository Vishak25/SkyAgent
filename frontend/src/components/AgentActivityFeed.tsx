interface Props {
  log: string[]
}

const ICONS: Record<string, string> = {
  FlightMonitor: '✈',
  Weather:       '🌤',
  DelayRisk:     '📊',
  Rerouting:     '🔀',
  Summary:       '🤖',
}

function getIcon(entry: string): string {
  for (const [key, icon] of Object.entries(ICONS)) {
    if (entry.startsWith(key)) return icon
  }
  return '•'
}

export default function AgentActivityFeed({ log }: Props) {
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
