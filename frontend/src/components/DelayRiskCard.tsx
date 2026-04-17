interface Props {
  predictedDelay: number | null
}

function riskLabel(mins: number): { label: string; color: string } {
  if (mins <= 5)  return { label: 'Low',       color: '#22c55e' }
  if (mins <= 20) return { label: 'Moderate',  color: '#eab308' }
  if (mins <= 45) return { label: 'High',      color: '#f97316' }
  return             { label: 'Very High',  color: '#ef4444' }
}

export default function DelayRiskCard({ predictedDelay }: Props) {
  const mins = predictedDelay ?? 0
  const { label, color } = riskLabel(mins)
  const pct = Math.min((mins / 60) * 100, 100)

  return (
    <div className="card delay-card">
      <h3 className="card-title">ST-GNN Delay Prediction</h3>

      <div className="delay-main">
        <span className="delay-mins" style={{ color }}>{mins}</span>
        <span className="delay-unit">min</span>
        <span className="delay-risk-label" style={{ background: color + '22', color, border: `1px solid ${color}44` }}>
          {label} Risk
        </span>
      </div>

      <div className="delay-bar-track">
        <div className="delay-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="delay-bar-labels">
        <span>0</span><span>15</span><span>30</span><span>45</span><span>60+</span>
      </div>

      <p className="delay-note muted">
        Predicted by Spatio-Temporal GNN trained on 30-day METAR observations across 20 hub airports.
      </p>
    </div>
  )
}
