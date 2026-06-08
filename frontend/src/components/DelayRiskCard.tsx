interface Props {
  predictedDelayMinutes: number | null
  observedDelayMinutes?: number
  inboundDelayMinutes?: number | null
  delayProbability?: number
}

function riskLabel(mins: number): { label: string; color: string } {
  if (mins <= 5)  return { label: 'Low',       color: '#22c55e' }
  if (mins <= 20) return { label: 'Moderate',  color: '#eab308' }
  if (mins <= 45) return { label: 'High',      color: '#f97316' }
  return             { label: 'Very High',  color: '#ef4444' }
}

export default function DelayRiskCard({
  predictedDelayMinutes,
  observedDelayMinutes,
  inboundDelayMinutes,
  delayProbability,
}: Props) {
  const mins = predictedDelayMinutes ?? 0
  const { label, color } = riskLabel(mins)
  const pct = Math.min((mins / 60) * 100, 100)

  return (
    <div className="card delay-card">
      <h3 className="card-title">Delay Prediction</h3>

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

      <div className="delay-breakdown">
        {observedDelayMinutes != null && observedDelayMinutes > 0 && (
          <span className="delay-detail">Observed: +{observedDelayMinutes} min</span>
        )}
        {inboundDelayMinutes != null && inboundDelayMinutes > 0 && (
          <span className="delay-detail">Inbound aircraft: +{inboundDelayMinutes} min</span>
        )}
        {delayProbability != null && (
          <span className="delay-detail">Delay probability: {delayProbability}%</span>
        )}
      </div>
    </div>
  )
}
