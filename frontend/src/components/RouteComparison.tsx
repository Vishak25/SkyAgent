import { Itinerary } from '../types'

interface Props {
  routes: Itinerary[]
}

function riskColor(risk: string): string {
  switch (risk) {
    case 'Low':       return '#22c55e'
    case 'Moderate':  return '#eab308'
    case 'High':      return '#f97316'
    case 'Very High': return '#ef4444'
    default:          return '#94a3b8'
  }
}

export default function RouteComparison({ routes }: Props) {
  if (!routes || routes.length === 0) return null

  return (
    <div className="card route-comparison">
      <h3 className="card-title">Alternative Routes</h3>
      <div className="routes-list">
        {routes.slice(0, 5).map((route, i) => {
          const legs = route.legs ?? []
          // Build full path: origin → hub1 → ... → destination
          const airports = legs.length > 0
            ? [legs[0].origin, ...legs.map(l => l.destination)]
            : [route.flightNumber]

          const color = riskColor(route.delayRisk)

          return (
            <div key={i} className={`route-row ${route.recommended ? 'route-best' : ''}`}>
              {route.recommended && <span className="best-badge">Best</span>}

              <span className="route-path">
                {airports.map((ap, j) => (
                  <span key={j}>
                    {j > 0 && <span className="route-sep">→</span>}
                    <span className="route-airport">{ap}</span>
                  </span>
                ))}
              </span>

              <span className="route-meta">
                <span className="route-delay">{route.predictedDelayMinutes} min</span>
                <span
                  className="route-risk-badge"
                  style={{ color, background: color + '22', border: `1px solid ${color}44` }}
                >
                  {route.delayRisk}
                </span>
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
