import { Itinerary } from '../types'

interface Props {
  routes: Itinerary[]
}

export default function RouteComparison({ routes }: Props) {
  if (!routes || routes.length === 0) return null

  return (
    <div className="card route-comparison">
      <h3 className="card-title">Alternative Routes</h3>
      <div className="routes-list">
        {routes.slice(0, 5).map((route, i) => {
          const legs = route.legs ?? []
          const total = route.total_delay_minutes ?? legs.reduce((s, l) => s + (l.predicted_delay_minutes ?? 0), 0)
          return (
            <div key={i} className={`route-row ${i === 0 ? 'route-best' : ''}`}>
              {i === 0 && <span className="best-badge">Best</span>}
              <span className="route-path">
                {legs.map((l, j) => (
                  <span key={j}>
                    {j > 0 && <span className="route-sep">→</span>}
                    <span className="route-airport">{l.destination}</span>
                  </span>
                ))}
              </span>
              <span className="route-delay">{total} min</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
