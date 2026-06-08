import { WeatherWidget } from '../types'

interface Props {
  label: string
  iata: string
  data: WeatherWidget | null
}

const CAT_COLOR: Record<string, string> = {
  VFR:  '#22c55e',
  MVFR: '#3b82f6',
  IFR:  '#f97316',
  LIFR: '#ef4444',
}

export default function WeatherCard({ label, iata, data }: Props) {
  if (!data) return (
    <div className="card weather-card">
      <h3 className="card-title">{label} <span className="iata-small">{iata}</span></h3>
      <p className="muted">Weather unavailable</p>
    </div>
  )

  const cat = data.flightCategory ?? 'VFR'
  const catColor = CAT_COLOR[cat] ?? '#94a3b8'

  return (
    <div className="card weather-card">
      <div className="weather-header">
        <h3 className="card-title">{label} <span className="iata-small">{iata}</span></h3>
        <span className="cat-badge" style={{ background: catColor + '22', color: catColor, border: `1px solid ${catColor}44` }}>
          {cat}
        </span>
      </div>

      <div className="weather-grid">
        <div className="wx-item">
          <span className="wx-label">Condition</span>
          <span className="wx-value">{data.condition}</span>
        </div>
        <div className="wx-item">
          <span className="wx-label">Visibility</span>
          <span className="wx-value">{data.visibility}</span>
        </div>
        <div className="wx-item">
          <span className="wx-label">Wind</span>
          <span className="wx-value">{data.windSpeed} kt</span>
        </div>
        <div className="wx-item">
          <span className="wx-label">Temp</span>
          <span className="wx-value">{data.temp}°F</span>
        </div>
        {data.precipLabel && data.precipLabel !== 'None' && (
          <div className="wx-item wx-item-full">
            <span className="wx-label">Precip</span>
            <span className="wx-value wx-warn">{data.precipLabel}</span>
          </div>
        )}
      </div>
    </div>
  )
}
