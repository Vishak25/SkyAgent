import { WeatherData } from '../types'

interface Props {
  label: string
  iata: string
  data: WeatherData | null
}

const CAT_COLOR: Record<string, string> = {
  VFR: '#22c55e',
  MVFR: '#3b82f6',
  IFR: '#f97316',
  LIFR: '#ef4444',
}

export default function WeatherCard({ label, iata, data }: Props) {
  if (!data) return (
    <div className="card weather-card">
      <h3 className="card-title">{label} <span className="iata-small">{iata}</span></h3>
      <p className="muted">Weather unavailable</p>
    </div>
  )

  const catColor = CAT_COLOR[data.flight_category] ?? '#94a3b8'

  return (
    <div className="card weather-card">
      <div className="weather-header">
        <h3 className="card-title">{label} <span className="iata-small">{iata}</span></h3>
        <span className="cat-badge" style={{ background: catColor + '22', color: catColor, border: `1px solid ${catColor}44` }}>
          {data.flight_category}
        </span>
      </div>

      <div className="weather-grid">
        <div className="wx-item">
          <span className="wx-label">Visibility</span>
          <span className="wx-value">{data.visibility_miles} sm</span>
        </div>
        <div className="wx-item">
          <span className="wx-label">Wind</span>
          <span className="wx-value">{data.wind_speed_kts} kt</span>
        </div>
        <div className="wx-item">
          <span className="wx-label">Ceiling</span>
          <span className="wx-value">
            {data.ceiling_ft >= 9999 ? 'CLR' : `${(data.ceiling_ft / 100).toFixed(0)}00 ft`}
          </span>
        </div>
        <div className="wx-item">
          <span className="wx-label">Temp</span>
          <span className="wx-value">{data.temp_c}°C / {data.temp_f}°F</span>
        </div>
        {data.precip_label && data.precip_label !== 'None' && (
          <div className="wx-item wx-item-full">
            <span className="wx-label">Precip</span>
            <span className="wx-value wx-warn">{data.precip_label}</span>
          </div>
        )}
      </div>
    </div>
  )
}
