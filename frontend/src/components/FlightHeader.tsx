import { FlightStatus } from '../types'

interface Props {
  flightNumber: string
  status: FlightStatus
}

function fmtTime(iso: string | null, tz: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString('en-US', {
      hour: 'numeric', minute: '2-digit', hour12: true, timeZone: tz,
    })
  } catch {
    return iso.slice(11, 16)
  }
}

function statusColor(s: string): string {
  const l = s.toLowerCase()
  if (l.includes('on time') || l === 'landed') return '#22c55e'
  if (l.includes('delay') || l.includes('late')) return '#f97316'
  if (l.includes('cancel')) return '#ef4444'
  return '#94a3b8'
}

export default function FlightHeader({ flightNumber, status }: Props) {
  // AeroAPI returns delays in seconds — convert to minutes
  const depDelay = Math.round((status.departure_delay ?? 0) / 60)
  const arrDelay = Math.round((status.arrival_delay ?? 0) / 60)

  return (
    <div className="card flight-header">
      <div className="header-top">
        <div className="flight-id">
          <span className="flight-num">{flightNumber}</span>
          <span className="operator">{status.operator}</span>
          <span className="aircraft">{status.aircraft_type}</span>
        </div>
        <span className="flight-status-badge" style={{ color: statusColor(status.status) }}>
          {status.status}
        </span>
      </div>

      <div className="route-row">
        <div className="airport-block">
          <span className="iata">{status.origin?.code_iata ?? '—'}</span>
          <span className="city">{status.origin?.city ?? '—'}</span>
          <span className="time">{fmtTime(status.estimated_out ?? status.scheduled_out, status.origin?.timezone)}</span>
          {status.terminal_origin && <span className="detail">T{status.terminal_origin}</span>}
        </div>

        <div className="route-arrow">
          <span className="distance">{status.route_distance ? `${status.route_distance} nm` : ''}</span>
          <div className="arrow-line"><span>→</span></div>
        </div>

        <div className="airport-block right">
          <span className="iata">{status.destination?.code_iata ?? '—'}</span>
          <span className="city">{status.destination?.city ?? '—'}</span>
          <span className="time">{fmtTime(status.estimated_in ?? status.scheduled_in, status.destination?.timezone)}</span>
          {status.gate_destination && <span className="detail">Gate {status.gate_destination}</span>}
        </div>
      </div>

      {(depDelay !== 0 || arrDelay !== 0) && (
        <div className="delay-pills">
          {depDelay !== 0 && (
            <span className={`pill ${depDelay > 0 ? 'pill-warn' : 'pill-ok'}`}>
              Dep {depDelay > 0 ? `+${depDelay}` : depDelay} min
            </span>
          )}
          {arrDelay !== 0 && (
            <span className={`pill ${arrDelay > 0 ? 'pill-warn' : 'pill-ok'}`}>
              Arr {arrDelay > 0 ? `+${arrDelay}` : arrDelay} min
            </span>
          )}
        </div>
      )}
    </div>
  )
}
