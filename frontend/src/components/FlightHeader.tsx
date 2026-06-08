import { FlightAnalysis } from '../types'

interface Props {
  result: FlightAnalysis
}

function statusColor(s: string): string {
  const l = s.toLowerCase()
  if (l.includes('on time') || l.includes('early') || l.includes('arrived')) return '#22c55e'
  if (l.includes('delay') || l.includes('late')) return '#f97316'
  if (l.includes('cancel')) return '#ef4444'
  if (l.includes('en route')) return '#3b82f6'
  return '#94a3b8'
}

export default function FlightHeader({ result }: Props) {
  const {
    flightNumber, airline, status,
    origin, destination,
    scheduledDep, actualDep, depTimeKind,
    scheduledArr, actualArr,
    gateOrigin, gateDest, terminalOrigin, terminalDest,
    observedDelayMinutes, inboundDelayMinutes,
    incomingAircraftStatus,
  } = result

  const depDisplay = actualDep !== 'TBD' ? actualDep : scheduledDep
  const arrDisplay = actualArr !== 'TBD' ? actualArr : scheduledArr
  const isEstimated = depTimeKind === 'estimated'

  return (
    <div className="card flight-header">
      <div className="header-top">
        <div className="flight-id">
          <span className="flight-num">{flightNumber}</span>
          {airline && airline !== 'Unknown Airline' && (
            <span className="operator">{airline}</span>
          )}
          <span className="aircraft-status">{incomingAircraftStatus}</span>
        </div>
        <span className="flight-status-badge" style={{ color: statusColor(status) }}>
          {status}
        </span>
      </div>

      <div className="route-row">
        <div className="airport-block">
          <span className="iata">{origin}</span>
          <span className="time">
            {depDisplay}
            {isEstimated && <span className="est-label"> est.</span>}
          </span>
          {gateOrigin !== '-' && (
            <span className="detail">
              {terminalOrigin !== '-' ? `T${terminalOrigin} · ` : ''}Gate {gateOrigin}
            </span>
          )}
        </div>

        <div className="route-arrow">
          <div className="arrow-line"><span>→</span></div>
        </div>

        <div className="airport-block right">
          <span className="iata">{destination}</span>
          <span className="time">{arrDisplay}</span>
          {gateDest !== '-' && (
            <span className="detail">
              {terminalDest !== '-' ? `T${terminalDest} · ` : ''}Gate {gateDest}
            </span>
          )}
        </div>
      </div>

      <div className="delay-pills">
        {observedDelayMinutes > 0 && (
          <span className="pill pill-warn">Dep +{observedDelayMinutes} min</span>
        )}
        {observedDelayMinutes === 0 && status.toLowerCase().includes('on time') && (
          <span className="pill pill-ok">On Time</span>
        )}
        {inboundDelayMinutes != null && inboundDelayMinutes > 0 && (
          <span className="pill pill-warn">Inbound +{inboundDelayMinutes} min</span>
        )}
      </div>
    </div>
  )
}
