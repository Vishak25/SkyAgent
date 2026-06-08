import { useState } from 'react'
import { FlightAnalysis } from './types'
import FlightSearch from './components/FlightSearch'
import FlightHeader from './components/FlightHeader'
import WeatherCard from './components/WeatherCard'
import DelayRiskCard from './components/DelayRiskCard'
import AgentActivityFeed from './components/AgentActivityFeed'
import LLMSummary from './components/LLMSummary'
import RouteComparison from './components/RouteComparison'

const FETCH_TIMEOUT_MS = 45_000

export default function App() {
  const [result, setResult] = useState<FlightAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async (flightNumber: string) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)

    try {
      const res = await fetch(`/api/analyze/${flightNumber}`, { signal: controller.signal })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      const data: FlightAnalysis = await res.json()
      setResult(data)
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        setError('Request timed out — the pipeline took longer than expected. Try again.')
      } else {
        setError(e instanceof Error ? e.message : 'Unknown error')
      }
    } finally {
      clearTimeout(timer)
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">✈</span>
          <span className="logo-text">SkyAgent</span>
        </div>
        <span className="logo-sub">Agentic Aviation Delay Propagation</span>
      </header>

      <main className="app-main">
        <FlightSearch onSearch={handleSearch} loading={loading} />

        {loading && (
          <div className="loading-state">
            <div className="loading-steps">
              {[
                'Fetching live flight status & schedule…',
                'Pulling METAR weather for origin & destination…',
                'Running ST-GNN delay propagation…',
                'Generating AI assessment…',
              ].map((s, i) => (
                <div key={i} className="loading-step">
                  <span className="step-dot" />
                  <span>{s}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="error-box">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {result && (
          <div className="results">
            <FlightHeader result={result} />

            <div className="grid-2">
              <WeatherCard label="Origin" iata={result.origin} data={result.weatherOrigin} />
              <WeatherCard label="Destination" iata={result.destination} data={result.weatherDest} />
            </div>

            <div className="grid-2">
              <DelayRiskCard
                predictedDelayMinutes={result.predictedDelayMinutes}
                observedDelayMinutes={result.observedDelayMinutes}
                inboundDelayMinutes={result.inboundDelayMinutes ?? null}
                delayProbability={result.delayProbability}
              />
              {result.agentLog && result.agentLog.length > 0 && (
                <AgentActivityFeed log={result.agentLog} />
              )}
            </div>

            {result.llmSummary && <LLMSummary summary={result.llmSummary} />}

            {result.alternativeRoutes && result.alternativeRoutes.length > 0 && (
              <RouteComparison routes={result.alternativeRoutes} />
            )}
          </div>
        )}
      </main>
    </div>
  )
}
