import { useState } from 'react'
import { AnalysisResult } from './types'
import FlightSearch from './components/FlightSearch'
import FlightHeader from './components/FlightHeader'
import WeatherCard from './components/WeatherCard'
import DelayRiskCard from './components/DelayRiskCard'
import AgentActivityFeed from './components/AgentActivityFeed'
import LLMSummary from './components/LLMSummary'
import RouteComparison from './components/RouteComparison'

export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async (flightNumber: string) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`/api/analyze/${flightNumber}`)
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      const data: AnalysisResult = await res.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const status = result?.flight_status?.status

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
              {['Fetching flight status...', 'Fetching weather...', 'Running ST-GNN...', 'Generating assessment...'].map((s, i) => (
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

        {result && status && (
          <div className="results">
            <FlightHeader flightNumber={result.flight_number} status={status} />

            <div className="grid-2">
              <WeatherCard label="Origin" iata={result.origin} data={result.weather_origin} />
              <WeatherCard label="Destination" iata={result.destination} data={result.weather_destination} />
            </div>

            <div className="grid-2">
              <DelayRiskCard predictedDelay={result.predicted_delay} />
              <AgentActivityFeed log={result.agent_log} />
            </div>

            <LLMSummary summary={result.llm_summary} />

            {result.alternative_routes && result.alternative_routes.length > 0 && (
              <RouteComparison routes={result.alternative_routes} />
            )}
          </div>
        )}
      </main>
    </div>
  )
}
