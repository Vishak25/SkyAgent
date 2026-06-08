// Canonical camelCase response types — matches the unified backend /analyze and /predict shape.

export interface WeatherWidget {
  flightCategory: 'VFR' | 'MVFR' | 'IFR' | 'LIFR'
  temp: number              // °F
  condition: string         // e.g. "Clear", "Thunderstorm"
  windSpeed: number         // knots
  visibility: string        // pre-formatted, e.g. "10 mi"
  precipSeverity: number    // 0–100
  precipLabel: string       // e.g. "None", "Rain", "Heavy Snow"
}

export interface LivePosition {
  lat: number
  lon: number
  heading: number | null
  altitude: number | null
}

export interface ItineraryLeg {
  origin: string
  destination: string
  scheduledDep: string
  scheduledArr: string
  flightNumber: string
}

export interface Itinerary {
  type: string              // "direct" | "connection"
  flightNumber: string
  airline: string
  legs: ItineraryLeg[]
  predictedDelayMinutes: number
  delayRisk: string         // "Low" | "Moderate" | "High" | "Very High"
  propagationRisk: number   // 0–100
  precipSeverity: number    // 0–100
  stops: number
  rank?: number
  recommended?: boolean
  connectionHub?: string
  connectionHubName?: string
}

export interface FlightAnalysis {
  // Identity
  flightNumber: string
  origin: string
  destination: string
  originIcao?: string
  destinationIcao?: string
  status: string            // "On Time" | "Delayed" | "Cancelled" | "En Route" | ...
  airline: string

  // Timing (pre-formatted local-time strings, e.g. "2:45 PM EST")
  scheduledDep: string
  actualDep: string
  depTimeKind: string       // "actual" | "estimated" | "unknown"
  predictedTakeoff: string
  scheduledArr: string
  actualArr: string
  arrTimeKind: string       // "actual" | "estimated" | "unknown"

  // Gate / terminal
  terminalOrigin: string
  gateOrigin: string
  terminalDest: string
  gateDest: string
  baggageClaim: string

  // Delay
  predictedDelayMinutes: number
  observedDelayMinutes: number
  modelPredictedDelay: number
  inboundDelayMinutes?: number | null   // propagated from previous leg
  inboundFlightId?: string | null
  delayProbability: number              // 5–95 %

  // Risk metrics (0–100)
  networkCongestion: number
  propagationRisk: number
  precipSeverity: number
  incomingAircraftStatus: string

  // Weather (null when unavailable)
  weatherOrigin: WeatherWidget | null
  weatherDest: WeatherWidget | null

  // Live position (null when on ground / unavailable)
  livePosition: LivePosition | null

  // Pipeline-only extras — present on /analyze, absent on /predict
  llmSummary?: string | null
  alternativeRoutes?: Itinerary[] | null
  agentLog?: string[]

  note?: string
}
