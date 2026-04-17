export interface WeatherData {
  visibility_miles: number
  wind_speed_kts: number
  ceiling_ft: number
  flight_category: 'VFR' | 'MVFR' | 'IFR' | 'LIFR'
  temp_c: number
  temp_f: number
  precip_severity: number
  precip_label: string
  conditions_raw: string[]
}

export interface AirportRef {
  code: string
  code_iata: string
  code_icao: string
  name: string
  city: string
  timezone: string
}

export interface FlightStatus {
  ident: string
  ident_iata: string
  status: string
  aircraft_type: string
  operator: string
  origin: AirportRef
  destination: AirportRef
  departure_delay: number
  arrival_delay: number
  scheduled_out: string
  estimated_out: string
  actual_out: string | null
  scheduled_in: string
  estimated_in: string
  actual_in: string | null
  progress_percent: number
  terminal_origin: string | null
  terminal_destination: string | null
  gate_origin: string | null
  gate_destination: string | null
  route_distance: number
  filed_airspeed: number
}

export interface Itinerary {
  legs: { origin: string; destination: string; predicted_delay_minutes: number }[]
  total_delay_minutes: number
}

export interface AnalysisResult {
  flight_number: string
  origin: string
  destination: string
  flight_status: { status: FlightStatus; position: Record<string, unknown> }
  weather_origin: WeatherData | null
  weather_destination: WeatherData | null
  predicted_delay: number | null
  risk_score: number | null
  alternative_routes: Itinerary[] | null
  llm_summary: string | null
  agent_log: string[]
}
