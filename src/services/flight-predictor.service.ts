
import { Injectable } from '@angular/core';

// --- Track Mode types ---
export interface FlightScenario {
  flightNumber: string;
  origin: string;
  destination: string;

  scheduledDep: string;
  actualDep: string;
  depTimeKind?: 'actual' | 'estimated' | 'unknown';
  predictedTakeoff: string;
  scheduledArr: string;
  actualArr: string;
  arrTimeKind?: 'actual' | 'estimated' | 'unknown';

  terminalOrigin: string;
  gateOrigin: string;
  terminalDest: string;
  gateDest: string;
  baggageClaim: string;

  airline: string;
  status: string;  // 'On Time' | 'Delayed' | 'Severely Delayed' | 'Delayed - En Route' | 'Severely Delayed - En Route' | 'Slight Delay' | 'Cancelled' | 'Early'
  delayProbability: number;
  predictedDelayMinutes: number;
  observedDelayMinutes?: number;
  modelPredictedDelay?: number;
  weatherOrigin: WeatherInfo;
  weatherDest: WeatherInfo;
  networkCongestion: number;
  incomingAircraftStatus: 'Landed' | 'In Air' | 'At Gate' | 'Unknown' | 'Delayed at Previous Leg';
  propagationRisk: number;
  precipSeverity?: number;
  isSimulation: boolean;
  graphData?: any;
  sources?: { title: string; uri: string }[];
}

export interface WeatherInfo {
  temp: number;
  condition: string;
  windSpeed: number;
  visibility: string;
  precipSeverity?: number;
  precipLabel?: string;
}

// --- Suggest Mode types ---
export interface ItineraryLeg {
  origin: string;
  destination: string;
  scheduledDep: string;
  scheduledArr: string;
  flightNumber: string;
}

export interface ItineraryOption {
  type: 'direct' | 'connection';
  flightNumber: string;
  airline: string;
  legs: ItineraryLeg[];
  connectionHub?: string;
  connectionHubName?: string;
  predictedDelayMinutes: number;
  delayRisk: 'Low' | 'Moderate' | 'High' | 'Very High';
  propagationRisk: number;
  precipSeverity: number;
  stops: number;
  rank: number;
  recommended: boolean;
}

export interface ForecastInfo {
  visibility: string;
  windSpeed: number;
  flightCategory: string;
  precipSeverity: number;
  precipLabel: string;
}

export interface RouteSuggestion {
  origin: string;
  originName: string;
  destination: string;
  destinationName: string;
  date: string;
  itineraryCount: number;
  itineraries: ItineraryOption[];
  weatherOrigin: WeatherInfo;
  weatherDest: WeatherInfo;
  forecastOrigin: ForecastInfo;
  forecastDest: ForecastInfo;
  sources?: { title: string; uri: string }[];
}

@Injectable({
  providedIn: 'root'
})
export class FlightPredictorService {

  constructor() { }

  // --- Mode 1: Track a booked flight ---

  async getFlightStatus(flightNumber: string): Promise<FlightScenario> {
    try {
      const ident = flightNumber.trim().toUpperCase();
      const response = await fetch(`http://localhost:8000/predict/${encodeURIComponent(ident)}`);

      let data: any = null;
      try { data = await response.json(); } catch { data = null; }

      if (!response.ok) {
        const msg = data?.detail || data?.error || `Backend API Error (${response.status})`;
        throw new Error(msg);
      }
      if (data?.error) {
        throw new Error(data.detail || data.error || 'Flight not found in live data.');
      }

      return { ...data, isSimulation: false, sources: data.sources || [] };
    } catch (error) {
      console.warn('Real-time fetch failed', error);
      throw error;
    }
  }

  // --- Mode 2: Suggest routes (pre-departure) ---

  async suggestRoutes(origin: string, destination: string, date?: string): Promise<RouteSuggestion> {
    try {
      const o = origin.trim().toUpperCase();
      const d = destination.trim().toUpperCase();
      let url = `http://localhost:8000/suggest?origin=${encodeURIComponent(o)}&destination=${encodeURIComponent(d)}`;
      if (date) url += `&date=${encodeURIComponent(date)}`;

      const response = await fetch(url);
      let data: any = null;
      try { data = await response.json(); } catch { data = null; }

      if (!response.ok) {
        const msg = data?.detail || data?.error || `Backend API Error (${response.status})`;
        throw new Error(msg);
      }
      if (data?.error) {
        throw new Error(data.detail || data.error || 'Route lookup failed.');
      }

      return data as RouteSuggestion;
    } catch (error) {
      console.warn('Route suggestion failed', error);
      throw error;
    }
  }

  // --- AI Insight (rule-based, works for both modes) ---

  async analyzeScenario(scenario: FlightScenario): Promise<string> {
    const parts: string[] = [];
    const observed = scenario.observedDelayMinutes ?? 0;
    const modelDelay = scenario.modelPredictedDelay ?? 0;
    const totalDelay = scenario.predictedDelayMinutes;

    const statusLower = (scenario.status || '').toLowerCase();

    if (statusLower.includes('cancelled')) {
      parts.push(`This flight has been cancelled.`);
    } else if (statusLower.includes('arrived')) {
      // Flight has landed
      if (observed > 30) {
        parts.push(`Flight ${scenario.flightNumber} has landed, arriving ${observed} min behind schedule.`);
      } else if (observed > 0) {
        parts.push(`Flight ${scenario.flightNumber} has landed with a minor ${observed} min delay.`);
      } else if (statusLower.includes('early')) {
        parts.push(`Flight ${scenario.flightNumber} has landed ahead of schedule!`);
      } else {
        parts.push(`Flight ${scenario.flightNumber} has arrived on time.`);
      }
    } else if (statusLower.includes('severely')) {
      if (observed > 0) {
        parts.push(`This flight is running ${observed} min behind schedule.`);
        if (modelDelay > 5) {
          parts.push(`Weather conditions add an estimated ${modelDelay} min on top.`);
        }
      } else {
        parts.push(`SkyCast predicts a significant ${totalDelay} min delay.`);
      }
    } else if (statusLower.includes('delayed') || statusLower.includes('slight')) {
      if (observed > 0) {
        parts.push(`Flight ${scenario.flightNumber} is ${observed} min behind schedule.`);
      } else {
        parts.push(`SkyCast predicts a ${totalDelay} min delay.`);
      }
    } else if (totalDelay > 15 && statusLower.includes('on time')) {
      parts.push(`Heads up: While the airport says "On Time", our AI detects a potential ${totalDelay} min delay forming.`);
    } else {
      parts.push(`Good news! Flight ${scenario.flightNumber} looks set to depart On Time.`);
    }

    // Winter ops / precipitation
    const precipSev = scenario.precipSeverity ?? scenario.weatherOrigin?.precipSeverity ?? 0;
    if (precipSev > 60) {
      const label = scenario.weatherOrigin?.precipLabel || 'winter weather';
      parts.push(`Winter ops alert: ${label} at ${scenario.origin} may cause de-icing delays and possible runway cleaning.`);
    } else if (precipSev > 30) {
      parts.push(`Moderate precipitation at ${scenario.origin} could add minor delays.`);
    }

    if (scenario.incomingAircraftStatus === 'Delayed at Previous Leg') {
      parts.push("Your plane is arriving late from its previous leg, which will likely delay your boarding.");
    } else if (scenario.incomingAircraftStatus === 'Landed') {
      parts.push(`Your plane has landed at the destination.`);
    } else if (scenario.incomingAircraftStatus === 'In Air') {
      parts.push(`Your plane is currently in the air and on track.`);
    }

    if (scenario.networkCongestion > 70) {
      parts.push(`High operational congestion at ${scenario.origin} (${scenario.networkCongestion}%) might slow down pushback from the gate.`);
    }

    if (scenario.propagationRisk > 60) {
      parts.push(`Bad weather at flight corridors or destination is increasing the risk of ground stops.`);
    }

    if (parts.length === 1) {
      parts.push("Current operational conditions at the airport are smooth.");
    }

    return parts.join(' ');
  }

  analyzeRoutes(suggestion: RouteSuggestion): string {
    const parts: string[] = [];
    const best = suggestion.itineraries[0];

    if (!best) return "No itineraries found for this route.";

    if (best.delayRisk === 'Low') {
      parts.push(`Great news! The best option (${best.type === 'direct' ? 'direct' : 'via ' + best.connectionHub}) has low delay risk.`);
    } else if (best.delayRisk === 'Moderate') {
      parts.push(`The recommended route has moderate delay risk (${best.predictedDelayMinutes} min predicted).`);
    } else {
      parts.push(`All routes show elevated delay risk. The best option predicts ~${best.predictedDelayMinutes} min delay.`);
    }

    // Winter ops warning
    const maxPrecip = Math.max(...suggestion.itineraries.map(it => it.precipSeverity));
    if (maxPrecip > 50) {
      parts.push(`Winter weather (snow/ice) is affecting some routes. Check precipitation indicators on each option.`);
    }

    const directCount = suggestion.itineraries.filter(it => it.type === 'direct').length;
    const connCount = suggestion.itineraries.filter(it => it.type === 'connection').length;
    parts.push(`Showing ${directCount} direct and ${connCount} connection options.`);

    if (best.type === 'connection' && best.connectionHub) {
      const directOption = suggestion.itineraries.find(it => it.type === 'direct');
      if (directOption && directOption.predictedDelayMinutes > best.predictedDelayMinutes + 10) {
        parts.push(`Connecting via ${best.connectionHub} could save ~${directOption.predictedDelayMinutes - best.predictedDelayMinutes} min vs direct.`);
      }
    }

    return parts.join(' ');
  }

  generateDummyScenario(flightNumber: string): FlightScenario {
    const isBadDay = Math.random() > 0.4;
    const delayProb = isBadDay ? Math.floor(Math.random() * 40) + 60 : Math.floor(Math.random() * 20);
    const predictedDelay = isBadDay ? Math.floor(Math.random() * 120) + 15 : 0;
    return {
      flightNumber: flightNumber.toUpperCase(),
      origin: 'ORD', destination: 'IAD',
      scheduledDep: '14:30', actualDep: '14:35', predictedTakeoff: '14:45',
      scheduledArr: '17:30', actualArr: '17:42',
      terminalOrigin: '1', gateOrigin: 'B12', terminalDest: 'Main', gateDest: 'D4', baggageClaim: '3',
      airline: 'Simulated Air',
      status: delayProb > 80 ? 'Cancelled' : (delayProb > 50 ? 'Delayed' : 'On Time'),
      delayProbability: delayProb, predictedDelayMinutes: predictedDelay,
      weatherOrigin: { temp: 45, condition: isBadDay ? 'Snow' : 'Clear Sky', windSpeed: isBadDay ? 25 : 8, visibility: isBadDay ? '0.5 mi' : '10 mi', precipSeverity: isBadDay ? 65 : 0, precipLabel: isBadDay ? 'SN' : 'None' },
      weatherDest: { temp: 52, condition: 'Overcast', windSpeed: 12, visibility: '8 mi' },
      networkCongestion: isBadDay ? 85 : 40,
      incomingAircraftStatus: isBadDay ? 'Delayed at Previous Leg' : 'In Air',
      propagationRisk: isBadDay ? 92 : 15,
      precipSeverity: isBadDay ? 65 : 0,
      isSimulation: true
    };
  }
}
