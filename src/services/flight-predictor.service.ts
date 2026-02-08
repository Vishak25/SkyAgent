
import { Injectable } from '@angular/core';
export interface FlightScenario {
  flightNumber: string;
  origin: string;
  destination: string;

  // Times
  scheduledDep: string;
  actualDep: string;
  predictedTakeoff: string;
  scheduledArr: string;
  actualArr: string;

  // Gate / Terminal
  terminalOrigin: string;
  gateOrigin: string;
  terminalDest: string;
  gateDest: string;
  baggageClaim: string;

  airline: string;
  status: 'On Time' | 'Delayed' | 'Cancelled' | 'Early';
  delayProbability: number; // 0-100
  predictedDelayMinutes: number;
  weatherOrigin: {
    temp: number;
    condition: string;
    windSpeed: number;
    visibility: string;
  };
  weatherDest: {
    temp: number;
    condition: string;
    windSpeed: number;
    visibility: string;
  };
  networkCongestion: number; // 0-100
  incomingAircraftStatus: 'Landed' | 'In Air' | 'Delayed at Previous Leg';
  propagationRisk: number; // 0-100 (Risk of knock-on effect)
  isSimulation: boolean;
  sources?: { title: string; uri: string }[];
}

@Injectable({
  providedIn: 'root'
})
export class FlightPredictorService {

  constructor() { }


  async getFlightStatus(flightNumber: string): Promise<FlightScenario> {
    try {
      const ident = flightNumber.trim().toUpperCase();
      const response = await fetch(`http://localhost:8000/predict/${encodeURIComponent(ident)}`);

      // Always try to parse JSON so we can surface useful backend errors (even on non-2xx)
      let data: any = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok) {
        const msg =
          data?.detail ||
          data?.error ||
          data?.message ||
          (data?.detail?.detail ?? data?.detail?.error) ||
          `Backend API Error (${response.status})`;
        throw new Error(msg);
      }

      if (data?.error) {
        throw new Error(data.detail || data.error || 'Flight not found in live data.');
      }

      return {
        ...data,
        isSimulation: false,
        // Ensure sources array exists even if backend doesn't send it
        sources: data.sources || []
      };

    } catch (error) {
      console.warn('Real-time fetch failed', error);
      // Propagate the specific error instead of falling back to simulation silently
      throw error;
    }
  }

  // Fallback for demo purposes or error states
  generateDummyScenario(flightNumber: string): FlightScenario {
    const isBadDay = Math.random() > 0.4;
    const delayProb = isBadDay ? Math.floor(Math.random() * 40) + 60 : Math.floor(Math.random() * 20);
    const predictedDelay = isBadDay ? Math.floor(Math.random() * 120) + 15 : 0;

    return {
      flightNumber: flightNumber.toUpperCase(),
      origin: 'ORD',
      destination: 'IAD',
      scheduledDep: '14:30',
      actualDep: '14:35',
      predictedTakeoff: '14:45',
      scheduledArr: '17:30',
      actualArr: '17:42',
      terminalOrigin: '1',
      gateOrigin: 'B12',
      terminalDest: 'Main',
      gateDest: 'D4',
      baggageClaim: '3',
      airline: 'Simulated Air',
      status: delayProb > 80 ? 'Cancelled' : (delayProb > 50 ? 'Delayed' : 'On Time'),
      delayProbability: delayProb,
      predictedDelayMinutes: predictedDelay,
      weatherOrigin: {
        temp: 45,
        condition: isBadDay ? 'Thunderstorms' : 'Clear Sky',
        windSpeed: isBadDay ? 25 : 8,
        visibility: isBadDay ? '0.5 mi' : '10 mi'
      },
      weatherDest: {
        temp: 52,
        condition: 'Overcast',
        windSpeed: 12,
        visibility: '8 mi'
      },
      networkCongestion: isBadDay ? 85 : 40,
      incomingAircraftStatus: isBadDay ? 'Delayed at Previous Leg' : 'In Air',
      propagationRisk: isBadDay ? 92 : 15,
      isSimulation: true
    };
  }

  async analyzeScenario(scenario: FlightScenario): Promise<string> {
    console.log('Analyzing Scenario:', scenario);

    // Rule-based Insight Generation (Passenger-Centric)
    const parts: string[] = [];

    // 1. Core Prediction vs Official Status
    if (scenario.status === 'Delayed') {
      parts.push(`⚠️ SkyCast predicts a ${scenario.predictedDelayMinutes} min delay.`);
    } else if (scenario.predictedDelayMinutes > 15 && scenario.status === 'On Time') {
      parts.push(`⚠️ **Heads up:** While the airport says "On Time", our AI detects a potential ${scenario.predictedDelayMinutes} min delay forming.`);
    } else {
      parts.push(`✅ Good news! Flight ${scenario.flightNumber} looks set to depart On Time.`);
    }

    // 2. Incoming Aircraft (The "Where is my plane?" factor)
    if (scenario.incomingAircraftStatus === 'Delayed at Previous Leg') {
      parts.push("Your plane is arriving late from its previous leg, which will likely delay your boarding.");
    } else if (scenario.incomingAircraftStatus === 'In Air') {
      const timeToGate = 25; // heuristic
      parts.push(`Your plane is currently in the air and on track.`);
    }

    // 3. Network Congestion (The invisible factor)
    if (scenario.networkCongestion > 70) {
      parts.push(`Be aware: High operational congestion at ${scenario.origin} (${scenario.networkCongestion}%) might slow down pushback from the gate.`);
    }

    // 4. Weather (Destination constraints)
    if (scenario.propagationRisk > 60) {
      parts.push(`Bad weather at flight corridors or destination is increasing the risk of ground stops.`);
    }

    if (parts.length === 1) {
      parts.push("Current operational conditions at the airport are smooth.");
    }

    return parts.join(' ');
  }
}
