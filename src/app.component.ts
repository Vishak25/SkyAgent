
import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FlightPredictorService, FlightScenario } from './services/flight-predictor.service';
import { FlightLegCardComponent } from './components/flight-leg-card.component';
import { NetworkGraphComponent } from './components/network-graph.component';
import { StatCardComponent } from './components/stat-card.component';
import { WeatherWidgetComponent } from './components/weather-widget.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, NetworkGraphComponent, StatCardComponent, WeatherWidgetComponent, FlightLegCardComponent],
  templateUrl: './app.component.html'
})
export class AppComponent {
  private predictor = inject(FlightPredictorService);

  // State
  flightInput = signal<string>('');
  isLoading = signal<boolean>(false);
  flightData = signal<FlightScenario | null>(null);
  aiAnalysis = signal<string>('');
  errorMessage = signal<string>('');

  constructor() {
    this.trackFlight();
  }

  async trackFlight() {
    if (!this.flightInput()) return;

    this.isLoading.set(true);
    this.errorMessage.set('');
    this.aiAnalysis.set('');
    this.flightData.set(null);

    try {
      // Fetch real data
      const data = await this.predictor.getFlightStatus(this.flightInput());
      this.flightData.set(data);

      // Get AI explanation
      const explanation = await this.predictor.analyzeScenario(data);
      this.aiAnalysis.set(explanation);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Failed to fetch flight data.');
    } finally {
      this.isLoading.set(false);
    }
  }

  // Format Helpers
  get delayColor() {
    const delay = this.flightData()?.predictedDelayMinutes || 0;
    if (delay > 60) return 'text-rose-500';
    if (delay > 15) return 'text-amber-500';
    return 'text-emerald-500';
  }

  get statusBadgeColor() {
    const status = this.flightData()?.status;
    if (status === 'Cancelled') return 'bg-rose-500/10 text-rose-500 border-rose-500/20';
    if (status === 'Delayed') return 'bg-amber-500/10 text-amber-500 border-amber-500/20';
    return 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20';
  }
}
