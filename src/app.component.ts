
import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FlightPredictorService, FlightScenario, RouteSuggestion } from './services/flight-predictor.service';
import { FlightLegCardComponent } from './components/flight-leg-card.component';
import { NetworkGraphComponent } from './components/network-graph.component';
import { StatCardComponent } from './components/stat-card.component';
import { WeatherWidgetComponent } from './components/weather-widget.component';
import { ItineraryCardComponent } from './components/itinerary-card.component';

type SearchMode = 'track' | 'suggest';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, FormsModule,
    NetworkGraphComponent, StatCardComponent, WeatherWidgetComponent,
    FlightLegCardComponent, ItineraryCardComponent,
  ],
  templateUrl: './app.component.html'
})
export class AppComponent {
  private predictor = inject(FlightPredictorService);

  // Search mode
  searchMode = signal<SearchMode>('track');

  // Track mode state
  flightInput = signal<string>('');
  isLoading = signal<boolean>(false);
  flightData = signal<FlightScenario | null>(null);
  aiAnalysis = signal<string>('');
  errorMessage = signal<string>('');

  // Suggest mode state
  originInput = signal<string>('');
  destInput = signal<string>('');
  dateInput = signal<string>('');
  routeSuggestion = signal<RouteSuggestion | null>(null);
  routeAnalysis = signal<string>('');

  constructor() { }

  setMode(mode: SearchMode) {
    this.searchMode.set(mode);
    this.errorMessage.set('');
  }

  // --- Track mode ---

  async trackFlight() {
    if (!this.flightInput()) return;

    this.isLoading.set(true);
    this.errorMessage.set('');
    this.aiAnalysis.set('');
    this.flightData.set(null);
    this.routeSuggestion.set(null);

    try {
      const data = await this.predictor.getFlightStatus(this.flightInput());
      this.flightData.set(data);
      const explanation = await this.predictor.analyzeScenario(data);
      this.aiAnalysis.set(explanation);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Failed to fetch flight data.');
    } finally {
      this.isLoading.set(false);
    }
  }

  // --- Suggest mode ---

  async suggestRoutes() {
    if (!this.originInput() || !this.destInput()) return;

    this.isLoading.set(true);
    this.errorMessage.set('');
    this.routeAnalysis.set('');
    this.routeSuggestion.set(null);
    this.flightData.set(null);

    try {
      const data = await this.predictor.suggestRoutes(
        this.originInput(), this.destInput(),
        this.dateInput() || undefined
      );
      this.routeSuggestion.set(data);
      const analysis = this.predictor.analyzeRoutes(data);
      this.routeAnalysis.set(analysis);
    } catch (err: any) {
      this.errorMessage.set(err.message || 'Failed to suggest routes.');
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
