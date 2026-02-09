
import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ItineraryOption } from '../services/flight-predictor.service';

@Component({
  selector: 'app-itinerary-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-slate-800/80 border rounded-xl p-4 transition-all hover:bg-slate-800 cursor-pointer relative overflow-hidden"
         [class.border-emerald-500/40]="itinerary().recommended"
         [class.border-slate-700]="!itinerary().recommended"
         [class.ring-1]="itinerary().recommended"
         [class.ring-emerald-500/20]="itinerary().recommended">

      <!-- Recommended badge -->
      @if (itinerary().recommended) {
      <div class="absolute top-0 right-0 bg-emerald-500 text-[9px] font-bold uppercase text-white px-2 py-0.5 rounded-bl-lg">
        Recommended
      </div>
      }

      <div class="flex items-start justify-between gap-3 mb-3">
        <!-- Route info -->
        <div class="flex-1">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold uppercase tracking-wider" [class]="riskColor()">{{ itinerary().delayRisk }} Risk</span>
            <span class="text-slate-600">|</span>
            <span class="text-xs text-slate-400">{{ itinerary().type === 'direct' ? 'Direct' : '1 Stop' }}</span>
          </div>
          <div class="flex items-center gap-2">
            @for (leg of itinerary().legs; track $index) {
              @if ($index > 0) {
                <div class="flex items-center gap-1">
                  <div class="w-1.5 h-1.5 rounded-full bg-amber-500"></div>
                  <span class="text-[10px] text-amber-400 font-mono">{{ itinerary().connectionHub }}</span>
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-slate-500"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
                </div>
              }
              <span class="text-lg font-bold text-white">{{ leg.origin }}</span>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-slate-500"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
              <span class="text-lg font-bold text-white">{{ leg.destination }}</span>
            }
          </div>
          <div class="text-xs text-slate-500 mt-1">{{ itinerary().airline }}</div>
        </div>

        <!-- Delay prediction -->
        <div class="text-right shrink-0">
          <div class="text-2xl font-mono font-bold" [class]="delayColor()">
            {{ itinerary().predictedDelayMinutes }}<span class="text-xs ml-0.5">min</span>
          </div>
          <div class="text-[10px] text-slate-500 uppercase">Predicted Delay</div>
        </div>
      </div>

      <!-- Risk indicators -->
      <div class="flex gap-3 mt-2 pt-2 border-t border-slate-700/50">
        <div class="flex-1">
          <div class="text-[10px] text-slate-500 uppercase mb-1">Propagation</div>
          <div class="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
            <div class="h-full rounded-full transition-all duration-700" [style.width.%]="itinerary().propagationRisk" [class]="progressColor(itinerary().propagationRisk)"></div>
          </div>
        </div>
        <div class="flex-1">
          <div class="text-[10px] text-slate-500 uppercase mb-1">Precip / Winter Ops</div>
          <div class="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
            <div class="h-full rounded-full transition-all duration-700" [style.width.%]="itinerary().precipSeverity" [class]="progressColor(itinerary().precipSeverity)"></div>
          </div>
        </div>
      </div>
    </div>
  `
})
export class ItineraryCardComponent {
  itinerary = input.required<ItineraryOption>();

  riskColor() {
    switch (this.itinerary().delayRisk) {
      case 'Low': return 'text-emerald-400';
      case 'Moderate': return 'text-amber-400';
      case 'High': return 'text-orange-400';
      case 'Very High': return 'text-rose-400';
      default: return 'text-slate-400';
    }
  }

  delayColor() {
    const d = this.itinerary().predictedDelayMinutes;
    if (d > 45) return 'text-rose-400';
    if (d > 15) return 'text-amber-400';
    return 'text-emerald-400';
  }

  progressColor(v: number) {
    if (v < 30) return 'bg-emerald-500';
    if (v < 60) return 'bg-amber-500';
    return 'bg-rose-500';
  }
}
