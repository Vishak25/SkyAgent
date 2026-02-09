
import { Component, input, computed } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-flight-leg-card',
    standalone: true,
    imports: [CommonModule],
    template: `
    <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 h-full flex flex-col relative overflow-hidden">
        <!-- Badge -->
        <div class="absolute top-4 right-4 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
             [class]="statusBadgeClass()">
             {{ status() }}
        </div>

        <div class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">{{ type() }}</div>
        
        <div class="flex items-baseline gap-3 mb-2">
            <div class="text-4xl font-bold text-white">{{ airportCode() }}</div>
        </div>
        <div class="flex items-center gap-4 mb-6 text-sm">
            <div class="text-slate-400">
               {{ actualLabel() }}: <span class="text-white font-mono">{{ actualTime() }}</span>
            </div>
            <div class="text-slate-500 ml-auto text-xs">
               Sched: {{ scheduledTime() }}
            </div>
        </div>

        <div class="mt-auto grid grid-cols-2 gap-4 border-t border-slate-700/50 pt-4">
            <div>
                <div class="text-[10px] uppercase text-slate-500 font-bold mb-1">Terminal</div>
                <div class="text-lg text-white font-medium">{{ terminal() }}</div>
            </div>
            <div>
                <div class="text-[10px] uppercase text-slate-500 font-bold mb-1">{{ secondaryLabel() }}</div>
                <div class="text-lg text-white font-medium">{{ secondaryValue() }}</div>
            </div>
        </div>
    </div>
  `
})
export class FlightLegCardComponent {
    type = input.required<'DEPARTURE' | 'ARRIVAL'>();
    airportCode = input.required<string>();
    scheduledTime = input.required<string>();
    actualTime = input.required<string>();
    actualLabel = input<string>('Actual');
    terminal = input.required<string>();
    // For Departure: Gate. For Arrival: Baggage or Gate.
    secondaryLabel = input.required<string>();
    secondaryValue = input.required<string>();
    status = input<string>('On Time');

    statusBadgeClass = computed(() => {
        const s = (this.status() || '').toLowerCase();
        if (s.includes('cancelled')) {
            return 'bg-rose-900 text-rose-200';
        }
        if (s.includes('arrived') && s.includes('severely')) {
            return 'bg-rose-900 text-rose-200';
        }
        if (s.includes('arrived') && (s.includes('delayed') || s.includes('slight'))) {
            return 'bg-amber-900 text-amber-200';
        }
        if (s.includes('arrived')) {
            // Arrived on time or early
            return 'bg-emerald-900 text-emerald-200';
        }
        if (s.includes('severely')) {
            return 'bg-rose-900 text-rose-200';
        }
        if (s.includes('delayed') || s.includes('slight')) {
            return 'bg-amber-900 text-amber-200';
        }
        // On Time, Early
        return 'bg-emerald-900 text-emerald-200';
    });
}
