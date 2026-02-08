
import { Component, input } from '@angular/core';

@Component({
  selector: 'app-weather-widget',
  standalone: true,
  template: `
    <div class="flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
      <!-- Icon based on condition -->
      <div class="w-10 h-10 rounded-full flex items-center justify-center bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg">
         @if (condition().includes('Rain') || condition().includes('Storm')) {
           <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19c0-1.7-1.3-3-3-3h-11a3 3 0 0 1-3-3c0-2.8 2.2-5 5-5 .4 0 .7 0 1 .1C7.8 4.6 10.9 2 14.5 2c4 0 7.4 2.9 8 6.8 2.4.6 4.2 2.8 4.2 5.2 0 3-2.5 5.2-5.7 5h-3.5"></path><path d="M8 13v8"></path><path d="M12 13v8"></path><path d="M16 13v8"></path></svg>
         } @else if (condition().includes('Clear')) {
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
         } @else {
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19c0-1.7-1.3-3-3-3h-11a3 3 0 0 1-3-3c0-2.8 2.2-5 5-5 .4 0 .7 0 1 .1C7.8 4.6 10.9 2 14.5 2c4 0 7.4 2.9 8 6.8 2.4.6 4.2 2.8 4.2 5.2 0 3-2.5 5.2-5.7 5h-3.5"></path></svg>
         }
      </div>
      <div>
        <div class="text-[10px] uppercase text-slate-500 font-bold tracking-wider">{{ location() }}</div>
        <div class="text-sm font-semibold text-slate-200">{{ temp() }}°F · {{ condition() }}</div>
        <div class="text-xs text-slate-400">Wind: {{ wind() }}kts · Vis: {{ vis() }}</div>
      </div>
    </div>
  `
})
export class WeatherWidgetComponent {
  location = input.required<string>();
  temp = input.required<number>();
  condition = input.required<string>();
  wind = input.required<number>();
  vis = input.required<string>();
}
