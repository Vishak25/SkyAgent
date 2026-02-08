
import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-stat-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 flex flex-col h-full hover:bg-slate-800/80 transition-colors">
      <div class="text-slate-400 text-xs uppercase tracking-wider mb-1">{{ label() }}</div>
      <div class="flex items-end justify-between mt-auto">
        <div class="text-xl font-bold text-white">{{ value() }}</div>
        @if (subtext()) {
          <div class="text-xs" [class]="subtextClass()">{{ subtext() }}</div>
        }
      </div>
      @if (progress()) {
        <div class="w-full bg-slate-700 h-1.5 mt-3 rounded-full overflow-hidden">
          <div class="h-full rounded-full transition-all duration-1000"
               [style.width.%]="progressValue()"
               [class]="progressColor()">
          </div>
        </div>
      }
    </div>
  `
})
export class StatCardComponent {
  label = input.required<string>();
  value = input.required<string | number>();
  subtext = input<string>();
  subtextType = input<'neutral' | 'good' | 'bad' | 'warning'>('neutral');
  progress = input<boolean>(false);
  progressValue = input<number>(0);

  subtextClass() {
    switch (this.subtextType()) {
      case 'good': return 'text-emerald-400';
      case 'bad': return 'text-rose-400';
      case 'warning': return 'text-amber-400';
      default: return 'text-slate-500';
    }
  }

  progressColor() {
    const v = this.progressValue();
    if (v < 40) return 'bg-emerald-500';
    if (v < 70) return 'bg-amber-500';
    return 'bg-rose-500';
  }
}
