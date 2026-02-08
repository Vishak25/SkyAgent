
import { Component, ElementRef, ViewChild, effect, input, OnDestroy, AfterViewInit } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-network-graph',
  standalone: true,
  template: `
    <div class="w-full h-full min-h-[300px] relative bg-slate-900/50 rounded-xl overflow-hidden border border-slate-700/50">
      <div #graphContainer class="w-full h-full absolute inset-0"></div>
      <div class="absolute top-4 left-4 text-xs font-mono text-slate-400 bg-slate-900/80 p-2 rounded border border-slate-700">
        <div class="flex items-center gap-2 mb-1"><span class="w-2 h-2 rounded-full bg-blue-500"></span> Normal Node</div>
        <div class="flex items-center gap-2 mb-1"><span class="w-2 h-2 rounded-full bg-red-500"></span> High Delay Risk</div>
        <div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-yellow-500"></span> Propagation Path</div>
      </div>
    </div>
  `
})
export class NetworkGraphComponent implements OnDestroy, AfterViewInit {
  riskLevel = input.required<number>(); // 0-100
  @ViewChild('graphContainer') graphContainer!: ElementRef;

  private simulation: any;

  constructor() {
    effect(() => {
      const risk = this.riskLevel();
      if (this.graphContainer) {
        this.renderGraph(risk);
      }
    });
  }

  ngAfterViewInit() {
    // Ensure graph renders even if effect ran before view init
    this.renderGraph(this.riskLevel());
  }

  ngOnDestroy() {
    if (this.simulation) this.simulation.stop();
  }

  private renderGraph(risk: number) {
    if (!this.graphContainer) return;
    const element = this.graphContainer.nativeElement;

    // Debug logging
    console.log('Rendering Graph with Risk:', risk);
    console.log('Container Dimensions:', element.clientWidth, element.clientHeight);

    d3.select(element).selectAll('*').remove();

    // Fix: Fallback dimensions if container is 0 height
    const width = element.clientWidth || 400;
    const height = element.clientHeight || 300;

    const svg = d3.select(element)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Simplified US hub network
    const nodes = [
      { id: 'ORD', x: width * 0.5, y: height * 0.4, type: 'hub' }, // Origin
      { id: 'IAD', x: width * 0.8, y: height * 0.5, type: 'dest' }, // Dest
      { id: 'SFO', x: width * 0.1, y: height * 0.4, type: 'hub' },
      { id: 'DEN', x: width * 0.3, y: height * 0.5, type: 'hub' },
      { id: 'DFW', x: width * 0.45, y: height * 0.7, type: 'hub' },
      { id: 'ATL', x: width * 0.7, y: height * 0.7, type: 'hub' },
      { id: 'JFK', x: width * 0.85, y: height * 0.35, type: 'hub' }
    ];

    const links = [
      { source: 'SFO', target: 'DEN' },
      { source: 'DEN', target: 'ORD' }, // Propagation path to origin
      { source: 'ORD', target: 'IAD' }, // The flight
      { source: 'ORD', target: 'DFW' },
      { source: 'ORD', target: 'ATL' },
      { source: 'ORD', target: 'JFK' },
      { source: 'DFW', target: 'ATL' },
      { source: 'ATL', target: 'IAD' }
    ];

    const isHighRisk = risk > 50;

    // Simulation
    this.simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .on('tick', ticked);

    // Draw Links
    const link = svg.append('g')
      .attr('stroke', '#475569')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', (d) => (d.source === 'ORD' && d.target === 'IAD') ? 3 : 1)
      .attr('stroke', (d) => {
        if (isHighRisk && d.source === 'DEN' && d.target === 'ORD') return '#f59e0b'; // Propagation source
        if (isHighRisk && d.source === 'ORD' && d.target === 'IAD') return '#ef4444'; // The flight impacted
        return '#334155';
      });

    // Draw Nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('g');

    node.append('circle')
      .attr('r', (d) => d.id === 'ORD' || d.id === 'IAD' ? 12 : 6)
      .attr('fill', (d) => {
        if (d.id === 'ORD') return isHighRisk ? '#ef4444' : '#3b82f6';
        if (d.id === 'IAD') return '#10b981';
        if (isHighRisk && d.id === 'DEN') return '#f59e0b'; // Problem source
        return '#64748b';
      })
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 2);

    // Add pulse animation for delayed nodes
    if (isHighRisk) {
      node.filter((d: any) => d.id === 'ORD')
        .append('circle')
        .attr('r', 12)
        .attr('fill', 'none')
        .attr('stroke', '#ef4444')
        .attr('stroke-width', 2)
        .append('animate')
        .attr('attributeName', 'r')
        .attr('from', 12)
        .attr('to', 24)
        .attr('dur', '1.5s')
        .attr('repeatCount', 'indefinite')
        .select(function () { return this.parentNode; }) // Go back to circle
        .append('animate')
        .attr('attributeName', 'opacity')
        .attr('from', 1)
        .attr('to', 0)
        .attr('dur', '1.5s')
        .attr('repeatCount', 'indefinite');
    }

    node.append('text')
      .text((d: any) => d.id)
      .attr('x', 15)
      .attr('y', 4)
      .attr('fill', '#cbd5e1')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold');

    function ticked() {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    }
  }
}
